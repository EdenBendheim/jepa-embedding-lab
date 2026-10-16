"""Run the committed width experiment with bounded local training and caches."""
import argparse
from dataclasses import asdict,replace
import hashlib,json
from statistics import mean,stdev
from pathlib import Path

import torch

from jepa_lab.cifar10 import CIFAR10Binary,make_manifest
from jepa_lab.compare import write_report
from jepa_lab.confirmation import score_confirmation
from jepa_lab.embeddings import extract_cifar10
from jepa_lab.features import load_features
from jepa_lab.neighbor_artifacts import select_neighbors,save_neighbors,load_neighbors
from jepa_lab.neighbor_confirmation import score_neighbor_confirmation
from jepa_lab.neighbors import NeighborConfig
from jepa_lab.probes import ProbeConfig
from jepa_lab.width_compare import compare_widths,audit_widths
from jepa_lab.model import ModelConfig
from jepa_lab.model_size import model_size
from jepa_lab.selection import excluded_validation_indices,make_confirmation,validate_pilot
from jepa_lab.training_budget import run_training_budget,verify_budget_state

ROOT=Path('data/cifar-10-batches-bin')
PROTOCOL=Path('experiments/width-sources-20261015.json')
PROTOCOL_SHA='60fdad23ce573f364fa5e33329424c5fe5417122a17a50f79b1719a2fa7dae91'


def inputs(directory):
    if hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()!=PROTOCOL_SHA:raise ValueError('Width protocol changed')
    d=json.loads(PROTOCOL.read_text())
    for row in d['sources']:
        if hashlib.sha256(Path(row['path']).read_bytes()).hexdigest()!=row['sha256']:raise ValueError('Declared width source changed')
    manifest=json.loads(Path(d['manifest_path']).read_text());train,test=CIFAR10Binary(ROOT),CIFAR10Binary(ROOT,train=False)
    if manifest!=make_manifest(train,test,seed=manifest['seed']) or manifest['manifest_sha256']!=d['manifest_sha256']:
        raise ValueError('Width dataset/split differs')
    ref=validate_pilot(json.loads(Path(d['selection_path']).read_text()),manifest,train.labels)
    prior=validate_pilot(json.loads(Path(d['previous_confirmation_path']).read_text()),manifest,train.labels)
    fresh=make_confirmation(manifest,train.labels,ref,exclusions=(prior,),seed=d['confirmation_seed'],validation_per_class=d['confirmation_per_class'])
    if ref['selection_sha256']!=d['selection_sha256'] or fresh['selection_sha256']!=d['confirmation_selection_sha256']:
        raise ValueError('Width tuning/confirmation rows differ')
    path=Path(directory)/'confirmation-selection.json'
    if path.exists() and json.loads(path.read_text())!=fresh:raise ValueError('Width cached confirmation selection differs')
    return d,manifest,ref,fresh,train.labels


def config(d,width):
    return ModelConfig(patch_dim=48,embedding_dim=width,heads=d['heads'],encoder_depth=d['encoder_depth'],predictor_depth=d['predictor_depth'])


def checkpoint_path(directory,d,width,seed):
    old=d['reused_checkpoints'].get(str(seed)) if width==32 else None
    return Path(old) if old else Path(directory)/'checkpoints'/f'width{width}-seed{seed}.pt'


def training(directory=Path('runs/width-20261015'),*,reuse_only=False):
    directory=Path(directory);d,manifest,ref,fresh,labels=inputs(directory)
    for width in d['widths']:
        for seed in d['training_seeds']:
            path=checkpoint_path(directory,d,width,seed)
            if path.exists():
                identity=dict(format_version=1,dataset='CIFAR-10',manifest_sha256=manifest['manifest_sha256'],config=asdict(config(d,width)),
                    settings=dict(seed=seed,batch_size=d['batch_size'],learning_rate=d['learning_rate'],momentum=d['momentum']),
                    runtime=dict(torch=str(torch.__version__),cpu_threads=torch.get_num_threads()))
                verify_budget_state(path,identity,manifest['partitions']['train']['indices'],d['total_steps'])
    selected=directory/'confirmation-selection.json'
    if not selected.exists():
        if reuse_only:raise ValueError('Reserved width selection is missing')
        write_report(selected,fresh)
    reports={}
    for width in d['widths']:
        for seed in d['training_seeds']:
            name=f'width{width}-seed{seed}';path=checkpoint_path(directory,d,width,seed)
            result=run_training_budget(ROOT,d['manifest_path'],path,total_steps=d['total_steps'],chunk_steps=d['chunk_steps'],
                checkpoint_every=d['checkpoint_every'],batch_size=d['batch_size'],learning_rate=d['learning_rate'],momentum=d['momentum'],
                seed=seed,config=config(d,width),target_rows=d['target_rows'],target_cols=d['target_cols'],context_fraction=d['context_fraction'],
                reuse_only=reuse_only or (width==32 and str(seed) in d['reused_checkpoints']))
            result['model_size']=model_size(config(d,width))
            output=directory/'training'/f'{name}.json'
            if output.exists() and json.loads(output.read_text())!=result:raise ValueError('Completed width report changed')
            if not output.exists():write_report(output,result)
            reports[name]=result
            print(f'{name}: {result["completed_steps"]} steps, {result["model_size"]["parameters"]} parameters',flush=True)
    for seed in d['training_seeds']:
        if len({reports[f'width{w}-seed{seed}']['train_image_order_sha256'] for w in d['widths']})!=1:
            raise ValueError('Actual image order differs across widths')
    return reports



def source_key(width,seed,rep):return 'pixels' if rep=='pixels' else f'width{width}-seed{seed}-{rep}'


def feature(directory,d,manifest,selection,labels,width,seed,rep,partition,reuse_only):
    key=source_key(width,seed,rep);suffix='fresh' if selection.get('kind')=='confirmation' else partition
    path=Path(directory)/'features'/f'{key}-{suffix}.pt'
    if not path.exists():
        if reuse_only:raise ValueError('Cached width feature is missing')
        extract_cifar10(ROOT,d['manifest_path'],representation=rep,partition=partition,
            checkpoint=checkpoint_path(directory,d,width,seed) if rep=='checkpoint' else None,
            config=config(d,width) if rep=='random' else None,seed=seed if seed is not None else 29,
            selection_path=Path(directory)/'confirmation-selection.json' if suffix=='fresh' else d['selection_path'],
            pooling=d['pooling'],encoder=d['encoder'],output=path)
    value=load_features(path,manifest=manifest,selection=selection,training_labels=labels,partition=partition)
    if rep!='pixels':
        if value.metadata['config']!=asdict(config(d,width)) or value.metadata['encoder']!=d['encoder']:
            raise ValueError('Cached width config/encoder differs')
        if rep=='random' and value.metadata['seed']!=seed:raise ValueError('Random initialization control differs')
        if rep=='checkpoint' and value.metadata['checkpoint_sha256']!=hashlib.sha256(checkpoint_path(directory,d,width,seed).read_bytes()).hexdigest():
            raise ValueError('Cached features refer to a different width checkpoint')
    return value


def neighbor_record(path,pair,candidates,reuse_only):
    index_path=path.with_suffix('.pt')
    if not path.exists():
        if reuse_only:raise ValueError('Selected neighbor report missing')
        index,selected=select_neighbors(*pair,candidates)
        if index_path.exists():
            restored,artifact=load_neighbors(index_path,*pair)
            if restored.config!=index.config or not torch.equal(restored.vectors,index.vectors):raise ValueError('Partial neighbor state differs')
        else:artifact=save_neighbors(index_path,index,*pair)
        write_report(path,dict(selection=selected,index=artifact,path=str(index_path)))
    row=json.loads(path.read_text());index,artifact=load_neighbors(row['path'],*pair)
    options=row['selection']['candidates'];chosen=row['selection']['selected_candidate']
    if ([r['config'] for r in options]!=[asdict(c) for c in candidates]
            or type(chosen) is not int or not 0<=chosen<len(options)
            or chosen!=min(range(len(options)),key=lambda i:(-options[i]['validation']['accuracy'],i))
            or row['index']!=artifact or options[chosen]['config']!=artifact['config']
            or options[chosen]['validation']!=artifact['scores']['validation']):
        raise ValueError('Selected neighbor grid/state/result differs')
    for option,config in zip(options,candidates,strict=True):
        candidate=replace(index,config=config)
        if option['validation']!=candidate.score(pair[1].features,pair[1].labels):
            raise ValueError('Reported neighbor candidate scores do not reproduce')
    return row


def evaluation(directory=Path('runs/width-20261015'),*,reuse_only=False):
    directory=Path(directory);d,manifest,ref,fresh,labels=inputs(directory)
    completed=training(directory,reuse_only=True)
    candidates=tuple(ProbeConfig(d['probe_steps'],lr,wd,d['classifier_seeds'][0]) for lr in d['learning_rates'] for wd in d['weight_decays'])
    neighbor_configs=[NeighborConfig(k) for k in d['neighbor_counts']]
    sources={};comparisons={};selected_linear=[];selected_neighbors={};verified=0
    pixel_pair=tuple(feature(directory,d,manifest,ref,labels,None,None,'pixels',p,reuse_only) for p in ('train','validation'))
    sources['pixels']=pixel_pair
    for seed in d['training_seeds']:
        arms={}
        for width in d['widths']:
            data=dict(pixels=pixel_pair)
            for rep in ('checkpoint','random'):
                key=source_key(width,seed,rep)
                pair=tuple(feature(directory,d,manifest,ref,labels,width,seed,rep,p,reuse_only) for p in ('train','validation'))
                data[rep]=pair;sources[key]=pair
            arms[f'width{width}']=data
        path=directory/'comparisons'/f'seed{seed}.json'
        if not path.exists():
            if reuse_only:raise ValueError('Selected width comparison missing')
            report=compare_widths(arms,candidates,d['classifier_seeds'],ridge_alphas=d['ridge_alphas'],probe_dir=directory/'probes'/f'encoder{seed}')
            write_report(path,report)
        report=json.loads(path.read_text());verified+=audit_widths(report,arms,candidates,d['classifier_seeds'],ridge_alphas=d['ridge_alphas'])['verified_classifiers']
        comparisons[str(seed)]=report['summary']
        for run in report['runs']:
            rows=[(None,'pixels',run['pixels'])]+[(w,rep,run['arms'][f'width{w}'][rep]) for w in d['widths'] for rep in ('checkpoint','random')]
            for width,rep,row in rows:
                for head,scored in [('adamw',row)]+([('ridge',row['ridge'])] if 'ridge' in row else []):
                    selected_linear.append(dict(width=width,training_seed=seed,representation=rep,head=head,classifier_seed=run['seed'] if head=='adamw' else None,
                        source_key=source_key(width,seed,rep),selected_config=scored['selected_config'],original_tuning=scored['validation'],
                        path=scored['selected_probe_artifact']['path'],state_sha256=scored['selected_probe_artifact']['state_sha256'],artifact_sha256=scored['selected_probe_artifact']['artifact_sha256']))
    for key,pair in sources.items():
        row=neighbor_record(directory/'neighbors'/f'{key}.json',pair,neighbor_configs,reuse_only)
        selected_neighbors[key]=row
    lock=dict(linear=selected_linear,neighbors=selected_neighbors)
    lock_path=directory/'selection-lock.json'
    if lock_path.exists() and json.loads(lock_path.read_text())!=lock:raise ValueError('Selected width/classifier lock differs')
    if not lock_path.exists():
        if reuse_only:raise ValueError('Width selection lock missing')
        write_report(lock_path,lock)
    # Every learned and neighbor classifier is now selected and locked before fresh pixels are read.
    fresh_sources={'pixels':feature(directory,d,manifest,fresh,labels,None,None,'pixels','validation',reuse_only)}
    for width in d['widths']:
        for seed in d['training_seeds']:
            for rep in ('checkpoint','random'):
                fresh_sources[source_key(width,seed,rep)]=feature(directory,d,manifest,fresh,labels,width,seed,rep,'validation',reuse_only)
    scored_rows=[]
    for row in selected_linear:
        key=row['source_key'];scored=score_confirmation(row['path'],*sources[key],fresh_sources[key],fresh)
        scored_rows.append({k:v for k,v in row.items() if k!='path'}|dict(confirmation=scored['confirmation']))
    for key,row in selected_neighbors.items():
        scored=score_neighbor_confirmation(row['path'],*sources[key],fresh_sources[key],fresh)
        if key=='pixels':width,seed,rep=None,None,'pixels'
        else:
            w,s,rep=key.split('-');width,seed=int(w[5:]),int(s[4:])
        scored_rows.append(dict(width=width,training_seed=seed,representation=rep,head='knn',classifier_seed=None,source_key=key,
            selected_config=row['index']['config'],state_sha256=row['index']['state_sha256'],artifact_sha256=row['index']['artifact_sha256'],
            original_tuning=row['index']['scores']['validation'],confirmation=scored['confirmation']))
    aggregates={}
    for width,rep in [(w,r) for w in d['widths'] for r in ('checkpoint','random')]+[(None,'pixels')]:
        group='pixels' if rep=='pixels' else f'width{width}-{rep}';outcomes={}
        for head in ('adamw','ridge','knn'):
            rows=[r for r in scored_rows if r['width']==width and r['representation']==rep and r['head']==head]
            if rep=='pixels':
                unique={}
                for row in rows:
                    key=row['classifier_seed'];signature=(row['state_sha256'],row['original_tuning'],row['confirmation'])
                    if key in unique and unique[key][1]!=signature:raise ValueError('Shared pixel states/results differ')
                    unique[key]=(row,signature)
                rows=[value[0] for value in unique.values()]
            scores={}
            groups=[rows] if rep=='pixels' else [[r for r in rows if r['training_seed']==s] for s in d['training_seeds']]
            for phase in ('original_tuning','confirmation'):
                values=[mean(r[phase]['accuracy'] for r in group_rows) for group_rows in groups]
                scores[phase]=dict(encoder_means=values,mean_accuracy=mean(values),encoder_mean_sample_sd=stdev(values) if len(values)>1 else 0.)
            outcomes[head]=scores
        aggregates[group]=outcomes
    training_metadata={key:{k:v for k,v in row.items() if k not in ('history','limitation')} for key,row in completed.items()}
    excluded=set().union(*(excluded_validation_indices(p) for p in [ref,*fresh['excluded_selections']]))
    result=dict(version=1,date='2026-10-15',protocol_sha256=PROTOCOL_SHA,manifest_sha256=manifest['manifest_sha256'],
        reference_selection_sha256=ref['selection_sha256'],confirmation_selection_sha256=fresh['selection_sha256'],
        counts=dict(train=len(ref['partitions']['train']['indices']),tuning=len(ref['partitions']['validation']['indices']),confirmation=len(fresh['partitions']['validation']['indices']),excluded_observed_validation=len(excluded)),
        summary=aggregates,training=training_metadata,classifiers=scored_rows,verified_linear_states=verified,verified_neighbor_states=len(selected_neighbors),
        unique_linear_state_count=len({r['state_sha256'] for r in selected_linear}),actual_image_order_matched=True,
        all_states_locked_before_confirmation=True,confirmation_refitting=False,test_evaluated=False,
        limitation='Shared validation data; seed variation is descriptive. Width changes compute/capacity and downstream dimensions; random controls use the exact training initialization seed. Confirmation cannot select a width winner.')
    write_report(directory/'summary.json',result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',type=Path,default=Path('runs/width-20261015'))
    p.add_argument('--reuse-only',action='store_true');p.add_argument('--evaluate',action='store_true');args=p.parse_args();torch.set_num_threads(1)
    try:
        if args.evaluate:
            result=evaluation(args.directory,reuse_only=args.reuse_only)
            print(json.dumps(dict(summary=result['summary'],linear=result['verified_linear_states'],knn=result['verified_neighbor_states']),indent=2))
        else:training(args.directory,reuse_only=args.reuse_only)
    except (ValueError,OSError,KeyError,TypeError,RuntimeError) as exc:p.error(str(exc))

if __name__=='__main__':main()
