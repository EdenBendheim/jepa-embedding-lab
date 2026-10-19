"""Run the committed position experiment with bounded local training and caches."""
import argparse
from dataclasses import asdict,replace
import hashlib,json
from statistics import mean,stdev
from pathlib import Path

import torch

from jepa_lab.cifar10 import CIFAR10Binary,make_manifest
from jepa_lab.compare import write_report
from jepa_lab.confirmation import score_confirmation
from jepa_lab.confirmation_compare import compare_confirmed_probes
from jepa_lab.embeddings import extract_cifar10
from jepa_lab.features import load_features
from jepa_lab.neighbor_artifacts import select_neighbors,save_neighbors,load_neighbors
from jepa_lab.neighbor_confirmation import score_neighbor_confirmation
from jepa_lab.neighbors import NeighborConfig
from jepa_lab.probes import ProbeConfig
from jepa_lab.position_compare import compare_positions,audit_positions
from jepa_lab.model import ModelConfig,config_dict
from jepa_lab.model_size import model_size
from jepa_lab.selection import excluded_validation_indices,make_confirmation,validate_pilot
from jepa_lab.training_budget import run_training_budget,verify_budget_state

ROOT=Path('data/cifar-10-batches-bin')
PROTOCOL=Path('experiments/position-sources-20261018.json')
PROTOCOL_SHA='7248e3fdb4290f47c148504e714eecd6c95835cf9e6e6bb3831a4d046848bf41'


def inputs(directory):
    if hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()!=PROTOCOL_SHA:raise ValueError('Position protocol changed')
    d=json.loads(PROTOCOL.read_text())
    for row in d['sources']:
        if hashlib.sha256(Path(row['path']).read_bytes()).hexdigest()!=row['sha256']:raise ValueError('Declared position source changed')
    manifest=json.loads(Path(d['manifest_path']).read_text());train,test=CIFAR10Binary(ROOT),CIFAR10Binary(ROOT,train=False)
    if manifest!=make_manifest(train,test,seed=manifest['seed']) or manifest['manifest_sha256']!=d['manifest_sha256']:
        raise ValueError('Position dataset/split differs')
    ref=validate_pilot(json.loads(Path(d['selection_path']).read_text()),manifest,train.labels)
    prior=validate_pilot(json.loads(Path(d['previous_confirmation_path']).read_text()),manifest,train.labels)
    fresh=make_confirmation(manifest,train.labels,ref,exclusions=(prior,),seed=d['confirmation_seed'],validation_per_class=d['confirmation_per_class'])
    if ref['selection_sha256']!=d['selection_sha256'] or fresh['selection_sha256']!=d['confirmation_selection_sha256']:
        raise ValueError('Position tuning/confirmation rows differ')
    path=Path(directory)/'confirmation-selection.json'
    if path.exists() and json.loads(path.read_text())!=fresh:raise ValueError('Position cached confirmation selection differs')
    return d,manifest,ref,fresh,train.labels


def config(d,position):
    return ModelConfig(patch_dim=48,embedding_dim=d['embedding_dim'],heads=d['heads'],encoder_depth=d['encoder_depth'],predictor_depth=d['predictor_depth'],position_encoding=position)


def checkpoint_path(directory,d,position,seed):
    old=d['reused_checkpoints'].get(str(seed)) if position=='learned' else None
    return Path(old) if old else Path(directory)/'checkpoints'/f'position{position}-seed{seed}.pt'


def training(directory=Path('runs/position-20261018'),*,reuse_only=False):
    directory=Path(directory);d,manifest,ref,fresh,labels=inputs(directory)
    for position in d['positions']:
        for seed in d['training_seeds']:
            path=checkpoint_path(directory,d,position,seed)
            if path.exists():
                identity=dict(format_version=1,dataset='CIFAR-10',manifest_sha256=manifest['manifest_sha256'],config=config_dict(config(d,position)),
                    settings=dict(seed=seed,batch_size=d['batch_size'],learning_rate=d['learning_rate'],momentum=d['momentum']),
                    runtime=dict(torch=str(torch.__version__),cpu_threads=torch.get_num_threads()))
                verify_budget_state(path,identity,manifest['partitions']['train']['indices'],d['total_steps'])
    selected=directory/'confirmation-selection.json'
    if not selected.exists():
        if reuse_only:raise ValueError('Reserved position selection is missing')
        write_report(selected,fresh)
    reports={}
    for position in d['positions']:
        for seed in d['training_seeds']:
            name=f'position{position}-seed{seed}';path=checkpoint_path(directory,d,position,seed)
            result=run_training_budget(ROOT,d['manifest_path'],path,total_steps=d['total_steps'],chunk_steps=d['chunk_steps'],
                checkpoint_every=d['checkpoint_every'],batch_size=d['batch_size'],learning_rate=d['learning_rate'],momentum=d['momentum'],
                seed=seed,config=config(d,position),target_rows=d['target_rows'],target_cols=d['target_cols'],context_fraction=d['context_fraction'],
                reuse_only=reuse_only or (position=='learned' and str(seed) in d['reused_checkpoints']))
            result['model_size']=model_size(config(d,position))
            output=directory/'training'/f'{name}.json'
            if output.exists() and json.loads(output.read_text())!=result:raise ValueError('Completed position report changed')
            if not output.exists():write_report(output,result)
            reports[name]=result
            print(f'{name}: {result["completed_steps"]} steps, {result["model_size"]["parameters"]} parameters',flush=True)
    for seed in d['training_seeds']:
        if len({reports[f'position{w}-seed{seed}']['train_image_order_sha256'] for w in d['positions']})!=1:
            raise ValueError('Actual image order differs across positions')
    return reports




def source_key(position,seed,rep):return 'pixels' if rep=='pixels' else f'position{position}-seed{seed}-{rep}'


def feature(directory,d,manifest,selection,labels,position,seed,rep,partition,reuse_only):
    key=source_key(position,seed,rep);suffix='fresh' if selection.get('kind')=='confirmation' else partition
    path=Path(directory)/'features'/f'{key}-{suffix}.pt'
    if not path.exists():
        if reuse_only:raise ValueError('Cached position feature is missing')
        extract_cifar10(ROOT,d['manifest_path'],representation=rep,partition=partition,
            checkpoint=checkpoint_path(directory,d,position,seed) if rep=='checkpoint' else None,
            config=config(d,position) if rep=='random' else None,seed=seed if seed is not None else 29,
            selection_path=Path(directory)/'confirmation-selection.json' if suffix=='fresh' else d['selection_path'],
            pooling=d['pooling'],encoder=d['encoder'],output=path)
    value=load_features(path,manifest=manifest,selection=selection,training_labels=labels,partition=partition)
    if rep!='pixels':
        if value.metadata['config']!=config_dict(config(d,position)) or value.metadata['encoder']!=d['encoder']:
            raise ValueError('Cached position config/encoder differs')
        if rep=='random' and value.metadata['seed']!=seed:raise ValueError('Random initialization control differs')
        if rep=='checkpoint' and value.metadata['checkpoint_sha256']!=hashlib.sha256(checkpoint_path(directory,d,position,seed).read_bytes()).hexdigest():
            raise ValueError('Cached features refer to a different position checkpoint')
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


def evaluation(directory=Path('runs/position-20261015'),*,reuse_only=False):
    directory=Path(directory);d,manifest,ref,fresh,labels=inputs(directory)
    completed=training(directory,reuse_only=True)
    candidates=tuple(ProbeConfig(d['probe_steps'],lr,wd,d['classifier_seeds'][0]) for lr in d['learning_rates'] for wd in d['weight_decays'])
    neighbor_configs=[NeighborConfig(k) for k in d['neighbor_counts']]
    sources={};comparisons={};selected_linear=[];selected_neighbors={};verified=0
    pixel_pair=tuple(feature(directory,d,manifest,ref,labels,None,None,'pixels',p,reuse_only) for p in ('train','validation'))
    sources['pixels']=pixel_pair
    for seed in d['training_seeds']:
        arms={}
        for position in d['positions']:
            data=dict(pixels=pixel_pair)
            for rep in ('checkpoint','random'):
                key=source_key(position,seed,rep)
                pair=tuple(feature(directory,d,manifest,ref,labels,position,seed,rep,p,reuse_only) for p in ('train','validation'))
                data[rep]=pair;sources[key]=pair
            arms[f'position{position}']=data
        path=directory/'comparisons'/f'seed{seed}.json'
        if not path.exists():
            if reuse_only:raise ValueError('Selected position comparison missing')
            report=compare_positions(arms,candidates,d['classifier_seeds'],ridge_alphas=d['ridge_alphas'],probe_dir=directory/'probes'/f'encoder{seed}')
            write_report(path,report)
        report=json.loads(path.read_text());verified+=audit_positions(report,arms,candidates,d['classifier_seeds'],ridge_alphas=d['ridge_alphas'])['verified_classifiers']
        comparisons[str(seed)]=report['summary']
        for run in report['runs']:
            rows=[(None,'pixels',run['pixels'])]+[(w,rep,run['arms'][f'position{w}'][rep]) for w in d['positions'] for rep in ('checkpoint','random')]
            for position,rep,row in rows:
                for head,scored in [('adamw',row)]+([('ridge',row['ridge'])] if 'ridge' in row else []):
                    selected_linear.append(dict(position=position,training_seed=seed,representation=rep,head=head,classifier_seed=run['seed'] if head=='adamw' else None,
                        source_key=source_key(position,seed,rep),selected_config=scored['selected_config'],original_tuning=scored['validation'],
                        path=scored['selected_probe_artifact']['path'],state_sha256=scored['selected_probe_artifact']['state_sha256'],artifact_sha256=scored['selected_probe_artifact']['artifact_sha256']))
    for key,pair in sources.items():
        row=neighbor_record(directory/'neighbors'/f'{key}.json',pair,neighbor_configs,reuse_only)
        selected_neighbors[key]=row
    lock=dict(linear=selected_linear,neighbors=selected_neighbors)
    lock_path=directory/'selection-lock.json'
    if lock_path.exists() and json.loads(lock_path.read_text())!=lock:raise ValueError('Selected position/classifier lock differs')
    if not lock_path.exists():
        if reuse_only:raise ValueError('Position selection lock missing')
        write_report(lock_path,lock)
    # Every learned and neighbor classifier is now selected and locked before fresh pixels are read.
    fresh_sources={'pixels':feature(directory,d,manifest,fresh,labels,None,None,'pixels','validation',reuse_only)}
    for position in d['positions']:
        for seed in d['training_seeds']:
            for rep in ('checkpoint','random'):
                fresh_sources[source_key(position,seed,rep)]=feature(directory,d,manifest,fresh,labels,position,seed,rep,'validation',reuse_only)
    scored_rows=[]
    for row in selected_linear:
        key=row['source_key'];scored=score_confirmation(row['path'],*sources[key],fresh_sources[key],fresh)
        scored_rows.append({k:v for k,v in row.items() if k!='path'}|dict(confirmation=scored['confirmation']))
    for key,row in selected_neighbors.items():
        scored=score_neighbor_confirmation(row['path'],*sources[key],fresh_sources[key],fresh)
        if key=='pixels':position,seed,rep=None,None,'pixels'
        else:
            w,s,rep=key.split('-');position,seed=w[len('position'):],int(s[4:])
        scored_rows.append(dict(position=position,training_seed=seed,representation=rep,head='knn',classifier_seed=None,source_key=key,
            selected_config=row['index']['config'],state_sha256=row['index']['state_sha256'],artifact_sha256=row['index']['artifact_sha256'],
            original_tuning=row['index']['scores']['validation'],confirmation=scored['confirmation']))
    paired_positions=[]
    for seed in d['training_seeds']:
        for head,classifier_seed in [('adamw',value) for value in d['classifier_seeds']]+[('ridge',None)]:
            rows=[next(r for r in selected_linear if r['training_seed']==seed and r['position']==position
                and r['representation']=='checkpoint' and r['head']==head and r['classifier_seed']==classifier_seed)
                for position in d['positions']]
            before,after=rows
            baseline=(*sources[before['source_key']],fresh_sources[before['source_key']])
            candidate=(*sources[after['source_key']],fresh_sources[after['source_key']])
            paired=compare_confirmed_probes(baseline,candidate,before['path'],after['path'],fresh)
            paired_positions.append(dict(training_seed=seed,head=head,classifier_seed=classifier_seed,
                baseline_state_sha256=before['state_sha256'],candidate_state_sha256=after['state_sha256'],
                baseline_accuracy=paired['baseline_accuracy'],candidate_accuracy=paired['candidate_accuracy'],
                paired_confirmation=paired['paired_confirmation']))
    aggregates={}
    for position,rep in [(w,r) for w in d['positions'] for r in ('checkpoint','random')]+[(None,'pixels')]:
        group='pixels' if rep=='pixels' else f'position{position}-{rep}';outcomes={}
        for head in ('adamw','ridge','knn'):
            rows=[r for r in scored_rows if r['position']==position and r['representation']==rep and r['head']==head]
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
    result=dict(version=1,date='2026-10-18',protocol_sha256=PROTOCOL_SHA,manifest_sha256=manifest['manifest_sha256'],
        reference_selection_sha256=ref['selection_sha256'],confirmation_selection_sha256=fresh['selection_sha256'],
        counts=dict(train=len(ref['partitions']['train']['indices']),tuning=len(ref['partitions']['validation']['indices']),confirmation=len(fresh['partitions']['validation']['indices']),excluded_observed_validation=len(excluded)),
        summary=aggregates,paired_positions=paired_positions,training=training_metadata,classifiers=scored_rows,verified_linear_states=verified,verified_neighbor_states=len(selected_neighbors),
        unique_linear_state_count=len({r['state_sha256'] for r in selected_linear}),actual_image_order_matched=True,
        all_states_locked_before_confirmation=True,confirmation_refitting=False,test_evaluated=False,
        limitation='Shared confirmation rows; seed variation is descriptive. Position encoding changes amplitude and trainable capacity at fixed downstream width. Exact-initialization random controls and frozen heads are matched. No winner selected or official test evaluated.')
    write_report(directory/'summary.json',result)
    return result



def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',type=Path,default=Path('runs/position-20261018'))
    p.add_argument('--reuse-only',action='store_true');p.add_argument('--evaluate',action='store_true');args=p.parse_args();torch.set_num_threads(1)
    try:
        if args.evaluate:
            result=evaluation(args.directory,reuse_only=args.reuse_only)
            print(json.dumps(dict(summary=result['summary'],linear=result['verified_linear_states'],knn=result['verified_neighbor_states']),indent=2))
        else:training(args.directory,reuse_only=args.reuse_only)
    except (ValueError,OSError,KeyError,TypeError,RuntimeError) as exc:p.error(str(exc))


if __name__=='__main__':main()
