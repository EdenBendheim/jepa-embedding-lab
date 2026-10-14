"""Run the declared longer-training mask experiment; local artifacts only."""
import argparse
from dataclasses import asdict
from statistics import mean,stdev
import hashlib
import json
from pathlib import Path

import torch

from jepa_lab.cifar10 import CIFAR10Binary, make_manifest
from jepa_lab.compare import compare_repeated_features,write_report
from jepa_lab.confirmation import score_confirmation
from jepa_lab.embeddings import extract_cifar10
from jepa_lab.features import load_features
from jepa_lab.probe_audit import audit_repeated_report
from jepa_lab.probes import ProbeConfig
from jepa_lab.model import ModelConfig
from jepa_lab.selection import excluded_validation_indices,make_confirmation, validate_pilot
from jepa_lab.training_budget import run_training_budget, verify_budget_state

ROOT=Path('data/cifar-10-batches-bin')
PROTOCOL=Path('experiments/budget-sources-20261014.json')
PROTOCOL_SHA='819d7a829ce3f64f2f1ccdb1689427a90972219d95e81605c5d77327a2e93acf'


def inputs(directory):
    if hashlib.sha256(PROTOCOL.read_bytes()).hexdigest() != PROTOCOL_SHA:
        raise ValueError('Declared budget protocol changed')
    declared=json.loads(PROTOCOL.read_text())
    for source in declared['sources']:
        if hashlib.sha256(Path(source['path']).read_bytes()).hexdigest()!=source['sha256']:
            raise ValueError('Declared historical source changed')
    manifest=json.loads(Path(declared['manifest_path']).read_text())
    train,test=CIFAR10Binary(ROOT),CIFAR10Binary(ROOT,train=False)
    if manifest!=make_manifest(train,test,seed=manifest['seed']) or manifest['manifest_sha256']!=declared['manifest_sha256']:
        raise ValueError('Fixed data/split changed')
    reference=validate_pilot(json.loads(Path(declared['selection_path']).read_text()),manifest,train.labels)
    previous=validate_pilot(json.loads(Path(declared['previous_confirmation_path']).read_text()),manifest,train.labels)
    fresh=make_confirmation(manifest,train.labels,reference,exclusions=(previous,),
        seed=declared['confirmation_seed'],validation_per_class=declared['confirmation_per_class'])
    if reference['selection_sha256']!=declared['selection_sha256'] or fresh['selection_sha256']!=declared['confirmation_selection_sha256']:
        raise ValueError('Declared evaluation selection changed')
    path=Path(directory)/'confirmation-selection.json'
    if path.exists() and json.loads(path.read_text())!=fresh:raise ValueError('Cached reserved examples changed')
    return declared,manifest,reference,fresh,train.labels


def recipe(declared,seed,size):
    return dict(total_steps=declared['total_steps'],chunk_steps=declared['chunk_steps'],
        checkpoint_every=declared['checkpoint_every'],batch_size=declared['batch_size'],
        learning_rate=declared['learning_rate'],momentum=declared['momentum'],seed=seed,
        config=ModelConfig(**declared['config']),target_rows=size,target_cols=size,
        context_fraction=declared['context_fraction'])


def checkpoint_path(directory,size,seed):return Path(directory)/'checkpoints'/f'mask{size}-seed{seed}.pt'


def training(directory=Path('runs/budget-20261014'),*,reuse_only=False):
    directory=Path(directory)
    declared,manifest,reference,fresh,labels=inputs(directory)
    # Inspect all existing journals before extending any trajectory.
    for size in declared['target_sizes']:
        for seed in declared['training_seeds']:
            path=checkpoint_path(directory,size,seed)
            if not path.exists():continue
            settings={k:declared[k] for k in ('batch_size','learning_rate','momentum')}|dict(seed=seed)
            if size!=2:settings['mask']=dict(target_rows=size,target_cols=size,context_fraction=declared['context_fraction'])
            identity=dict(format_version=1,dataset='CIFAR-10',manifest_sha256=manifest['manifest_sha256'],
                config=declared['config'],settings=settings,runtime=dict(torch=str(torch.__version__),cpu_threads=torch.get_num_threads()))
            verify_budget_state(path,identity,manifest['partitions']['train']['indices'],declared['total_steps'])
    if not (directory/'confirmation-selection.json').exists():
        if reuse_only:raise ValueError('Reserved selection is missing')
        write_report(directory/'confirmation-selection.json',fresh)
    reports={}
    for size in declared['target_sizes']:
        for seed in declared['training_seeds']:
            name=f'mask{size}-seed{seed}'
            result=run_training_budget(ROOT,declared['manifest_path'],checkpoint_path(directory,size,seed),
                                      **recipe(declared,seed,size),reuse_only=reuse_only)
            path=directory/'training'/f'{name}.json'
            if path.exists() and json.loads(path.read_text())!=result:raise ValueError('Cached completed training report changed')
            if not path.exists():write_report(path,result)
            reports[name]=result
            print(f'{name}: {result["completed_steps"]} steps; {result["unique_training_images"]} unique training images',flush=True)
    for seed in declared['training_seeds']:
        orders=[reports[f'mask{size}-seed{seed}']['train_image_order_sha256'] for size in declared['target_sizes']]
        if len(set(orders))!=1:raise ValueError('Mask comparisons differ in actual image order')
    return reports



def feature_source(directory,name,representation,partition,declared,manifest,selection,labels,checkpoint,random_seed,reuse_only):
    fresh=selection.get('kind')=='confirmation'
    path=Path(directory)/'features'/name/f'{representation}-{"fresh" if fresh else partition}.pt'
    selection_path=Path(directory)/'confirmation-selection.json' if fresh else Path(declared['selection_path'])
    if not path.exists():
        if reuse_only:raise ValueError('A cached feature arm is missing')
        extract_cifar10(ROOT,declared['manifest_path'],representation=representation,partition=partition,
            selection_path=selection_path,checkpoint=checkpoint if representation=='checkpoint' else None,
            config=ModelConfig(**declared['config']) if representation=='random' else None,
            seed=random_seed,pooling=declared['pooling'],encoder=declared['encoder'],output=path)
    item=load_features(path,manifest=manifest,selection=selection,training_labels=labels,partition=partition)
    if representation=='checkpoint' and item.metadata['checkpoint_sha256']!=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest():
        raise ValueError('Cached learned features refer to a different checkpoint')
    if representation=='random' and (item.metadata['seed']!=random_seed or item.metadata['config']!=declared['config']):
        raise ValueError('Cached random encoder differs from the declared paired seed/config')
    return item


def evaluation(directory=Path('runs/budget-20261014'),*,reuse_only=False):
    directory=Path(directory)
    declared,manifest,reference,fresh,labels=inputs(directory)
    completed=training(directory,reuse_only=True)
    candidates=tuple(ProbeConfig(declared['probe_steps'],lr,wd,declared['classifier_seeds'][0])
        for lr in declared['learning_rates'] for wd in declared['weight_decays'])
    reports={};sources={};verified=0;lock=[]
    # Finish all selection and reproduce old scores before reading any fresh pixels.
    for size in declared['target_sizes']:
        for seed,random_seed in zip(declared['training_seeds'],declared['random_seeds'],strict=True):
            name=f'mask{size}-seed{seed}';checkpoint=checkpoint_path(directory,size,seed)
            sets={rep:tuple(feature_source(directory,name,rep,part,declared,manifest,reference,labels,
                checkpoint,random_seed,reuse_only) for part in ('train','validation')) for rep in ('checkpoint','random','pixels')}
            path=directory/'comparisons'/f'{name}.json'
            if not path.exists():
                if reuse_only:raise ValueError('Cached comparison missing')
                report=compare_repeated_features(sets,candidates,declared['classifier_seeds'],
                    ridge_alphas=declared['ridge_alphas'],probe_dir=directory/'probes'/name)
                write_report(path,report)
            report=json.loads(path.read_text())
            verified+=audit_repeated_report(report,sets,candidates,declared['classifier_seeds'],ridge_alphas=declared['ridge_alphas'])['verified_classifiers']
            for classifier_seed,run in zip(declared['classifier_seeds'],report['runs'],strict=True):
                for rep,row in run['representations'].items():
                    for head,selected in [('adamw',row)]+([('ridge',row['ridge'])] if 'ridge' in row else []):
                        lock.append(dict(name=name,representation=rep,head=head,classifier_seed=classifier_seed if head=='adamw' else None,
                            state_sha256=selected['selected_probe_artifact']['state_sha256'],
                            artifact_sha256=selected['selected_probe_artifact']['artifact_sha256'],selected_config=selected['selected_config']))
            reports[name]=report;sources[name]=sets
    lock_path=directory/'selection-lock.json'
    if lock_path.exists() and json.loads(lock_path.read_text())!=lock:raise ValueError('Selected classifier lock changed')
    if not lock_path.exists():
        if reuse_only:raise ValueError('Selected classifier lock is missing')
        write_report(lock_path,lock)
    records=[];rows=[]
    for size in declared['target_sizes']:
        for seed,random_seed in zip(declared['training_seeds'],declared['random_seeds'],strict=True):
            name=f'mask{size}-seed{seed}';checkpoint=checkpoint_path(directory,size,seed);sets=sources[name]
            current={rep:feature_source(directory,name,rep,'validation',declared,manifest,fresh,labels,
                checkpoint,random_seed,reuse_only) for rep in sets}
            scores={rep:dict(tuning=[],confirmation=[]) for rep in sets}
            for classifier_seed,run in zip(declared['classifier_seeds'],reports[name]['runs'],strict=True):
                for rep,row in run['representations'].items():
                    for head,selected in [('adamw',row)]+([('ridge',row['ridge'])] if 'ridge' in row else []):
                        result=score_confirmation(selected['selected_probe_artifact']['path'],*sets[rep],current[rep],fresh)
                        if head=='adamw':
                            scores[rep]['tuning'].append(selected['validation']['accuracy'])
                            scores[rep]['confirmation'].append(result['confirmation']['accuracy'])
                        else:
                            scores[rep]['ridge_tuning']=selected['validation']['accuracy']
                            scores[rep]['ridge_confirmation']=result['confirmation']['accuracy']
                        rows.append(dict(name=name,representation=rep,head=head,classifier_seed=classifier_seed if head=='adamw' else None,
                            selected_config=selected['selected_config'],state_sha256=selected['selected_probe_artifact']['state_sha256'],
                            original_tuning_accuracy=selected['validation']['accuracy'],confirmation=result['confirmation'],
                            original_scores_reproduced=True,classifier_refitted=False,normalization_refitted=False))
            records.append(dict(mask_size=size,training_seed=seed,random_encoder_seed=random_seed,scores=scores,
                training={k:v for k,v in completed[name].items() if k not in ('history','limitation')},
                training_report_sha256=hashlib.sha256((directory/'training'/f'{name}.json').read_bytes()).hexdigest()))
    for records_for_seed in ([r for r in records if r['training_seed']==seed] for seed in declared['training_seeds']):
        if any(r['scores']['random']!=records_for_seed[0]['scores']['random'] for r in records_for_seed):
            raise ValueError('Matched random-baseline scores differ between masks')
    if any(r['scores']['pixels']!=records[0]['scores']['pixels'] for r in records):
        raise ValueError('Shared pixel-baseline scores differ between model runs')
    aggregates={}
    groups={f'mask{size}':(size,'checkpoint') for size in declared['target_sizes']}
    groups|={'random':(declared['target_sizes'][0],'random'),'pixels':(declared['target_sizes'][0],'pixels')}
    for group,(size,rep) in groups.items():
        chosen=[r for r in records if r['mask_size']==size]
        if rep=='pixels':chosen=chosen[:1]  # Shared pixels are one baseline, not independent encoder runs.
        aggregate={}
        for partition in ('tuning','confirmation'):
            means=[mean(r['scores'][rep][partition]) for r in chosen]
            ridge=[r['scores'][rep]['ridge_'+partition] for r in chosen]
            aggregate[partition]=dict(encoder_means=means,mean_accuracy=mean(means),
                encoder_mean_sample_sd=stdev(means) if len(means)>1 else 0.,ridge_accuracies=ridge,ridge_mean_accuracy=mean(ridge))
        aggregates[group]=aggregate
    output=dict(version=1,date='2026-10-14',protocol_sha256=PROTOCOL_SHA,manifest_sha256=manifest['manifest_sha256'],
        tuning_selection_sha256=reference['selection_sha256'],confirmation_selection_sha256=fresh['selection_sha256'],
        counts=dict(train=len(reference['partitions']['train']['indices']),tuning=len(reference['partitions']['validation']['indices']),
            confirmation=len(fresh['partitions']['validation']['indices']),excluded_observed_validation=
            len(set().union(*(excluded_validation_indices(p) for p in [reference,*fresh['excluded_selections']])))),
        summary=aggregates,records=records,classifiers=rows,verified_classifiers=verified,
        actual_image_order_matched=True,classifier_refitted_on_confirmation=False,normalization_refitted_on_confirmation=False,
        test_evaluated=False,limitation='Small fixed-data pilot; encoder-seed SD is descriptive, classifiers share data, mask token/compute budgets differ. No official test result or selected mask winner.')
    write_report(directory/'summary.json',output)
    return output


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',type=Path,default=Path('runs/budget-20261014'))
    p.add_argument('--reuse-only',action='store_true');p.add_argument('--evaluate',action='store_true');args=p.parse_args();torch.set_num_threads(1)
    try:
        if args.evaluate:
            result=evaluation(args.directory,reuse_only=args.reuse_only)
            print(json.dumps(dict(summary=result['summary'],verified_classifiers=result['verified_classifiers']),indent=2))
        else:training(args.directory,reuse_only=args.reuse_only)
    except (ValueError,OSError,KeyError,TypeError,RuntimeError) as exc:p.error(str(exc))

if __name__=='__main__':main()
