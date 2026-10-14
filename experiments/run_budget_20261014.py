"""Run the declared longer-training mask experiment; local artifacts only."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import torch

from jepa_lab.cifar10 import CIFAR10Binary, make_manifest
from jepa_lab.compare import write_report
from jepa_lab.model import ModelConfig
from jepa_lab.selection import make_confirmation, validate_pilot
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


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',type=Path,default=Path('runs/budget-20261014'))
    p.add_argument('--reuse-only',action='store_true');args=p.parse_args();torch.set_num_threads(1)
    try:training(args.directory,reuse_only=args.reuse_only)
    except (ValueError,OSError,KeyError,TypeError,RuntimeError) as exc:p.error(str(exc))

if __name__=='__main__':main()
