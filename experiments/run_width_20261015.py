"""Run the committed width experiment with bounded local training and caches."""
import argparse
from dataclasses import asdict
import hashlib,json
from pathlib import Path

import torch

from jepa_lab.cifar10 import CIFAR10Binary,make_manifest
from jepa_lab.compare import write_report
from jepa_lab.model import ModelConfig
from jepa_lab.model_size import model_size
from jepa_lab.selection import make_confirmation,validate_pilot
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


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',type=Path,default=Path('runs/width-20261015'))
    p.add_argument('--reuse-only',action='store_true');args=p.parse_args();torch.set_num_threads(1)
    try:training(args.directory,reuse_only=args.reuse_only)
    except (ValueError,OSError,KeyError,TypeError,RuntimeError) as exc:p.error(str(exc))

if __name__=='__main__':main()
