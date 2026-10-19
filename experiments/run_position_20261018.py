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




def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',type=Path,default=Path('runs/position-20261018'))
    p.add_argument('--reuse-only',action='store_true');args=p.parse_args();torch.set_num_threads(1)
    try:training(args.directory,reuse_only=args.reuse_only)
    except (ValueError,OSError,KeyError,TypeError,RuntimeError) as exc:p.error(str(exc))


if __name__=='__main__':main()
