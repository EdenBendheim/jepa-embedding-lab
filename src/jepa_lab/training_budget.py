"""Finish a declared local training budget in bounded, resumable chunks."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random

import torch

from .cifar10 import CIFAR10Binary, make_manifest
from .cifar10_train import DEFAULT_MASK, checkpoint_history, train_cifar10
from .compare import write_report
from .model import ModelConfig
from .model_size import validate_local_config
from .model_cli import add_model_options,config_from_args


def verify_budget_state(path, identity, indices, total_steps):
    state = torch.load(path, map_location='cpu', weights_only=True)
    if (not isinstance(state, dict) or any(state.get(k) != v for k, v in identity.items())
            or type(state.get('step')) is not int or not 1 <= state['step'] <= total_steps):
        raise ValueError('Budget checkpoint recipe, runtime or completed step differs')
    start, history = checkpoint_history(state)
    if start != 0:
        raise ValueError('Budget runner requires a complete journal from step zero')
    generator = random.Random(identity['settings']['seed'])
    for row in history:
        if row['train_indices'] != generator.sample(indices, identity['settings']['batch_size']):
            raise ValueError('Budget checkpoint actual image order differs from its declared seed')
    if state.get('sample_rng') != generator.getstate():
        raise ValueError('Budget checkpoint sampling RNG differs from its journal')
    return state


def run_training_budget(root, manifest_path, checkpoint, *, total_steps=2223, chunk_steps=1000,
                        checkpoint_every=50, batch_size=4, learning_rate=0.001, momentum=0.99,
                        seed=None, config=None, target_rows=2, target_cols=2, context_fraction=0.75,
                        reuse_only=False):
    if (type(total_steps) is not int or not 1 <= total_steps <= 10000
            or type(chunk_steps) is not int or not 1 <= chunk_steps <= 1000
            or type(checkpoint_every) is not int or not 1 <= checkpoint_every <= 1000):
        raise ValueError('Budget is 1-10000 total steps; chunks and save intervals are 1-1000')
    manifest = json.loads(Path(manifest_path).read_text())
    train, test = CIFAR10Binary(root), CIFAR10Binary(root, train=False)
    if manifest != make_manifest(train, test, seed=manifest['seed']):
        raise ValueError('Budget manifest differs from the fixed dataset')
    config = validate_local_config(config or ModelConfig(patch_dim=48))
    seed = manifest['seed'] if seed is None else seed
    settings = dict(seed=seed, batch_size=batch_size, learning_rate=learning_rate, momentum=momentum)
    mask = dict(target_rows=target_rows, target_cols=target_cols, context_fraction=float(context_fraction))
    if mask != DEFAULT_MASK: settings['mask'] = mask
    identity = dict(format_version=1, dataset='CIFAR-10', manifest_sha256=manifest['manifest_sha256'],
                    config=asdict(config), settings=settings,
                    runtime=dict(torch=str(torch.__version__), cpu_threads=torch.get_num_threads()))
    path = Path(checkpoint)
    indices = manifest['partitions']['train']['indices']
    state = verify_budget_state(path, identity, indices, total_steps) if path.exists() else None
    completed = 0 if state is None else state['step']
    if reuse_only and completed != total_steps:
        raise ValueError('Completed budget checkpoint is missing; reuse-only cannot train')
    while completed < total_steps:
        count = min(chunk_steps, total_steps-completed)
        train_cifar10(root, manifest_path, steps=count, batch_size=batch_size, learning_rate=learning_rate,
                      momentum=momentum, seed=seed, config=config, **mask,
                      checkpoint=path, resume=path if state is not None else None,
                      checkpoint_every=checkpoint_every)
        state = verify_budget_state(path, identity, indices, total_steps)
        completed = state['step']
    history = state['training_history']
    order = [row['train_indices'] for row in history]
    return identity | dict(total_steps=total_steps, completed_steps=completed, checkpoint_sha256=
        hashlib.sha256(path.read_bytes()).hexdigest(), history=history,
        train_image_order_sha256=hashlib.sha256(json.dumps(order, separators=(',', ':')).encode()).hexdigest(),
        sample_draws=total_steps*batch_size, unique_training_images=len({i for row in order for i in row}),
        test_evaluated=False, limitation='Bounded local training; finite loss is not downstream quality evidence')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path); parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--total-steps', type=int, default=2223)
    parser.add_argument('--chunk-steps', type=int, default=1000)
    parser.add_argument('--checkpoint-every', type=int, default=50)
    parser.add_argument('--seed', type=int); parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--target-rows', type=int, default=2); parser.add_argument('--target-cols', type=int, default=2)
    add_model_options(parser)
    parser.add_argument('--reuse-only', action='store_true')
    parser.add_argument('--output', type=Path)
    args=parser.parse_args(); torch.set_num_threads(1)
    try:
        result=run_training_budget(args.root, args.manifest, args.checkpoint, total_steps=args.total_steps,
            chunk_steps=args.chunk_steps, checkpoint_every=args.checkpoint_every, seed=args.seed,
            batch_size=args.batch_size, target_rows=args.target_rows, target_cols=args.target_cols,
            config=config_from_args(args),
            reuse_only=args.reuse_only)
        if args.output: write_report(args.output, result)
    except (ValueError, OSError, KeyError, TypeError, RuntimeError) as exc: parser.error(str(exc))
    print(json.dumps({k:v for k,v in result.items() if k != 'history'}, indent=2))


if __name__ == '__main__': main()
