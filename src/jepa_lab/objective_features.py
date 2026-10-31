"""Verified objective features with one random control per seed and shared pixels."""
from pathlib import Path

from .embeddings import extract_cifar10
from .features import load_features
from .model import ModelConfig
from .objective_run import file_sha
from .pooling import POOLING


def source_specs(d):
    specs = {'pixels': dict(representation='pixels', seed=None, arm=None)}
    for seed in d['training_seeds']:
        specs[f'random-seed{seed}'] = dict(representation='random', seed=seed, arm=None)
        for arm in d['arms']:
            specs[f'{arm["name"]}-seed{seed}'] = dict(representation='checkpoint', seed=seed, arm=arm)
    return specs


def feature(run, d, manifest, selection, labels, key, partition, *, reuse_only=False):
    spec = source_specs(d)[key]
    fresh = selection.get('kind') == 'confirmation'
    if fresh:
        if partition != 'validation':
            raise ValueError('Fresh objective features are validation-only')
        from .objective_observation import observation_gate
        observation_gate(run, d, manifest, selection, labels)
    path = run.directory / 'features' / f'{key}-{"fresh" if fresh else partition}.pt'
    rep, seed, arm = spec['representation'], spec['seed'], spec['arm']
    checkpoint = run.checkpoint(arm, seed) if arm else None
    recipe = d['evaluation']
    if not path.exists():
        if reuse_only:
            raise ValueError('Cached objective feature is missing')
        extract_cifar10(run.root, d['manifest_path'], representation=rep, partition=partition,
            checkpoint=checkpoint, config=ModelConfig(**d['config']) if rep == 'random' else None,
            seed=seed if seed is not None else 29,
            selection_path=run.directory/'confirmation-selection.json' if fresh else d['selection_path'],
            pooling=recipe['pooling'], encoder=recipe['encoder'], output=path)
    value = load_features(path, manifest=manifest, selection=selection, training_labels=labels, partition=partition)
    metadata = value.metadata
    if metadata['representation'] != rep:
        raise ValueError('Cached objective representation differs')
    if rep != 'pixels' and (metadata['config'] != d['config'] or metadata['encoder'] != recipe['encoder']
                           or metadata['pooling'] != POOLING[recipe['pooling']]):
        raise ValueError('Cached objective architecture/encoder/pooling differs')
    if rep == 'random' and metadata['seed'] != seed:
        raise ValueError('Cached exact-initialization random control differs')
    if arm and (metadata['checkpoint_sha256'] != file_sha(checkpoint)
                or metadata['completed_steps'] != d['total_steps']
                or metadata['training_settings'] != run.settings(d, arm, seed)):
        raise ValueError('Cached objective checkpoint, budget or training recipe differs')
    return value


def seed_sources(pairs, seed):
    return dict(prediction=pairs[f'prediction-seed{seed}'],
                context_variance=pairs[f'context_variance-seed{seed}'],
                random=pairs[f'random-seed{seed}'], pixels=pairs['pixels'])
