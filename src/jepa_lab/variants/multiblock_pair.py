"""Train or verify a matched single-tile and four-tile CIFAR-10 pair."""

from dataclasses import asdict
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

from ..compare import write_report
from ..model import ModelConfig
from .multiblock_features import load_cifar_encoder
from .multiblock_train import _digest, _recipe, train_multiblock

ARMS = {
    'one_4x4': dict(blocks=1, target_rows=4, target_cols=4),
    'four_2x2': dict(blocks=4, target_rows=2, target_cols=2),
}


def pair_spec(data, *, steps=32, batch_size=4, seed=29, config=None):
    if type(steps) is not int or not 1 <= steps <= 100:
        raise ValueError('Use 1-100 optimizer updates per arm')
    config = ModelConfig(patch_dim=48) if config is None else config
    settings = dict(batch_size=batch_size, seed=seed, learning_rate=.001, momentum=.99,
                    context_fraction=.75)
    identities = {
        name: _recipe(config=config, training_data=data, **settings, **geometry)[1]
        for name, geometry in ARMS.items()
    }
    return dict(version=1, steps=steps, config=asdict(config), settings=settings, arms=identities)


def _load_arm(data, directory, spec, name):
    model, metadata, journal = load_cifar_encoder(directory/f'{name}.pt', data)
    identity = spec['arms'][name]
    if (metadata['completed_steps'] != spec['steps']
            or any(metadata[key] != value for key, value in identity.items())):
        raise ValueError(f'Checkpoint for {name} differs from the matched budget or recipe')
    return model, metadata, journal


def verify_pair(data, directory, spec):
    directory = Path(directory)
    if json.loads((directory/'pair-spec.json').read_text()) != spec:
        raise ValueError('Matched pair differs from its declared training recipe')
    arms = {name: _load_arm(data, directory, spec, name) for name in ARMS}
    first, second = arms.values()
    initial = first[1]['initial_state_sha256']
    image_order = [row['train_indices'] for row in first[2]]
    if (initial != second[1]['initial_state_sha256']
            or image_order != [row['train_indices'] for row in second[2]]):
        raise ValueError('Arms have different initial weights or observed image order')
    for name, (_, _, journal) in arms.items():
        if any((row.get('context_tokens'), row.get('target_tokens_per_image'), row.get('predictor_passes'))
               != (36, 16, ARMS[name]['blocks']) for row in journal):
            raise ValueError('Arms must match their declared predictor passes and shared token counts')
    receipt = dict(initial_state_sha256=initial, image_order_sha256=_digest(image_order),
                   updates_per_arm=spec['steps'], sampled_rows_per_arm=sum(map(len, image_order)),
                   unique_training_rows=len({index for batch in image_order for index in batch}),
                   checkpoint_sha256={name: arm[1]['checkpoint_sha256'] for name, arm in arms.items()})
    return arms, receipt


def train_pair(data, directory, spec):
    """Preflight existing arms, then train each missing arm at its exact budget."""
    directory = Path(directory)
    config = ModelConfig(**spec['config'])
    common = spec['settings']
    if spec != pair_spec(data, steps=spec['steps'], batch_size=common['batch_size'],
                         seed=common['seed'], config=config):
        raise ValueError('Invalid or stale matched training recipe')
    declaration = directory/'pair-spec.json'
    existing = [name for name in ARMS if (directory/f'{name}.pt').exists()]
    if not declaration.exists():
        if existing:
            raise ValueError('Existing pair checkpoints require their original declaration')
        directory.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=directory) as temporary:
            staged = Path(temporary)/declaration.name
            write_report(staged, spec)
            try:
                os.link(staged, declaration)
            except FileExistsError:
                pass
    if json.loads(declaration.read_text()) != spec:
        raise ValueError('Matched pair differs from its declared training recipe')
    for name in existing:
        _load_arm(data, directory, spec, name)
    for name, geometry in ARMS.items():
        if name not in existing:
            train_multiblock(training_data=data, config=config, steps=spec['steps'],
                checkpoint=directory/f'{name}.pt', **common, **geometry)
    return verify_pair(data, directory, spec)[1]
