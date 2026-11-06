"""Verify completed prediction encoders and extract paired fitting-only views."""

from dataclasses import dataclass

import torch

from ..cifar10 import CIFAR10Binary
from ..embeddings import load_encoder, random_encoder
from ..model import ModelConfig
from .color_audit import verified_sources
from .spatial_embeddings import MODES, paired_embeddings


@dataclass(frozen=True)
class SpatialInputs:
    identity: dict
    indices: tuple
    labels: torch.Tensor
    models: dict
    references: dict


def prepare_sources(run):
    identity, indices, cached = verified_sources(run)
    identity = {key: identity[key] for key in ('version', 'protocol_sha256', 'manifest_sha256',
        'selection_sha256', 'ordered_training_indices_sha256', 'count', 'checkpoints', 'sources', 'runtime')}
    recipe = run.inputs()[0]
    config = ModelConfig(**recipe['config'])
    arm = next(arm for arm in recipe['arms'] if arm['name'] == 'prediction')
    models = {}
    for seed in recipe['training_seeds']:
        key = f'prediction-seed{seed}'
        model, metadata = load_encoder(run.checkpoint(arm, seed), identity['manifest_sha256'])
        if (metadata['checkpoint_sha256'] != identity['checkpoints'][key]
                or metadata['config'] != recipe['config']
                or metadata['completed_steps'] != recipe['total_steps']
                or metadata['training_settings'] != run.settings(recipe, arm, seed)):
            raise ValueError('Spatial encoder differs from the verified completed prediction source')
        models[key] = model
        models[f'random-seed{seed}'] = random_encoder(config, seed)
    spatial_identity = dict(version=1, source_identity=identity, encoder='context',
        sources=list(models), modes=list(MODES), bins=2, batch_size=16,
        shuffle_rule='quadrants-v1 SHA-256 image-index seed; local torch permutation', shuffle_seed=29,
        partition='train', feature_dims={mode: config.embedding_dim * (1 if mode == 'mean' else 4)
                                         for mode in MODES})
    return SpatialInputs(spatial_identity, tuple(indices), cached['pixels'].labels.clone(), models, cached)


@torch.inference_mode()
def extract_fitting(root, inputs):
    """Index the pinned fitting rows once; keep each source's mean control aligned."""
    dataset = CIFAR10Binary(root)
    rows = [dataset[index] for index in inputs.indices]
    images = torch.stack([row[0] for row in rows])
    if not torch.equal(torch.tensor([row[1] for row in rows]), inputs.labels):
        raise ValueError('Spatial image labels differ from the verified fitting selection')
    values = {'pixels': inputs.references['pixels'].features.clone()}
    for key, model in inputs.models.items():
        parts = {mode: [] for mode in MODES}
        for start in range(0, len(images), inputs.identity['batch_size']):
            stop = start + inputs.identity['batch_size']
            batch = paired_embeddings(model, images[start:stop], inputs.indices[start:stop])
            for mode in MODES:
                parts[mode].append(batch[mode])
        combined = {mode: torch.cat(parts[mode]) for mode in MODES}
        if not torch.allclose(combined['mean'], inputs.references[key].features, atol=1e-6, rtol=1e-5):
            raise ValueError('Spatial global mean differs from the verified historical feature control')
        values.update({f'{key}/{mode}': value for mode, value in combined.items()})
    return values
