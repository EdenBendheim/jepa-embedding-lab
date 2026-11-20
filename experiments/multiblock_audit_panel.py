"""Pin aligned original and shuffled features in an ignored local cache."""

import io
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from jepa_lab.analysis.color_statistics import rgb_statistics
from jepa_lab.analysis.spatial_cache import tensor_sha
from jepa_lab.embeddings import image_embeddings
from jepa_lab.probes import _values

from experiments.multiblock_patch_control import shuffle_patches


def _fingerprints(values):
    return {name: tensor_sha(features) for name, features in values.items()}


def _extract(data, sources, batch_size, shuffle_seed):
    parts = {'rgb': []}
    for key in sources.models:
        parts[f'{key}/original'] = []
        parts[f'{key}/shuffled'] = []
    for start in range(0, len(sources.indices), batch_size):
        indices = sources.indices[start:start+batch_size]
        images, labels = data.read(indices)
        if labels.tolist() != [data.labels[index] for index in indices]:
            raise ValueError('Image labels differ from the pinned fitting rows')
        shuffled = shuffle_patches(images, indices, seed=shuffle_seed)
        rgb = rgb_statistics(images)
        if not torch.allclose(rgb, rgb_statistics(shuffled), atol=1e-7, rtol=0):
            raise ValueError('Patch permutation changed the RGB controls')
        parts['rgb'].append(rgb)
        for key, model in sources.models.items():
            parts[f'{key}/original'].append(image_embeddings(model, images))
            parts[f'{key}/shuffled'].append(image_embeddings(model, shuffled))
    return {key: torch.cat(chunks) for key, chunks in parts.items()}


def _read_cache(path, identity, expected, labels):
    with path.open('rb') as handle:
        payload = handle.read(16*1024*1024 + 1)
    if len(payload) > 16*1024*1024:
        raise ValueError('Audit cache exceeds the 16 MiB feature limit')
    state = torch.load(io.BytesIO(payload), weights_only=True, map_location='cpu')
    if (not isinstance(state, dict) or state.get('identity') != identity
            or not isinstance(state.get('features'), dict)
            or state['features'].keys() != expected.keys()
            or not isinstance(state.get('labels'), torch.Tensor)
            or state['labels'].dtype != torch.long or not torch.equal(state['labels'], labels)):
        raise ValueError('Audit cache identity, source set or labels changed')
    values = state['features']
    for key, width in expected.items():
        _values(values[key], dimension=width)
        if len(values[key]) != len(labels):
            raise ValueError('Audit feature rows differ from the declared panel')
    if state.get('feature_sha256') != _fingerprints(values):
        raise ValueError('Audit feature cache content changed')
    return values


def feature_panel(data, sources, path, *, protocol_sha, batch_size, shuffle_seed, replay=False):
    """Read fitting images once. Replay load verifies cache without encoding."""
    if type(batch_size) is not int or not 1 <= batch_size <= 64:
        raise ValueError('Use a feature batch size between 1 and 64')
    path = Path(path)
    expected = {'rgb': 6}
    for key, model in sources.models.items():
        for view in ('original', 'shuffled'):
            expected[f'{key}/{view}'] = model.config.embedding_dim
    labels = torch.tensor([data.labels[index] for index in sources.indices], dtype=torch.long)
    identity = dict(version=1, protocol_sha256=protocol_sha, sources=sources.identity,
                    batch_size=batch_size, shuffle_seed=shuffle_seed)
    if not path.exists():
        if replay:
            raise ValueError('Replay requires the original fitting feature cache')
        features = _extract(data, sources, batch_size, shuffle_seed)
        state = dict(identity=identity, features=features, labels=labels, feature_sha256=_fingerprints(features))
        path.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=path.parent) as temporary:
            staging = Path(temporary)/path.name
            torch.save(state, staging)
            try:
                os.link(staging, path)
            except FileExistsError:
                pass
    return _read_cache(path, identity, expected, labels), labels
