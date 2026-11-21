"""Cache the aligned global and regional feature panel from training rows."""

import os
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from experiments.multiblock_audit_panel import _fingerprints, _read_cache
from experiments.multiblock_spatial_readout import encoder_views, rgb_views


IMAGE_VIEWS = ('original', 'patch_shuffle')
READOUTS = ('mean', 'quadrants')


def _extract(data, sources, expected, batch_size, patch_seed):
    parts = {key: [] for key in expected}
    for start in range(0, len(sources.indices), batch_size):
        indices = sources.indices[start:start+batch_size]
        images, labels = data.read(indices)
        if labels.tolist() != [data.labels[index] for index in indices]:
            raise ValueError('Observed labels differ from the fixed fitting selection')
        views = {'rgb': rgb_views(images, indices, patch_seed=patch_seed)}
        for name, model in sources.models.items():
            views[name] = encoder_views(model, images, indices, patch_seed=patch_seed)
        for name, image_views in views.items():
            for view, readouts in image_views.items():
                for readout, rows in readouts.items():
                    parts[f'{name}/{view}/{readout}'].append(rows)
    return {name: torch.cat(parts) for name, parts in parts.items()}


def spatial_panel(data, sources, path, *, protocol_sha, batch_size, patch_seed, replay=False):
    """Reuse exact cached features or index the ordered fitting selection once."""
    if type(batch_size) is not int or not 1 <= batch_size <= 32:
        raise ValueError('Use a spatial extraction batch size from 1 to 32')
    path = Path(path)
    widths = {'rgb': 6} | {name: model.config.embedding_dim for name, model in sources.models.items()}
    expected = {
        f'{name}/{view}/{readout}': width*factor
        for name, width in widths.items()
        for view in IMAGE_VIEWS
        for readout, factor in (('mean', 1), ('quadrants', 4))
    }
    labels = torch.tensor([data.labels[index] for index in sources.indices], dtype=torch.long)
    identity = dict(version=1, protocol_sha256=protocol_sha, sources=sources.identity,
                    row_count=len(sources.indices), batch_size=batch_size, patch_seed=patch_seed)
    if not path.exists():
        if replay:
            raise ValueError('Spatial replay requires the original feature panel')
        values = _extract(data, sources, expected, batch_size, patch_seed)
        state = dict(identity=identity, labels=labels, features=values, feature_sha256=_fingerprints(values))
        path.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=path.parent) as directory:
            stage = Path(directory)/path.name
            torch.save(state, stage)
            try:
                os.link(stage, path)
            except FileExistsError:
                pass
    features = _read_cache(path, identity, expected, labels)
    original, shuffled = (features[f'rgb/{view}/mean'] for view in IMAGE_VIEWS)
    if not torch.allclose(original, shuffled, atol=1e-7, rtol=0):
        raise ValueError('Spatial cache disagrees with the preserved global RGB control')
    for name in sources.models:
        for view in IMAGE_VIEWS:
            mean = features[f'{name}/{view}/mean']
            cells = features[f'{name}/{view}/quadrants'].reshape(len(labels), 4, -1)
            if not torch.allclose(cells.mean(1), mean, atol=1e-6, rtol=1e-5):
                raise ValueError('The spatial readout does not reproduce the global context mean')
    return features, labels
