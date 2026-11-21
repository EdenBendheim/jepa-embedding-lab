"""Read global and ordered quadrant context means under matched image views."""

import torch

from jepa_lab.analysis.color_statistics import rgb_statistics
from jepa_lab.analysis.spatial_embeddings import paired_embeddings

from experiments.multiblock_patch_control import shuffle_patches


def encoder_views(model, images, indices, *, patch_seed):
    """Extract both readouts from a single context pass per view."""
    shuffled = shuffle_patches(images, indices, seed=patch_seed)
    return {
        view: {name: features[name] for name in ('mean', 'quadrants')}
        for view, features in (
            ('original', paired_embeddings(model, images, indices)),
            ('patch_shuffle', paired_embeddings(model, shuffled, indices)),
        )
    }


def rgb_views(images, indices, *, patch_seed):
    """Compute six RGB moments globally and six in each ordered quadrant."""
    original_rgb = rgb_statistics(images)
    shuffled = shuffle_patches(images, indices, seed=patch_seed)
    shuffled_rgb = rgb_statistics(shuffled)
    if not torch.allclose(original_rgb, shuffled_rgb, atol=1e-7, rtol=0):
        raise ValueError('Patch shuffling changed a global RGB moment')
    return {
        'original': {'mean': original_rgb, 'quadrants': _quadrant_rgb(images)},
        'patch_shuffle': {'mean': shuffled_rgb, 'quadrants': _quadrant_rgb(shuffled)},
    }


def _quadrant_rgb(images):
    cells = images.reshape(len(images), 3, 2, 16, 2, 16).permute(0, 2, 4, 1, 3, 5)
    pixels = cells.reshape(len(images), 4, 3, 256).double().div(255)
    moments = torch.cat((pixels.mean(-1), pixels.std(-1, unbiased=False)), dim=-1)
    return moments.flatten(1).float()
