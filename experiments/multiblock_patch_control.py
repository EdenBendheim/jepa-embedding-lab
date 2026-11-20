"""Rearrange whole RGB patches while preserving each image's color samples."""

import hashlib

import torch

from jepa_lab.analysis.spatial_control import _validate_indices


def shuffle_patches(images, indices, *, seed):
    """Apply a row-stable 4×4 patch permutation independent of batch and labels."""
    if (not isinstance(images, torch.Tensor) or images.device.type != 'cpu'
            or images.dtype != torch.uint8 or images.ndim != 4
            or images.shape[1:] != (3, 32, 32) or not 1 <= len(images) <= 1000
            or type(seed) is not int or not 0 <= seed < 2**63):
        raise ValueError('Use 1–1000 RGB uint8 images of size 32 by 32 and an integer seed')
    _validate_indices(indices, len(images))
    tiles = images.reshape(-1, 3, 8, 4, 8, 4).permute(0, 2, 4, 1, 3, 5).reshape(-1, 64, 3, 4, 4)
    orders = []
    for index in indices:
        key = hashlib.sha256(f'cifar10-patch-order-v1:{seed}:{index}'.encode()).digest()
        generator = torch.Generator().manual_seed(int.from_bytes(key[:8], 'big') % (2**63))
        orders.append(torch.randperm(64, generator=generator))
    shuffled = tiles[torch.arange(len(images))[:, None], torch.stack(orders)]
    return shuffled.reshape(-1, 8, 8, 3, 4, 4).permute(0, 3, 1, 4, 2, 5).reshape(-1, 3, 32, 32)
