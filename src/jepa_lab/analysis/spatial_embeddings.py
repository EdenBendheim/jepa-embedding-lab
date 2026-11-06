"""Extract matched global and spatial summaries from one frozen forward pass."""

import torch

from ..cifar10 import patchify_rgb
from ..embeddings import rgb_config
from .spatial_control import _validate_indices, shuffle_quadrants
from .spatial_pool import spatial_mean

MODES = ('mean', 'quadrants', 'shuffled_quadrants')


@torch.inference_mode()
def paired_embeddings(model, images, indices):
    rgb_config(model.config)
    if (not isinstance(images, torch.Tensor) or images.device.type != 'cpu'
            or images.dtype != torch.uint8 or images.ndim != 4
            or not 1 <= len(images) <= 32 or images.shape[1:] != (3, 32, 32)):
        raise ValueError('Spatial extraction requires 1-32 CPU uint8 RGB images of size 32x32')
    _validate_indices(indices, len(images))
    patches = patchify_rgb(images)
    was_training = model.training
    try:
        model.eval()
        positions = torch.arange(64).expand(len(images), -1)
        tokens = model.context_encoder(patches, positions)
        quadrants = spatial_mean(tokens)
        return dict(mean=spatial_mean(tokens, bins=1), quadrants=quadrants,
                    shuffled_quadrants=shuffle_quadrants(quadrants, indices))
    finally:
        model.train(was_training)
