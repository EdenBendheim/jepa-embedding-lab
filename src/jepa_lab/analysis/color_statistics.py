"""Six per-image RGB statistics for fitting-image representation diagnostics."""

import torch

FEATURE_NAMES = ('red_mean', 'green_mean', 'blue_mean',
                 'red_population_std', 'green_population_std', 'blue_population_std')


@torch.inference_mode()
def rgb_statistics(images: torch.Tensor) -> torch.Tensor:
    if (not isinstance(images, torch.Tensor) or images.device.type != 'cpu'
            or images.dtype != torch.uint8 or images.ndim != 4
            or not 1 <= len(images) <= 1000 or images.shape[1:] != (3, 32, 32)):
        raise ValueError('Use 1-1000 CPU uint8 RGB images of size 32 by 32')
    values = images.double().div(255).flatten(2)
    return torch.cat((values.mean(dim=2), values.std(dim=2, unbiased=False)), dim=1).float()
