"""Continuous ridge reconstruction using normalization from fitting rows."""

from dataclasses import dataclass

import torch

from ..probes import _values
from ..ridge import RidgeConfig


@dataclass(frozen=True)
class Reconstruction:
    source_mean: torch.Tensor
    source_scale: torch.Tensor
    target_mean: torch.Tensor
    coefficients: torch.Tensor

    @torch.inference_mode()
    def predict(self, source):
        _values(source, dimension=len(self.source_mean))
        normalized = (source.double() - self.source_mean) / self.source_scale
        prediction = (normalized @ self.coefficients + self.target_mean).float()
        if not torch.isfinite(prediction).all():
            raise ValueError('Reconstruction prediction is non-finite')
        return prediction


@torch.inference_mode()
def fit_reconstruction(source, target, config=RidgeConfig(.01)):
    _values(source)
    _values(target)
    if len(source) != len(target) or len(source) < 2 or source.shape[1] > 64:
        raise ValueError('Use aligned fitting rows and at most 64 source coordinates')
    if not isinstance(config, RidgeConfig):
        raise ValueError('Use a validated RidgeConfig')
    source = source.double()
    target = target.double()
    mean = source.mean(dim=0)
    std = source.std(dim=0, unbiased=False)
    scale = torch.where(std < 1e-6, torch.ones_like(std), std)
    normalized = (source - mean) / scale
    target_mean = target.mean(dim=0)
    gram = normalized.T @ normalized
    gram += config.alpha * len(source) * torch.eye(source.shape[1], dtype=torch.float64)
    coefficients = torch.linalg.solve(gram, normalized.T @ (target - target_mean))
    if not torch.isfinite(coefficients).all():
        raise ValueError('Reconstruction coefficients are non-finite')
    return Reconstruction(mean, scale, target_mean, coefficients)
