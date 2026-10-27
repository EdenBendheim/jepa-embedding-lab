"""Opt-in image-level context variance penalty; no representation-quality claim."""

from dataclasses import dataclass
import math

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class ObjectiveConfig:
    variance_weight: float = 0.0
    std_floor: float = 1.0

    def __post_init__(self):
        for name, value in vars(self).items():
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number")
        if self.variance_weight < 0 or self.std_floor <= 0:
            raise ValueError("Variance weight must be nonnegative; std floor must be positive")

    def settings(self):
        """No new identity keys for the historical, unregularized objective."""
        if self.variance_weight == 0:
            return {}
        return {"objective": {"context_variance_weight": float(self.variance_weight),
                              "context_std_floor": float(self.std_floor)}}


def context_variance(tokens, *, std_floor=1.0):
    """Penalize low between-image variance, after visible-token mean pooling.

    Population variance is over images, not image/patch pairs. The fixed 1e-4
    epsilon stabilizes the square root; a constant batch still has zero gradient.
    Mask differences may contribute to this statistic. It cannot prove that
    embeddings retain semantic information or avoid all collapse modes.
    """
    ObjectiveConfig(1.0, std_floor)
    if (not isinstance(tokens, torch.Tensor) or tokens.ndim != 3
            or tokens.shape[0] < 2 or min(tokens.shape[1:]) < 1
            or not tokens.is_floating_point() or not torch.isfinite(tokens).all()):
        raise ValueError("Context tokens must be finite floating [images>=2, patches, width]")
    pooled = tokens.float().mean(dim=1)
    std = torch.sqrt(pooled.var(dim=0, unbiased=False) + 1e-4)
    penalty = F.relu(std_floor - std).mean()
    if not torch.isfinite(penalty):
        raise ValueError("Context variance penalty is not finite")
    return penalty, std.mean()


def objective_terms(predictions, targets, context_tokens=None, *, config=None):
    """Return differentiable loss components, keeping target gradients stopped."""
    config = config or ObjectiveConfig()
    if not isinstance(config, ObjectiveConfig):
        raise ValueError("Expected ObjectiveConfig")
    if (not isinstance(predictions, torch.Tensor) or not isinstance(targets, torch.Tensor)
            or predictions.ndim != 3 or predictions.shape != targets.shape
            or min(predictions.shape) < 1 or not predictions.is_floating_point()
            or not targets.is_floating_point() or not torch.isfinite(predictions).all()
            or not torch.isfinite(targets).all()):
        raise ValueError("Predictions and targets must be aligned finite floating [batch, targets, width]")
    prediction = F.smooth_l1_loss(predictions, targets.detach())
    terms = {"loss": prediction, "prediction_loss": prediction}
    if config.variance_weight:
        if (not isinstance(context_tokens, torch.Tensor) or context_tokens.ndim != 3
                or context_tokens.shape[0] != predictions.shape[0]
                or context_tokens.shape[-1] != predictions.shape[-1]):
            raise ValueError("Context batch and width must match predictions")
        penalty, std = context_variance(context_tokens, std_floor=config.std_floor)
        terms.update(loss=prediction + config.variance_weight * penalty,
                     context_variance_penalty=penalty, context_std_mean=std)
    return terms


def objective_diagnostics(terms, config):
    """Add only enabled-objective terms to journals; keep old rows unchanged."""
    if not config.variance_weight:
        return {}
    return {name: float(value.detach()) for name, value in terms.items() if name != "loss"}
