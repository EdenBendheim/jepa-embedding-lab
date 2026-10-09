"""Training-only normalized multiclass ridge probes for a stable linear baseline."""

from dataclasses import asdict, dataclass
import math

import torch
from torch.nn import functional as F

from .probes import LinearProbe, validate_probe_data


@dataclass(frozen=True)
class RidgeConfig:
    alpha: float = 1.0

    def __post_init__(self):
        if (isinstance(self.alpha, bool) or not isinstance(self.alpha, (float, int))
                or not math.isfinite(self.alpha) or not 1e-6 <= self.alpha <= 1e4):
            raise ValueError("Ridge alpha must be finite and in [1e-6, 10000]")


@dataclass(frozen=True)
class RidgeProbe(LinearProbe):
    config: RidgeConfig

    @torch.inference_mode()
    def score(self, features, labels):
        result = super().score(features, labels)
        loss = F.mse_loss(self.logits(features), F.one_hot(labels, 10).float())
        if not torch.isfinite(loss):
            raise ValueError("Ridge scoring produced non-finite MSE")
        result["mean_squared_error"] = float(loss)
        return result


def fit_ridge(features, labels, config: RidgeConfig = RidgeConfig()) -> RidgeProbe:
    validate_probe_data(features, labels, fitting=True)
    if not isinstance(config, RidgeConfig):
        raise ValueError("Use a validated RidgeConfig")
    values = features.detach().clone().double()
    mean = values.mean(0).float()
    std = values.std(0, unbiased=False).float()
    constant = std < 1e-6
    scale = torch.where(constant, torch.ones_like(std), std)
    # Center a second time in float64 to make the intercept unpenalized despite float32 rounding.
    normalized = ((features.detach().clone() - mean) / scale).double()
    if not torch.isfinite(mean).all() or not torch.isfinite(scale).all() or not torch.isfinite(normalized).all():
        raise ValueError("Ridge normalization produced non-finite values")
    center = normalized.mean(0)
    x = normalized - center
    y = F.one_hot(labels, 10).double()
    prior = y.mean(0)
    target = y - prior
    penalty = config.alpha * len(labels)
    if x.shape[1] <= x.shape[0]:
        gram = x.T @ x + penalty * torch.eye(x.shape[1], dtype=torch.float64)
        coefficients = torch.linalg.solve(gram, x.T @ target)
    else:
        gram = x @ x.T + penalty * torch.eye(x.shape[0], dtype=torch.float64)
        coefficients = x.T @ torch.linalg.solve(gram, target)
    weight = coefficients.T.float()
    bias = (prior - center @ coefficients).float()
    if not torch.isfinite(weight).all() or not torch.isfinite(bias).all():
        raise ValueError("Ridge solve produced non-finite coefficients")
    return RidgeProbe(mean, scale, constant, weight, bias, config)


def validate_ridge_candidates(alphas):
    if not isinstance(alphas, (list, tuple)) or not 1 <= len(alphas) <= 8:
        raise ValueError("Use 1-8 distinct ridge alpha candidates")
    configs = tuple(RidgeConfig(alpha) for alpha in alphas)
    if len(set(configs)) != len(configs):
        raise ValueError("Ridge alpha candidates must be distinct")
    return configs


def select_ridge(train, train_labels, validation, validation_labels, alphas):
    validate_probe_data(train, train_labels, fitting=True)
    validate_probe_data(validation, validation_labels, dimension=train.shape[1])
    configs = validate_ridge_candidates(alphas)
    probes, results = [], []
    for i, config in enumerate(configs):
        probe = fit_ridge(train, train_labels, config)
        probes.append(probe)
        results.append(dict(candidate=i, config=asdict(config), training=probe.score(train, train_labels),
                            validation=probe.score(validation, validation_labels)))
    chosen = min(range(len(results)), key=lambda i: (-results[i]["validation"]["accuracy"],
                                                    results[i]["validation"]["mean_squared_error"], i))
    return probes[chosen], dict(selected_candidate=chosen, candidates=results,
        objective="Mean squared error against one-hot labels plus alpha * squared coefficient norm; intercept unpenalized",
        solver="Float64 primal solve when dimensions <= rows, otherwise dual solve; returned weights are float32",
        selection_rule="Highest validation accuracy, then lowest validation MSE, then declared order",
        normalization="Training-only population mean/std; std below 1e-6 uses scale 1",
        limitation="Validation-selected ridge scores are not calibrated probabilities or official test results")
