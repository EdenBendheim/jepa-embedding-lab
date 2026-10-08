"""Bounded full-batch linear probes with training-only feature normalization."""

from dataclasses import asdict, dataclass
import math

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class ProbeConfig:
    steps: int = 200
    learning_rate: float = 0.05
    weight_decay: float = 0.01
    seed: int = 29

    def __post_init__(self):
        if type(self.steps) is not int or not 1 <= self.steps <= 1000:
            raise ValueError("Probe steps must be 1-1000")
        for value, name, lower in ((self.learning_rate, "learning rate", 0), (self.weight_decay, "weight decay", -1)):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not lower < value <= 1:
                raise ValueError(f"Probe {name} must be finite and within its bounded range")
        if self.weight_decay < 0:
            raise ValueError("Probe weight decay cannot be negative")
        if type(self.seed) is not int or not 0 <= self.seed < 2**63:
            raise ValueError("Probe seed must be an integer in [0, 2**63)")


def _values(features, dimension=None):
    if (not isinstance(features, torch.Tensor) or features.dtype != torch.float32 or features.device.type != "cpu"
            or features.ndim != 2 or not 1 <= features.shape[0] <= 1000 or not 1 <= features.shape[1] <= 4096
            or (dimension is not None and features.shape[1] != dimension) or not torch.isfinite(features).all()):
        raise ValueError("Probe features must be finite CPU float32 rows with matching dimensions (max 1000x4096)")


def _labels(labels, count, *, all_classes=False):
    if (not isinstance(labels, torch.Tensor) or labels.device.type != "cpu" or labels.dtype != torch.long
            or tuple(labels.shape) != (count,) or (labels < 0).any() or (labels >= 10).any()
            or (all_classes and set(labels.tolist()) != set(range(10)))):
        raise ValueError("Probe labels must match feature rows and classes 0-9; fitting requires all ten classes")


@dataclass(frozen=True)
class LinearProbe:
    mean: torch.Tensor
    scale: torch.Tensor
    constant_columns: torch.Tensor
    weight: torch.Tensor
    bias: torch.Tensor
    config: ProbeConfig

    @torch.inference_mode()
    def logits(self, features: torch.Tensor) -> torch.Tensor:
        _values(features, self.weight.shape[1])
        values = (features.detach()-self.mean)/self.scale
        logits = F.linear(values, self.weight, self.bias)
        if not torch.isfinite(logits).all():
            raise ValueError("Probe produced non-finite scores")
        return logits

    @torch.inference_mode()
    def score(self, features: torch.Tensor, labels: torch.Tensor) -> dict:
        _values(features, self.weight.shape[1])
        _labels(labels, features.shape[0])
        logits = self.logits(features)
        predictions = logits.argmax(1)
        counts = torch.bincount(labels*10+predictions, minlength=100).reshape(10,10)
        loss = F.cross_entropy(logits, labels)
        if not torch.isfinite(loss):
            raise ValueError("Probe produced non-finite classification loss")
        per_class = []
        for label in range(10):
            support = int(counts[label].sum())
            per_class.append(dict(label=label, count=support, correct=int(counts[label,label]),
                                  accuracy=int(counts[label,label])/support if support else None))
        correct = int((predictions == labels).sum())
        return dict(count=len(labels), correct=correct,
                    accuracy=correct/len(labels), cross_entropy=float(loss),
                    per_class=per_class, confusion_matrix=counts.tolist())


def fit_probe(features: torch.Tensor, labels: torch.Tensor, config: ProbeConfig = ProbeConfig()) -> LinearProbe:
    _values(features)
    _labels(labels, features.shape[0], all_classes=True)
    if not isinstance(config, ProbeConfig):
        raise ValueError("Use a validated ProbeConfig")
    # No validation/test inputs enter this function. Detached copies cannot train the source encoder.
    values, targets = features.detach().clone(), labels.clone()
    mean = values.double().mean(0).float()
    std = values.double().std(0, unbiased=False).float()
    constant = std < 1e-6
    scale = torch.where(constant, torch.ones_like(std), std)
    values = (values-mean)/scale
    if not torch.isfinite(values).all():
        raise ValueError("Training normalization produced non-finite values")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config.seed)
        layer = torch.nn.Linear(values.shape[1], 10)
        optimizer = torch.optim.AdamW([dict(params=[layer.weight], weight_decay=config.weight_decay),
                                      dict(params=[layer.bias], weight_decay=0.0)], lr=config.learning_rate)
        for _ in range(config.steps):
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(layer(values), targets)
            if not torch.isfinite(loss):
                raise ValueError("Probe training produced non-finite loss")
            loss.backward()
            optimizer.step()
        weight, bias = layer.weight.detach().clone(), layer.bias.detach().clone()
    if not torch.isfinite(weight).all() or not torch.isfinite(bias).all():
        raise ValueError("Probe training produced non-finite weights")
    return LinearProbe(mean, scale, constant, weight, bias, config)


def validate_probe_data(features: torch.Tensor, labels: torch.Tensor, *, dimension=None, fitting=False):
    _values(features, dimension)
    _labels(labels, features.shape[0], all_classes=fitting)


def validate_candidates(candidates):
    if (not isinstance(candidates, (tuple, list)) or not 1 <= len(candidates) <= 8
            or any(not isinstance(config, ProbeConfig) for config in candidates)
            or len(set(candidates)) != len(candidates)
            or len({(config.seed, config.steps) for config in candidates}) != 1):
        raise ValueError("Use 1-8 distinct candidates with the same seed and step budget")


def select_probe(train: torch.Tensor, train_labels: torch.Tensor, validation: torch.Tensor,
                 validation_labels: torch.Tensor, candidates: tuple[ProbeConfig, ...]) -> tuple[LinearProbe, dict]:
    validate_probe_data(train, train_labels, fitting=True)
    validate_probe_data(validation, validation_labels, dimension=train.shape[1])
    validate_candidates(candidates)
    probes, results = [], []
    for index, config in enumerate(candidates):
        probe = fit_probe(train, train_labels, config)
        probes.append(probe)
        results.append(dict(candidate=index, config=asdict(config), training=probe.score(train, train_labels),
                            validation=probe.score(validation, validation_labels),
                            constant_columns=int(probe.constant_columns.sum())))
    chosen = min(range(len(results)), key=lambda index: (-results[index]["validation"]["accuracy"],
                                                        results[index]["validation"]["cross_entropy"], index))
    return probes[chosen], dict(selected_candidate=chosen, candidates=results,
        selection_rule="Highest validation accuracy, then lowest validation cross-entropy, then declared order",
        normalization="Training-only population mean/std; std below 1e-6 uses scale 1",
        optimizer="Full-batch AdamW; weight decay on weights only; no bias decay",
        limitation="Validation-selected pilot scores; no official test result or generalization claim")
