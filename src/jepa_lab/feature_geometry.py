"""Descriptive frozen-feature geometry; not an accuracy or collapse guarantee."""

import argparse
import json
import math
from pathlib import Path

import torch

from .cifar10 import CIFAR10Binary, make_manifest
from .compare import write_report
from .features import load_features
from .selection import validate_pilot


@torch.inference_mode()
def feature_geometry(features: torch.Tensor) -> dict:
    if (not isinstance(features, torch.Tensor) or features.dtype != torch.float32 or features.ndim != 2
            or not 1 <= features.shape[0] <= 1000 or not 1 <= features.shape[1] <= 4096
            or not torch.isfinite(features).all()):
        raise ValueError("Geometry requires finite float32 [1-1000 rows, 1-4096 columns]")
    values = features.detach().to(device="cpu", dtype=torch.float64)
    centered = values - values.mean(0)
    variances = centered.square().mean(0)
    # Smaller Gram matrix bounds memory even for the 3,072-column pixel baseline.
    gram = centered @ centered.T if values.shape[0] <= values.shape[1] else centered.T @ centered
    spectrum = torch.linalg.eigvalsh(gram).clamp_min(0)
    tolerance = float(spectrum.max()) * max(values.shape) * torch.finfo(torch.float64).eps
    positive = spectrum[spectrum > tolerance]
    effective_rank = 0.0
    if len(positive):
        probabilities = positive / positive.sum()
        effective_rank = math.exp(float(-(probabilities * probabilities.log()).sum()))
    norms = values.norm(dim=1)
    nonzero = norms > 0
    count = int(nonzero.sum())
    cosine = None
    if count >= 2:
        normalized = values[nonzero] / norms[nonzero, None]
        cosine = float((normalized.sum(0).square().sum() - count) / (count * (count - 1)))
        cosine = max(-1.0, min(1.0, cosine))
    return dict(count=len(values), feature_dim=values.shape[1],
        zero_norm_rows=int((~nonzero).sum()), zero_variance_columns=int((variances == 0).sum()),
        mean_feature_population_std=float(variances.sqrt().mean()),
        mean_row_l2=float(norms.mean()), min_row_l2=float(norms.min()), max_row_l2=float(norms.max()),
        centered_numerical_rank=len(positive), covariance_entropy_effective_rank=effective_rank,
        eigenvalue_tolerance=tolerance, mean_nonzero_row_pair_cosine=cosine,
        limitation="Descriptive geometry of the supplied rows; no fitted transform, quality claim, significance test, or guarantee against collapse")


def inspect_geometry(root, manifest_path, selection_path, artifact, *, partition):
    train, test = CIFAR10Binary(root), CIFAR10Binary(root, train=False)
    manifest = json.loads(Path(manifest_path).read_text())
    if manifest != make_manifest(train, test, seed=manifest["seed"]):
        raise ValueError("Current dataset differs from the fixed manifest")
    selection = validate_pilot(json.loads(Path(selection_path).read_text()), manifest, train.labels)
    values = load_features(artifact, manifest=manifest, selection=selection,
                           training_labels=train.labels, partition=partition)
    return dict(version=1, source=values.summary(), geometry=feature_geometry(values.features), test_evaluated=False)


def main():
    parser = argparse.ArgumentParser(description="Inspect verified pilot feature geometry without fitting or reading images")
    parser.add_argument("root", type=Path)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--partition", choices=("train", "validation"), required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    try:
        report = inspect_geometry(args.root, args.manifest, args.selection, args.artifact, partition=args.partition)
        if args.output: write_report(args.output, report)
    except (ValueError, OSError, KeyError, TypeError, RuntimeError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
