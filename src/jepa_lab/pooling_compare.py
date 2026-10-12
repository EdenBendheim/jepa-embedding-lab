"""Compare fixed pooling rules on the same checkpoint and examples."""

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
from statistics import mean, stdev

import torch

from .compare import fit_representation, validate_comparison_inputs, validate_probe_seeds, write_report
from .cifar10 import CIFAR10Binary, make_manifest
from .diagnostics import paired_diagnostics
from .features import _sha, load_features
from .pooling import POOLING
from .ridge import validate_ridge_candidates
from .probes import ProbeConfig
from .selection import validate_pilot


def validate_pooling_names(arms):
    if (not isinstance(arms, dict) or not 2 <= len(arms) <= 3
            or next(iter(arms), None) != "mean" or any(name not in POOLING for name in arms)):
        raise ValueError("Use 2-3 distinct pooling arms with mean first")


def validate_pooling_arms(arms, candidates):
    validate_pooling_names(arms)
    anchor = arms["mean"]
    reference = validate_comparison_inputs(anchor, candidates)
    fixed = ("checkpoint_sha256", "completed_steps", "training_settings", "config", "encoder", "normalization")
    contract = {key: reference[0].metadata.get(key) for key in fixed}
    if (not _sha(contract["checkpoint_sha256"]) or type(contract["completed_steps"]) is not int
            or contract["completed_steps"] < 1 or not isinstance(contract["training_settings"], dict)):
        raise ValueError("Pooling requires recorded checkpoint/training identity")
    random_seed = anchor["random"][0].metadata.get("seed")
    if type(random_seed) is not int or not 0 <= random_seed < 2**63:
        raise ValueError("Pooling arms require the same recorded random-encoder seed")
    for mode, sets in arms.items():
        pair = validate_comparison_inputs(sets, candidates)
        if (pair[0].metadata.get("pooling") != POOLING[mode]
                or {key: pair[0].metadata.get(key) for key in fixed} != contract
                or sets["random"][0].metadata.get("seed") != random_seed):
            raise ValueError("Pooling arms must vary only pooling, with the same checkpoint and random seed")
        for item, expected in zip(pair, reference):
            if (item.indices != expected.indices or not torch.equal(item.labels, expected.labels)
                    or item.manifest_sha256 != expected.manifest_sha256
                    or item.selection_sha256 != expected.selection_sha256):
                raise ValueError("Pooling arms require identical ordered examples and labels")
        for item, expected in zip(sets["pixels"], anchor["pixels"]):
            if (item.artifact_sha256 != expected.artifact_sha256 or item.metadata != expected.metadata
                    or not torch.equal(item.features, expected.features)):
                raise ValueError("Pooling arms must reuse the exact same pixel artifacts")
    return reference


def compare_pooling(arms, candidates, seeds, *, probe_dir=None, ridge_alphas=None):
    """Fit matched random pooling and one shared pixel classifier per seed."""
    reference = validate_pooling_arms(arms, candidates)
    validate_probe_seeds(seeds)
    if ridge_alphas is not None:
        validate_ridge_candidates(ridge_alphas)
    labels, runs = reference[1].labels, []
    for index, seed in enumerate(seeds):
        configs = tuple(replace(config, seed=seed) for config in candidates)
        directory = Path(probe_dir)/f"seed-{seed}" if probe_dir is not None else None
        alphas = ridge_alphas if index == 0 else None
        pixels, pixel_predictions = fit_representation(arms["mean"]["pixels"], configs, "pixels", directory, alphas)
        reports, predictions = {}, {}
        for mode, sets in arms.items():
            reports[mode], predictions[mode] = {}, {}
            for name in ("checkpoint", "random"):
                reports[mode][name], predictions[mode][name] = fit_representation(
                    sets[name], configs, f"{mode}-{name}", directory, alphas)
        paired = {}
        for head in pixel_predictions:
            paired[head] = {}
            for mode in arms:
                baselines = dict(random=predictions[mode]["random"][head], pixels=pixel_predictions[head])
                if mode != "mean": baselines["mean"] = predictions["mean"]["checkpoint"][head]
                paired[head][mode] = {f"vs_{name}": paired_diagnostics(labels, values,
                    predictions[mode]["checkpoint"][head]) for name, values in baselines.items()}
        runs.append(dict(seed=seed, candidate_grid=[asdict(config) for config in configs],
                         arms=reports, pixels=pixels, paired_validation=paired))

    def summary(values):
        return dict(validation_accuracies=values, mean_accuracy=mean(values),
                    sample_standard_deviation=stdev(values) if len(values) > 1 else 0.0)

    outcomes = {f"{mode}-{name}": summary([run["arms"][mode][name]["validation"]["accuracy"] for run in runs])
                for mode in arms for name in ("checkpoint", "random")}
    outcomes["pixels"] = summary([run["pixels"]["validation"]["accuracy"] for run in runs])
    return dict(version=1, task="Fixed-checkpoint pooling ablation", poolings=list(arms),
        checkpoint_sha256=reference[0].metadata["checkpoint_sha256"],
        completed_steps=reference[0].metadata["completed_steps"],
        manifest_sha256=reference[0].manifest_sha256, selection_sha256=reference[0].selection_sha256,
        counts=dict(train=len(reference[0].labels), validation=len(labels)), seeds=list(seeds),
        feature_dimensions={mode: sets["checkpoint"][0].features.shape[1] for mode, sets in arms.items()},
        ridge_alpha_grid=ridge_alphas, runs=runs, summary=outcomes,
        runtime=dict(torch=str(torch.__version__), cpu_threads=torch.get_num_threads()),
        test_evaluated=False,
        baseline_policy="Same random encoder with matching pooling per arm; pixel artifacts/classifiers shared; ridge fitted once",
        limitation="Validation-selected pooling on one fixed checkpoint. mean_std doubles classifier dimension; probe seeds are not encoder-training or test uncertainty")


def run_pooling(root, manifest_path, selection_path, pixel_dir, arm_dirs, *, candidates, seeds,
                probe_dir=None, ridge_alphas=None):
    """Load/verify saved pooling artifacts; do not extract features or read images."""
    validate_pooling_names(arm_dirs)
    train, test = CIFAR10Binary(root), CIFAR10Binary(root, train=False)
    manifest = json.loads(Path(manifest_path).read_text())
    if manifest != make_manifest(train, test, seed=manifest["seed"]):
        raise ValueError("Current data differs from the fixed manifest")
    selection = validate_pilot(json.loads(Path(selection_path).read_text()), manifest, train.labels)

    def pair(directory, name):
        return tuple(load_features(Path(directory)/f"{name}-{partition}.pt", manifest=manifest,
            selection=selection, training_labels=train.labels, partition=partition) for partition in ("train", "validation"))

    pixels = pair(pixel_dir, "pixels")
    arms = {mode: dict(pixels=pixels, checkpoint=pair(directory, "checkpoint"), random=pair(directory, "random"))
            for mode, directory in arm_dirs.items()}
    return compare_pooling(arms, candidates, seeds, probe_dir=probe_dir, ridge_alphas=ridge_alphas)


def main():
    parser = argparse.ArgumentParser(description="Compare verified saved pooling arms on one checkpoint; no test evaluation")
    parser.add_argument("root", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--pixel-dir", type=Path, required=True)
    parser.add_argument("--arm", nargs=2, action="append", required=True, metavar=("POOLING", "FEATURE_DIRECTORY"))
    parser.add_argument("--probe-dir", type=Path)
    parser.add_argument("--probe-seeds", type=int, nargs="+", default=[29, 31, 37])
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--learning-rates", type=float, nargs="+", default=[0.0001, 0.001, 0.01, 0.05])
    parser.add_argument("--weight-decays", type=float, nargs="+", default=[0.01, 0.1])
    parser.add_argument("--ridge-alphas", type=float, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    try:
        directories = dict(args.arm)
        if len(directories) != len(args.arm):
            raise ValueError("Pooling arm names must be distinct")
        if not args.probe_seeds:
            raise ValueError("At least one probe seed is required")
        candidates = tuple(ProbeConfig(args.steps, lr, wd, args.probe_seeds[0])
                           for lr in args.learning_rates for wd in args.weight_decays)
        report = run_pooling(args.root, args.manifest, args.selection, args.pixel_dir, directories,
                             candidates=candidates, seeds=args.probe_seeds, probe_dir=args.probe_dir,
                             ridge_alphas=args.ridge_alphas)
        write_report(args.output, report)
    except (ValueError, OSError, KeyError, TypeError, RuntimeError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
