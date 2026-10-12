"""Compare fixed pooling rules on the same checkpoint and examples."""

from dataclasses import asdict, replace
from pathlib import Path
from statistics import mean, stdev

import torch

from .compare import fit_representation, validate_comparison_inputs, validate_probe_seeds
from .diagnostics import paired_diagnostics
from .features import _sha
from .pooling import POOLING
from .ridge import validate_ridge_candidates


def validate_pooling_arms(arms, candidates):
    if (not isinstance(arms, dict) or not 2 <= len(arms) <= 3
            or next(iter(arms), None) != "mean" or any(name not in POOLING for name in arms)):
        raise ValueError("Use 2-3 distinct pooling arms with mean first")
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
