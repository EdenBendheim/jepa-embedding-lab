"""Controlled checkpoint-age comparisons with a single shared baseline pair."""

import re
from dataclasses import asdict, replace
from pathlib import Path
from statistics import mean, stdev

import torch

from .compare import fit_representation, validate_comparison_inputs, validate_probe_seeds
from .diagnostics import paired_diagnostics
from .features import _sha
from .ridge import validate_ridge_candidates


def validate_progression(arms, candidates):
    """Validate every arm before fitting; matching settings do not prove ancestry."""
    if (not isinstance(arms, dict) or not 2 <= len(arms) <= 4
            or any(not isinstance(name, str) or name in ("random", "pixels")
                   or re.fullmatch(r"[a-z][a-z0-9-]{0,47}", name) is None for name in arms)):
        raise ValueError("Use 2-4 ordered checkpoint arms with distinct short lowercase names")
    first = next(iter(arms.values()))
    reference = validate_comparison_inputs(first, candidates)
    contract = {key: reference[0].metadata.get(key) for key in
        ("config", "training_settings", "encoder", "pooling", "normalization")}
    previous_step, hashes, identity = 0, set(), []
    for name, sets in arms.items():
        pair = validate_comparison_inputs(sets, candidates)
        metadata = pair[0].metadata
        step, fingerprint = metadata.get("completed_steps"), metadata.get("checkpoint_sha256")
        if (type(step) is not int or step <= previous_step or not _sha(fingerprint) or fingerprint in hashes
                or not isinstance(metadata.get("training_settings"), dict)
                or {key: metadata.get(key) for key in contract} != contract):
            raise ValueError("Checkpoint ages must increase with distinct weights and identical architecture/training settings")
        for item, expected in zip(pair, reference):
            if (item.indices != expected.indices or not torch.equal(item.labels, expected.labels)
                    or item.manifest_sha256 != expected.manifest_sha256
                    or item.selection_sha256 != expected.selection_sha256):
                raise ValueError("Checkpoint arms require identical ordered examples and labels")
        for baseline in ("random", "pixels"):
            for item, expected in zip(sets[baseline], first[baseline]):
                if (item.artifact_sha256 != expected.artifact_sha256 or item.metadata != expected.metadata
                        or not torch.equal(item.features, expected.features)):
                    raise ValueError("All checkpoint arms must reuse the exact same random/pixel artifacts")
        identity.append(dict(name=name, completed_steps=step, checkpoint_sha256=fingerprint))
        previous_step = step
        hashes.add(fingerprint)
    return identity


def compare_progression(arms, candidates, seeds, *, probe_dir=None, ridge_alphas=None):
    """Select matching classifier grids, fitting shared baselines once per seed."""
    identity = validate_progression(arms, candidates)
    validate_probe_seeds(seeds)
    if ridge_alphas is not None:
        validate_ridge_candidates(ridge_alphas)
    anchor = next(iter(arms))
    reference = arms[anchor]["checkpoint"]
    labels = reference[1].labels
    runs = []
    for index, seed in enumerate(seeds):
        configs = tuple(replace(config, seed=seed) for config in candidates)
        alphas = ridge_alphas if index == 0 else None
        directory = Path(probe_dir)/f"seed-{seed}" if probe_dir is not None else None
        baselines, checkpoints, predictions = {}, {}, {}
        for name in ("random", "pixels"):
            baselines[name], predictions[name] = fit_representation(arms[anchor][name], configs, name, directory, alphas)
        for name, sets in arms.items():
            checkpoints[name], predictions[name] = fit_representation(sets["checkpoint"], configs, name, directory, alphas)
        paired = {head: {name: {f"vs_{baseline}": paired_diagnostics(labels,
            predictions[baseline][head], predictions[name][head]) for baseline in ("random", "pixels", anchor) if baseline != name}
            for name in arms} for head in predictions[anchor]}
        runs.append(dict(seed=seed, candidate_grid=[asdict(config) for config in configs], baselines=baselines,
                         checkpoints=checkpoints, paired_validation=paired))
    summary = {}
    for name in (*arms, "random", "pixels"):
        group = "checkpoints" if name in arms else "baselines"
        values = [run[group][name]["validation"]["accuracy"] for run in runs]
        summary[name] = dict(validation_accuracies=values, mean_accuracy=mean(values),
                             sample_standard_deviation=stdev(values) if len(values) > 1 else 0.0)
    return dict(version=1, task="Controlled checkpoint-age validation comparison", checkpoints=identity,
        manifest_sha256=reference[0].manifest_sha256, selection_sha256=reference[0].selection_sha256,
        counts=dict(train=len(reference[0].labels), validation=len(labels)), seeds=list(seeds),
        ridge_alpha_grid=ridge_alphas, runs=runs, summary=summary,
        shared_baseline_policy="Exact same random/pixel artifacts; AdamW fitted once per classifier seed and ridge once total",
        test_evaluated=False,
        limitation="Validation-selected checkpoint ages from matching recorded settings; metadata alone does not prove ancestry. Classifier-seed variation is not independent encoder training or test uncertainty")
