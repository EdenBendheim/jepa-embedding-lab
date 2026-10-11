"""Recheck and publish October 10 aggregate results from ignored local artifacts.

Run from the repository root after the protocol's comparisons. This never fits
a classifier or samples images. It fails if recorded scores/sources differ.
"""

import json
from pathlib import Path
from statistics import mean, stdev

import torch

from jepa_lab.cifar10 import CIFAR10Binary, make_manifest
from jepa_lab.compare import write_report
from jepa_lab.diagnostics import paired_diagnostics
from jepa_lab.features import load_features
from jepa_lab.probe_artifacts import load_probe


def read(path):
    return json.loads(Path(path).read_text())


def compact(report):
    output = {key: report[key] for key in
              ("train_artifact", "validation_artifact", "selected_config", "training", "validation")}
    artifact = report["selected_probe_artifact"]
    output["saved_classifier"] = {key: artifact[key] for key in ("state_sha256", "artifact_sha256")}
    output["selected_candidate"] = report["probe_selection"]["selected_candidate"]
    output["candidate_scores"] = [dict(config=c["config"], training_accuracy=c["training"]["accuracy"],
        validation_accuracy=c["validation"]["accuracy"], validation_cross_entropy=c["validation"]["cross_entropy"])
        for c in report["probe_selection"]["candidates"]]
    if "ridge" in report:
        ridge = report["ridge"]
        output["ridge"] = {key: ridge[key] for key in ("selected_config", "training", "validation")}
        output["ridge"]["saved_classifier"] = {key: ridge["selected_probe_artifact"][key]
                                              for key in ("state_sha256", "artifact_sha256")}
    return output


def main():
    torch.set_num_threads(1)
    base = read("runs/progression-20261010.json")
    manifest, selection = read("runs/cifar10-manifest.json"), read("runs/pilot-selection-20261009.json")
    dataset = CIFAR10Binary("data/cifar-10-batches-bin")
    assert manifest == make_manifest(dataset, CIFAR10Binary("data/cifar-10-batches-bin", train=False), seed=manifest["seed"])
    training_indices = set(manifest["partitions"]["train"]["indices"])
    records, checks, pixel_reference = [], [], None
    for seed, random_seed in ((20261003, 29), (20261010, 31), (20261011, 37)):
        if seed == 20261003:
            runs = [dict(seed=r["seed"], candidate_grid=r["candidate_grid"],
                representations=dict(checkpoint=r["checkpoints"]["later"], **r["baselines"]),
                paired_validation={head: {f"checkpoint_{key}": value for key, value in arms["later"].items() if key != "vs_early"}
                                   for head, arms in r["paired_validation"].items()}) for r in base["runs"]]
            directory = Path("runs/pilot-features-20261009")
            training = read("runs/cifar10-training-20261009.json")
        else:
            report = read(f"runs/pilot-comparison-seed-{seed}.json")
            runs = [dict(r, seed=s) for s, r in zip(report["seeds"], report["runs"], strict=True)]
            directory = Path(f"runs/pilot-features-seed-{seed}")
            training = read(f"runs/cifar10-training-seed-{seed}.json")
            assert training["start_step"] == 0
        assert training["completed_steps"] == 223 and training["settings"]["seed"] == seed
        assert all(index in training_indices for row in training["history"] for index in row["train_indices"])
        sets = {name: tuple(load_features(directory/f"{name}-{p}.pt", manifest=manifest, selection=selection,
            training_labels=dataset.labels, partition=p) for p in ("train", "validation"))
            for name in ("checkpoint", "random", "pixels")}
        if pixel_reference is None:
            pixel_reference = sets["pixels"]
        assert all(torch.equal(a.features, b.features) for a, b in zip(sets["pixels"], pixel_reference))
        assert sets["random"][0].metadata["seed"] == random_seed
        assert sets["checkpoint"][0].metadata["training_settings"] == training["settings"]
        assert sets["checkpoint"][0].metadata["config"] == base["runs"][0]["checkpoints"]["later"]["train_artifact"]["feature_metadata"]["config"]
        assert training["settings"] == dict(seed=seed, batch_size=4, learning_rate=0.001, momentum=0.99)
        assert [run["seed"] for run in runs] == [29, 31, 37]
        for run in runs:
            assert run["candidate_grid"] == base["runs"][[29, 31, 37].index(run["seed"])]["candidate_grid"]
            predictions = {}
            for name, rep in run["representations"].items():
                predictions[name] = {}
                for head, selected in [("adamw", rep)] + ([("ridge", rep["ridge"])] if "ridge" in rep else []):
                    probe, summary = load_probe(selected["selected_probe_artifact"]["path"], *sets[name])
                    assert summary["scores"] == dict(training=selected["training"], validation=selected["validation"])
                    predictions[name][head] = probe.logits(sets[name][1].features).argmax(1)
                    checks.append(dict(training_seed=seed, classifier_seed=run["seed"], representation=name,
                                       head=head, state_sha256=summary["state_sha256"], scores_reproduced=True))
            for head, comparisons in run["paired_validation"].items():
                for key, expected in comparisons.items():
                    baseline = key.removeprefix("checkpoint_vs_")
                    assert expected == paired_diagnostics(sets["checkpoint"][1].labels,
                        predictions[baseline][head], predictions["checkpoint"][head])
        values = {name: [r["representations"][name]["validation"]["accuracy"] for r in runs] for name in sets}
        records.append(dict(training_seed=seed, random_encoder_seed=random_seed, completed_steps=223,
            summary={n: dict(validation_accuracies=v, mean_accuracy=mean(v), classifier_seed_sample_sd=stdev(v)) for n, v in values.items()},
            ridge_accuracies={n: runs[0]["representations"][n]["ridge"]["validation"]["accuracy"] for n in sets},
            training_diagnostics=dict(settings=training["settings"], first_recorded_step=training["history"][0]["step"],
                first_loss=training["history"][0]["loss"], last_loss=training["history"][-1]["loss"],
                first_target=training["history"][0]["target"], last_target=training["history"][-1]["target"], all_samples_in_training_partition=True),
            runs=[dict(classifier_seed=r["seed"], representations={n: compact(rep) for n, rep in r["representations"].items()},
                       paired_validation=r["paired_validation"]) for r in runs]))
    aggregate = {name: dict(encoder_means=[r["summary"][name]["mean_accuracy"] for r in records],
        mean_accuracy=mean(r["summary"][name]["mean_accuracy"] for r in records),
        sample_standard_deviation=stdev(r["summary"][name]["mean_accuracy"] for r in records),
        ridge_accuracies=[r["ridge_accuracies"][name] for r in records], ridge_mean_accuracy=mean(r["ridge_accuracies"][name] for r in records))
        for name in ("checkpoint", "random", "pixels")}
    public = dict(version=1, task="Three encoder-training seeds on the fixed reused-validation pilot",
        protocol_commit="0c0c71e4b09f0866b1f227865830a484ec08f19b", manifest_sha256=manifest["manifest_sha256"],
        selection_sha256=selection["selection_sha256"], counts=dict(train=500, validation=200), classifier_seeds=[29, 31, 37],
        candidate_grid=base["runs"][0]["candidate_grid"], ridge_alphas=[0.01, 0.1, 1, 10], records=records, aggregate=aggregate,
        checks=dict(saved_classifiers_reproduced=len(checks), paired_errors_reproduced=True, pixel_values_identical=True,
                    training_partition_only=True, refitted=False), test_evaluated=False,
        limitation="Descriptive initialization/sampling/mask variation on reused validation examples, with validation-selected heads. Only three encoder seeds; not independent data, a confidence interval or official test accuracy.")
    write_report(Path("experiments/training-seeds-2026-10-10-results.json"), public)
    write_report(Path("runs/training-seeds-recheck-20261010.json"), dict(probes=checks, test_evaluated=False, refitted=False))
    print(json.dumps(aggregate, indent=2))
    print(f"Reproduced {len(checks)} saved classifiers and all paired errors without refitting")


if __name__ == "__main__":
    main()
