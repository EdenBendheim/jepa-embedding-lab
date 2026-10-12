"""Execute the predeclared bounded October 11 pilot; outputs stay ignored."""

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import random

import torch

from jepa_lab.cifar10 import CIFAR10Binary, make_manifest
from jepa_lab.cifar10_train import train_cifar10
from jepa_lab.compare import compare_repeated_features, write_report
from jepa_lab.diagnostics import paired_diagnostics
from jepa_lab.embeddings import extract_cifar10, load_encoder
from jepa_lab.feature_geometry import feature_geometry
from jepa_lab.features import load_features
from jepa_lab.model import ModelConfig
from jepa_lab.pooling_compare import compare_pooling
from jepa_lab.probe_artifacts import load_probe
from jepa_lab.probes import ProbeConfig
from jepa_lab.selection import validate_pilot

ROOT = Path("data/cifar-10-batches-bin")
MANIFEST = Path("runs/cifar10-manifest.json")
SELECTION = Path("runs/pilot-selection-20261009.json")
BASE = Path("runs/pooling-mask-20261011")
CHECKPOINT = Path("checkpoints/cifar10-pilot-20261009.pt")
CHECKPOINT_SHA = "9943decda69020dc6e96174f5d5c6587390d207f000cbe6ca1255da731097b9e"
MANIFEST_SHA = "0333e4c2400a89a694e5b2f3b51ff39482e3cdf9397c5e6969fc9995ad1cec44"
SELECTION_SHA = "e0c4645e619917b1f001ea3382961906349995f8df8c9e0d3307569ef73e0e78"
SEEDS = [29, 31, 37]
ALPHAS = [0.01, 0.1, 1.0, 10.0]
CANDIDATES = tuple(ProbeConfig(200, lr, wd, 29) for lr in (0.0001, 0.001, 0.01, 0.05) for wd in (0.01, 0.1))


def inputs():
    train, test = CIFAR10Binary(ROOT), CIFAR10Binary(ROOT, train=False)
    manifest = json.loads(MANIFEST.read_text())
    selection = validate_pilot(json.loads(SELECTION.read_text()), manifest, train.labels)
    if (manifest != make_manifest(train, test, seed=manifest["seed"])
            or manifest["manifest_sha256"] != MANIFEST_SHA or selection["selection_sha256"] != SELECTION_SHA
            or hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest() != CHECKPOINT_SHA):
        raise ValueError("Declared dataset, selection or baseline checkpoint changed")
    _, metadata = load_encoder(CHECKPOINT, MANIFEST_SHA)
    if metadata["completed_steps"] != 223 or metadata["training_settings"] != dict(seed=20261003, batch_size=4, learning_rate=0.001, momentum=0.99):
        raise ValueError("Declared baseline settings changed")
    return manifest, selection, train.labels, ModelConfig(**metadata["config"])


def pair(directory, name, manifest, selection, labels):
    return tuple(load_features(directory/f"{name}-{partition}.pt", manifest=manifest, selection=selection,
        training_labels=labels, partition=partition) for partition in ("train", "validation"))


def pooling_sets(manifest, selection, labels):
    pixels = pair(Path("runs/pilot-features-20261009"), "pixels", manifest, selection, labels)
    return {mode: dict(pixels=pixels, checkpoint=pair(BASE/mode, "checkpoint", manifest, selection, labels),
        random=pair(BASE/mode, "random", manifest, selection, labels)) for mode in ("mean", "max", "mean_std")}


def recheck(scored, source):
    model, artifact = load_probe(scored["selected_probe_artifact"]["path"], *source)
    if (model.score(source[0].features, source[0].labels) != scored["training"]
            or model.score(source[1].features, source[1].labels) != scored["validation"]
            or artifact["artifact_sha256"] != scored["selected_probe_artifact"]["artifact_sha256"]):
        raise ValueError("Selected classifier no longer reproduces its saved scores/bytes")
    return model.logits(source[1].features).argmax(1)


def selected_rows(name, scored, source, seed, rows):
    predictions = {}
    for head, result in (("adamw", scored), ("ridge", scored.get("ridge"))):
        if result is not None:
            predictions[head] = recheck(result, source)
            rows.append(dict(name=name, head=head, classifier_seed=seed if head == "adamw" else None,
                config=result["selected_config"], training_accuracy=result["training"]["accuracy"],
                validation_accuracy=result["validation"]["accuracy"],
                validation_cross_entropy=result["validation"]["cross_entropy"],
                state_sha256=result["selected_probe_artifact"]["state_sha256"]))
    return predictions


def check_pooling(report, arms):
    rows = []
    for run in report["runs"]:
        pixel = selected_rows("pixels", run["pixels"], arms["mean"]["pixels"], run["seed"], rows)
        predictions = {mode: {name: selected_rows(f"{mode}-{name}", run["arms"][mode][name],
            arms[mode][name], run["seed"], rows) for name in ("checkpoint", "random")} for mode in arms}
        for head in pixel:
            for mode in arms:
                baselines = dict(random=predictions[mode]["random"][head], pixels=pixel[head])
                if mode != "mean": baselines["mean"] = predictions["mean"]["checkpoint"][head]
                expected = {f"vs_{name}": paired_diagnostics(arms[mode]["checkpoint"][1].labels,
                    values, predictions[mode]["checkpoint"][head]) for name, values in baselines.items()}
                if expected != run["paired_validation"][head][mode]:
                    raise ValueError("Pooling paired errors did not reproduce")
    return rows


def pooling():
    manifest, selection, labels, config = inputs()
    for mode in ("mean", "max", "mean_std"):
        for name in ("checkpoint", "random"):
            for partition in ("train", "validation"):
                extract_cifar10(ROOT, MANIFEST, representation=name, partition=partition,
                    checkpoint=CHECKPOINT if name == "checkpoint" else None,
                    config=config if name == "random" else None, seed=29, pooling=mode,
                    selection_path=SELECTION, output=BASE/mode/f"{name}-{partition}.pt")
    arms = pooling_sets(manifest, selection, labels)
    report = compare_pooling(arms, CANDIDATES, SEEDS, ridge_alphas=ALPHAS, probe_dir=BASE/"pooling-probes")
    write_report(BASE/"pooling-report.json", report)
    rows = check_pooling(report, arms)
    geometry = {f"{mode}-{name}": feature_geometry(arms[mode][name][0].features)
                for mode in arms for name in ("checkpoint", "random")}
    geometry["pixels"] = feature_geometry(arms["mean"]["pixels"][0].features)
    summary = dict(phase="pooling", checkpoint_sha256=CHECKPOINT_SHA, manifest_sha256=MANIFEST_SHA,
        selection_sha256=SELECTION_SHA, counts=report["counts"], feature_dimensions=report["feature_dimensions"],
        candidate_grid=report["runs"][0]["candidate_grid"], probe_seeds=SEEDS, ridge_alphas=ALPHAS,
        summary=report["summary"], selected_classifiers=rows, verified_classifiers=len(rows),
        paired_validation=[dict(seed=run["seed"], errors=run["paired_validation"]) for run in report["runs"]],
        training_feature_geometry=geometry, test_evaluated=False, limitation=report["limitation"])
    write_report(BASE/"pooling-summary.json", summary)
    return summary


def mask():
    manifest, selection, labels, config = inputs()
    arms = pooling_sets(manifest, selection, labels)
    anchor = json.loads((BASE/"pooling-report.json").read_text())
    check_pooling(anchor, arms)
    checkpoint = Path("checkpoints/cifar10-mask4x4-20261011.pt")
    log_path = BASE/"mask-training.json"
    if not checkpoint.exists() and not log_path.exists():
        training = train_cifar10(ROOT, MANIFEST, steps=223, batch_size=4, seed=20261003,
            config=config, target_rows=4, target_cols=4, context_fraction=0.75, checkpoint=checkpoint)
        write_report(log_path, training)
    else:
        if not checkpoint.exists() or not log_path.exists():
            raise ValueError("Cached custom checkpoint and training log must both exist")
        training = json.loads(log_path.read_text())
    _, metadata = load_encoder(checkpoint, MANIFEST_SHA)
    expected_settings = dict(seed=20261003, batch_size=4, learning_rate=0.001, momentum=0.99,
                             mask=dict(target_rows=4, target_cols=4, context_fraction=0.75))
    if (metadata["completed_steps"] != 223 or metadata["config"] != arms["mean"]["checkpoint"][0].metadata["config"]
            or metadata["training_settings"] != expected_settings or training["settings"] != expected_settings
            or training["manifest_sha256"] != MANIFEST_SHA or training["start_step"] != 0
            or training["completed_steps"] != 223 or len(training["history"]) != 223):
        raise ValueError("Custom run differs from its declared budget/settings")
    historical = []
    for path in ("runs/cifar10-pilot.json", "runs/cifar10-pilot-resumed.json", "runs/cifar10-training-20261009.json"):
        historical += json.loads(Path(path).read_text())["history"]
    actual = [row["train_indices"] for row in training["history"]]
    rng = random.Random(20261003)
    scheduled = [rng.sample(manifest["partitions"]["train"]["indices"], 4) for _ in range(223)]
    if actual != scheduled or actual != [row["train_indices"] for row in historical]:
        raise ValueError("Custom/default actual training-image order differs")
    if not all(torch.isfinite(torch.tensor(row["loss"])) and torch.isfinite(torch.tensor(row["gradient_norm"]))
               for row in training["history"]):
        raise ValueError("Non-finite custom training diagnostics")
    for partition in ("train", "validation"):
        extract_cifar10(ROOT, MANIFEST, checkpoint=checkpoint, partition=partition,
                       selection_path=SELECTION, output=BASE/"mask4x4"/f"checkpoint-{partition}.pt")
    custom = arms["mean"] | dict(checkpoint=pair(BASE/"mask4x4", "checkpoint", manifest, selection, labels))
    report = compare_repeated_features(custom, CANDIDATES, SEEDS, ridge_alphas=ALPHAS, probe_dir=BASE/"mask-probes")
    write_report(BASE/"mask-report.json", report)
    rows, paired = [], []
    for run, old in zip(report["runs"], anchor["runs"], strict=True):
        if run["candidate_grid"] != old["candidate_grid"]:
            raise ValueError("Mask classifier grid changed")
        predictions = {name: selected_rows(name, run["representations"][name], custom[name], seed, rows)
                       for name in custom for seed in [old["seed"]]}
        earlier = {head: recheck(scored, arms["mean"]["checkpoint"])
                   for head, scored in (("adamw", old["arms"]["mean"]["checkpoint"]),
                                       ("ridge", old["arms"]["mean"]["checkpoint"].get("ridge"))) if scored is not None}
        errors = {head: paired_diagnostics(custom["checkpoint"][1].labels, before, predictions["checkpoint"][head])
                  for head, before in earlier.items()}
        for name in ("random", "pixels"):
            original = old["arms"]["mean"]["random"] if name == "random" else old["pixels"]
            for head, result in (("adamw", original), ("ridge", original.get("ridge"))):
                if result is not None and not torch.equal(predictions[name][head], recheck(result, arms["mean"][name])):
                    raise ValueError("Shared mask baselines did not reproduce")
        for head in predictions["checkpoint"]:
            for name in ("random", "pixels"):
                expected = paired_diagnostics(custom["checkpoint"][1].labels, predictions[name][head], predictions["checkpoint"][head])
                if expected != run["paired_validation"][head][f"checkpoint_vs_{name}"]:
                    raise ValueError("Mask paired errors did not reproduce")
        paired.append(dict(seed=old["seed"], mask4x4_vs_default=errors))
    summary = dict(phase="mask", checkpoint_sha256=metadata["checkpoint_sha256"], baseline_checkpoint_sha256=CHECKPOINT_SHA,
        manifest_sha256=MANIFEST_SHA, selection_sha256=SELECTION_SHA, settings=expected_settings,
        counts=report["counts"], candidate_grid=report["runs"][0]["candidate_grid"], probe_seeds=SEEDS, ridge_alphas=ALPHAS,
        summary=report["summary"], selected_classifiers=rows, verified_classifiers=len(rows),
        paired_vs_default=paired, training_image_order_matched=True,
        training_image_order_sha256=hashlib.sha256(json.dumps(actual, separators=(",", ":")).encode()).hexdigest(),
        training=dict(steps=223, first_loss=training["history"][0]["loss"], last_loss=training["history"][-1]["loss"],
                      mask_geometry=training["mask_geometry"]),
        training_feature_geometry=feature_geometry(custom["checkpoint"][0].features),
        test_evaluated=False, limitation="One encoder seed and reused validation tuning; equal optimizer steps do not fix compute, predicted-token count or visible-token count")
    write_report(BASE/"mask-summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("pooling", "mask"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    summary = pooling() if args.phase == "pooling" else mask()
    print(json.dumps(dict(phase=summary["phase"], summary=summary["summary"],
                         verified_classifiers=summary["verified_classifiers"], test_evaluated=False), indent=2))


if __name__ == "__main__": main()
