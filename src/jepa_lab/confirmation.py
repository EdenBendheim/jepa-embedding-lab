"""Evaluate an already selected classifier on reserved, untuned examples."""

from pathlib import Path
import hashlib
import argparse
import json

import torch

from .cifar10 import CIFAR10Binary, make_manifest
from .compare import write_report
from .features import FeatureSet, _sha, load_features
from .probe_artifacts import load_probe, probe_context
from .probes import validate_probe_data
from .selection import document_hash, validate_pilot, validate_confirmation


def score_confirmation(probe_path: str | Path, train: FeatureSet, validation: FeatureSet,
                       confirmation: FeatureSet, selection: dict) -> dict:
    """Reproduce the selected state on its old sources before scoring new rows."""
    context = probe_context(train, validation)
    if not isinstance(confirmation, FeatureSet) or confirmation.selection_kind != "confirmation":
        raise ValueError("Score a verified confirmation feature set")
    if (not isinstance(selection, dict) or type(selection.get("version")) is not int
            or selection["version"] != 2 or selection.get("kind") != "confirmation"
            or selection.get("selection_sha256") != document_hash(selection, "selection_sha256")):
        raise ValueError("Confirmation selection must retain its complete protocol fingerprint")
    try:
        reference = selection["reference_selection"]
        excluded = {index for parent in [reference, *selection["excluded_selections"]]
                    for index in parent["partitions"]["validation"]["indices"]}
        if (reference["selection_sha256"] != context["selection_sha256"]
                or reference["partitions"]["train"]["indices"] != list(train.indices)
                or reference["partitions"]["validation"]["indices"] != list(validation.indices)
                or selection["partitions"]["train"]["indices"] != list(train.indices)
                or selection["partitions"]["validation"]["indices"] != list(confirmation.indices)
                or selection["partitions"]["validation"]["official_split"] != "train"
                or selection["manifest_sha256"] != context["manifest_sha256"]):
            raise ValueError("Confirmation reference must match the classifier's original exact sources")
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError("Invalid confirmation reference schema") from exc
    if (confirmation.partition != "validation" or confirmation.manifest_sha256 != train.manifest_sha256
            or confirmation.selection_sha256 != selection["selection_sha256"]
            or confirmation.metadata != train.metadata or not _sha(confirmation.artifact_sha256)
            or len(confirmation.indices) != len(confirmation.labels)
            or len(set(confirmation.indices)) != len(confirmation.indices)
            or set(confirmation.indices).intersection(set(train.indices) | excluded)):
        raise ValueError("Confirmation examples must be disjoint with identical frozen representation provenance")
    validate_probe_data(confirmation.features, confirmation.labels, dimension=context["feature_dim"])
    for source in (train, validation, confirmation):
        if (source.features_sha256 != hashlib.sha256(source.features.contiguous().numpy().tobytes()).hexdigest()
                or source.labels_sha256 != hashlib.sha256(source.labels.contiguous().numpy().tobytes()).hexdigest()):
            raise ValueError("Verified feature rows or labels changed after loading")
    probe, original = load_probe(probe_path, train, validation)
    scores = probe.score(confirmation.features, confirmation.labels)
    return dict(version=1, task="Frozen selected-classifier confirmation on reserved validation examples",
                manifest_sha256=context["manifest_sha256"], reference_selection_sha256=train.selection_sha256,
                confirmation_selection_sha256=confirmation.selection_sha256,
                original_probe=original, confirmation_features=confirmation.summary(), confirmation=scores,
                normalization_refitted=False, classifier_refitted=False, settings_selected=False,
                excluded_validation_count=len(excluded), test_evaluated=False,
                limitation="Fresh examples relative to the declared prior pilots, not an official test result or a guarantee of encoder generalization; reusing these scores for tuning removes their confirmation role")


def score_saved_confirmation(root, manifest_path, reference_path, confirmation_path,
                             train_path, validation_path, fresh_path, probe_path) -> dict:
    train, test = CIFAR10Binary(root), CIFAR10Binary(root, train=False)
    manifest = json.loads(Path(manifest_path).read_text())
    if manifest != make_manifest(train, test, seed=manifest["seed"]):
        raise ValueError("Dataset files differ from the locked manifest")
    reference = validate_pilot(json.loads(Path(reference_path).read_text()), manifest, train.labels)
    if reference.get("kind") == "confirmation":
        raise ValueError("The classifier reference must be its original tuning pilot")
    selected = validate_confirmation(json.loads(Path(confirmation_path).read_text()), manifest, train.labels)
    original = tuple(load_features(path, manifest=manifest, selection=reference,
                                   training_labels=train.labels, partition=partition)
                     for path, partition in ((train_path,"train"),(validation_path,"validation")))
    fresh = load_features(fresh_path, manifest=manifest, selection=selected,
                          training_labels=train.labels, partition="validation")
    return score_confirmation(probe_path, *original, fresh, selected)


def main():
    parser = argparse.ArgumentParser(description="Score a fixed selected classifier on untuned confirmation rows; never refit or test")
    parser.add_argument("root", type=Path)
    parser.add_argument("probe", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--reference-selection", type=Path, required=True)
    parser.add_argument("--confirmation-selection", type=Path, required=True)
    parser.add_argument("--train-features", type=Path, required=True)
    parser.add_argument("--tuning-features", type=Path, required=True)
    parser.add_argument("--confirmation-features", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    try:
        report = score_saved_confirmation(args.root, args.manifest, args.reference_selection,
            args.confirmation_selection, args.train_features, args.tuning_features, args.confirmation_features, args.probe)
        if args.output:
            write_report(args.output, report)
    except (ValueError, OSError, KeyError, TypeError, RuntimeError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
