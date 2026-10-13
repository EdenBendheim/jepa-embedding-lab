"""Evaluate an already selected classifier on reserved, untuned examples."""

from pathlib import Path
import hashlib

from .features import FeatureSet, _sha
from .probe_artifacts import load_probe, probe_context
from .probes import validate_probe_data
from .selection import document_hash


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
