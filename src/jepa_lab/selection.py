"""Balanced, reproducible pilot subsets inside the locked CIFAR-10 splits."""

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from .cifar10 import CIFAR10Binary, make_manifest


def document_hash(document: dict, field: str) -> str:
    payload = {key:value for key,value in document.items() if key != field}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def make_pilot(manifest: dict, labels: list[int], *, train_per_class: int = 10,
               validation_per_class: int = 5, seed: int = 20261007) -> dict:
    for count in (train_per_class, validation_per_class):
        if type(count) is not int or not 1 <= count <= 100:
            raise ValueError("Pilot counts must be 1-100 per class")
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("Pilot seed must be an integer in [0, 2**63)")
    if (manifest.get("manifest_sha256") != document_hash(manifest, "manifest_sha256")
            or len(labels) != 50000 or any(type(label) is not int or not 0 <= label < 10 for label in labels)):
        raise ValueError("Pilot requires a valid fixed manifest and official training labels")
    counts = dict(train=train_per_class, validation=validation_per_class)
    partitions, seen = {}, set()
    for name, count in counts.items():
        original = manifest["partitions"][name]
        indices = original["indices"]
        if (original["official_split"] != "train" or not isinstance(indices, list)
                or any(type(index) is not int or not 0 <= index < len(labels) for index in indices)
                or len(indices) != len(set(indices)) or seen.intersection(indices)):
            raise ValueError("Pilot partitions must be disjoint official training-file indices")
        seen.update(indices)
        chosen = []
        for label in range(10):
            group = [index for index in indices if labels[index] == label]
            if len(group) < count:
                raise ValueError("Fixed partition cannot supply the requested count for every class")
            group.sort(key=lambda index: (hashlib.sha256(f"cifar10-pilot-v1:{seed}:{name}:{index}".encode()).digest(), index))
            chosen.extend(group[:count])
        partitions[name] = dict(official_split="train", indices=sorted(chosen))
    result = dict(version=1, dataset="CIFAR-10", manifest_sha256=manifest["manifest_sha256"], seed=seed,
                  algorithm="cifar10-pilot-v1 SHA-256 within each fixed partition/class",
                  per_class=counts, partitions=partitions)
    return result | dict(selection_sha256=document_hash(result, "selection_sha256"))


def validate_pilot(selection: dict, manifest: dict, labels: list[int]) -> dict:
    if isinstance(selection, dict) and selection.get("kind") == "confirmation":
        return validate_confirmation(selection, manifest, labels)
    try:
        expected = make_pilot(manifest, labels, train_per_class=selection["per_class"]["train"],
                              validation_per_class=selection["per_class"]["validation"], seed=selection["seed"])
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError("Invalid pilot selection schema") from exc
    if selection != expected:
        raise ValueError("Pilot selection differs from the fixed balanced protocol or data identity")
    return selection


def make_confirmation(manifest: dict, labels: list[int], reference: dict, *,
                      exclusions=(), validation_per_class: int = 50,
                      seed: int = 20261012) -> dict:
    """Keep fitted training rows; reserve unseen validation rows, never test."""
    if (type(validation_per_class) is not int or not 1 <= validation_per_class <= 100
            or type(seed) is not int or not 0 <= seed < 2**63
            or not isinstance(exclusions, (tuple, list)) or len(exclusions) > 8):
        raise ValueError("Confirmation requires 1-100 rows/class, a valid seed, and at most eight exclusions")
    parents = [reference, *exclusions]
    for parent in parents:
        if not isinstance(parent, dict) or parent.get("version") != 1 or "kind" in parent:
            raise ValueError("Confirmation sources must be original version-one pilot selections")
        validate_pilot(parent, manifest, labels)
    hashes = [parent["selection_sha256"] for parent in parents]
    if len(hashes) != len(set(hashes)):
        raise ValueError("Confirmation source selections must be distinct")
    excluded = {index for parent in parents for index in parent["partitions"]["validation"]["indices"]}
    chosen = []
    for label in range(10):
        group = [index for index in manifest["partitions"]["validation"]["indices"]
                 if labels[index] == label and index not in excluded]
        if len(group) < validation_per_class:
            raise ValueError("Not enough unseen validation examples for every class")
        group.sort(key=lambda index: (hashlib.sha256(
            f"cifar10-confirmation-v1:{seed}:{index}".encode()).digest(), index))
        chosen.extend(group[:validation_per_class])
    result = dict(version=2, kind="confirmation", dataset="CIFAR-10",
                  manifest_sha256=manifest["manifest_sha256"], seed=seed,
                  algorithm="cifar10-confirmation-v1 SHA-256 within each fixed validation class, excluding prior pilots",
                  reference_selection=deepcopy(reference),
                  excluded_selections=deepcopy(sorted(exclusions, key=lambda item: item["selection_sha256"])),
                  per_class=dict(train=reference["per_class"]["train"], validation=validation_per_class),
                  partitions=dict(train=deepcopy(reference["partitions"]["train"]),
                                  validation=dict(official_split="train", indices=sorted(chosen))))
    return result | dict(selection_sha256=document_hash(result, "selection_sha256"))


def validate_confirmation(selection: dict, manifest: dict, labels: list[int]) -> dict:
    try:
        expected = make_confirmation(manifest, labels, selection["reference_selection"],
                                     exclusions=selection["excluded_selections"],
                                     validation_per_class=selection["per_class"]["validation"], seed=selection["seed"])
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError("Invalid confirmation selection schema") from exc
    if selection != expected:
        raise ValueError("Confirmation selection differs from its disjoint fixed protocol or data identity")
    return selection


def main():
    parser = argparse.ArgumentParser(description="Choose shared balanced train/validation examples; never select test")
    parser.add_argument("root", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-per-class", type=int, default=10)
    parser.add_argument("--validation-per-class", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20261007)
    args = parser.parse_args()
    try:
        manifest = json.loads(args.manifest.read_text())
        train, test = CIFAR10Binary(args.root), CIFAR10Binary(args.root, train=False)
        if manifest != make_manifest(train, test, seed=manifest["seed"]):
            raise ValueError("Manifest does not match current data or the fixed split")
        selection = make_pilot(manifest, train.labels, train_per_class=args.train_per_class,
                               validation_per_class=args.validation_per_class, seed=args.seed)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(selection, indent=2)+"\n")
    print(json.dumps(dict(selection_sha256=selection["selection_sha256"],
        class_counts={name:dict(Counter(train.labels[i] for i in part["indices"]))
                      for name,part in selection["partitions"].items()}), indent=2))


if __name__ == "__main__":
    main()
