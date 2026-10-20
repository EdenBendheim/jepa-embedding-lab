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
    if not isinstance(selection, dict) or type(selection.get("version")) is not int:
        raise ValueError("Selection version must be an integer")
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


def excluded_validation_indices(selection: dict, *, _depth: int = 0) -> set[int]:
    """Collect observed rows across a bounded chain of earlier confirmations."""
    if _depth > 3:
        raise ValueError("Confirmation ancestry is limited to three earlier rounds")
    rows = set(selection["partitions"]["validation"]["indices"])
    if selection.get('version') == 3 and selection.get('kind') == 'confirmation':
        rows.update(selection['exposure_ledger']['observed_validation_indices'])
    elif selection.get("kind") == "confirmation":
        for parent in [selection["reference_selection"], *selection["excluded_selections"]]:
            rows.update(excluded_validation_indices(parent, _depth=_depth+1))
    return rows


def make_confirmation(manifest: dict, labels: list[int], reference: dict, *,
                      exclusions=(), validation_per_class: int = 50,
                      seed: int = 20261012, _depth: int = 0) -> dict:
    """Keep fitted training rows; reserve unseen validation rows, never test."""
    if (type(validation_per_class) is not int or not 1 <= validation_per_class <= 100
            or type(seed) is not int or not 0 <= seed < 2**63
            or not isinstance(exclusions, (tuple, list)) or len(exclusions) > 8):
        raise ValueError("Confirmation requires 1-100 rows/class, a valid seed, and at most eight exclusions")
    if _depth > 3:
        raise ValueError("Confirmation ancestry is limited to three earlier rounds")
    if not isinstance(reference, dict) or reference.get("version") != 1 or "kind" in reference:
        raise ValueError("Confirmation reference must be its original version-one fitting pilot")
    parents = [reference, *exclusions]
    for parent in parents:
        if not isinstance(parent, dict):
            raise ValueError("Confirmation sources must be verified selections")
        if parent.get('version') == 3:
            raise ValueError('Use a flattened exposure ledger for later confirmation sources')
        if parent.get("kind") == "confirmation":
            validate_confirmation(parent, manifest, labels, _depth=_depth+1)
        else:
            validate_pilot(parent, manifest, labels)
    hashes = [parent["selection_sha256"] for parent in parents]
    if len(hashes) != len(set(hashes)):
        raise ValueError("Confirmation source selections must be distinct")
    excluded = set().union(*(excluded_validation_indices(parent, _depth=_depth) for parent in parents))
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


def validate_confirmation(selection: dict, manifest: dict, labels: list[int], *, _depth: int = 0) -> dict:
    if isinstance(selection, dict) and type(selection.get('version')) is int and selection['version'] == 3:
        from .exposure import validate_ledger_confirmation
        return validate_ledger_confirmation(selection, manifest, labels)
    if not isinstance(selection, dict) or type(selection.get("version")) is not int or selection["version"] != 2:
        raise ValueError("Confirmation selection requires integer version two")
    try:
        expected = make_confirmation(manifest, labels, selection["reference_selection"],
                                     exclusions=selection["excluded_selections"],
                                     validation_per_class=selection["per_class"]["validation"], seed=selection["seed"], _depth=_depth)
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
    parser.add_argument("--train-per-class", type=int)
    parser.add_argument("--validation-per-class", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20261007)
    parser.add_argument("--confirmation-of", type=Path, help="Preserve this fitted pilot's training rows and exclude its validation rows")
    parser.add_argument("--exclude-pilot", type=Path, action="append", default=[], help="Additional historical validation selection to exclude")
    parser.add_argument('--exposure-bundle', type=Path, help='Recheck original pinned source files and reserve against their flat ledger')
    parser.add_argument('--expected-ledger-sha256', help='Require the predeclared ledger fingerprint with --exposure-bundle')
    args = parser.parse_args()
    try:
        manifest = json.loads(args.manifest.read_text())
        train, test = CIFAR10Binary(args.root), CIFAR10Binary(args.root, train=False)
        if manifest != make_manifest(train, test, seed=manifest["seed"]):
            raise ValueError("Manifest does not match current data or the fixed split")
        if args.confirmation_of:
            if args.train_per_class is not None:
                raise ValueError("Confirmation preserves fitted training rows; do not set --train-per-class")
            reference = json.loads(args.confirmation_of.read_text())
            if args.exposure_bundle:
                if args.exclude_pilot or not args.expected_ledger_sha256:
                    raise ValueError('Exposure bundle requires --expected-ledger-sha256 and cannot mix --exclude-pilot')
                from .exposure import load_exposure_bundle, make_ledger_confirmation
                bundle = load_exposure_bundle(args.exposure_bundle, manifest, train.labels)
                ledger = bundle['ledger']
                if ledger['ledger_sha256'] != args.expected_ledger_sha256:
                    raise ValueError('Exposure ledger differs from its predeclared fingerprint')
                selection = make_ledger_confirmation(manifest, train.labels, reference, ledger,
                    validation_per_class=args.validation_per_class, seed=args.seed)
            else:
                if args.expected_ledger_sha256:
                    raise ValueError('--expected-ledger-sha256 requires --exposure-bundle')
                selection = make_confirmation(manifest, train.labels, reference,
                    exclusions=[json.loads(path.read_text()) for path in args.exclude_pilot],
                    validation_per_class=args.validation_per_class, seed=args.seed)
        else:
            if args.exclude_pilot or args.exposure_bundle or args.expected_ledger_sha256:
                raise ValueError('Exclusions/exposure history require --confirmation-of')
            selection = make_pilot(manifest, train.labels, train_per_class=10 if args.train_per_class is None else args.train_per_class,
                                  validation_per_class=args.validation_per_class, seed=args.seed)
        inputs = [args.manifest, args.confirmation_of, args.exposure_bundle, *args.exclude_pilot]
        if args.exposure_bundle:
            inputs.extend(Path(row['path']) for row in bundle['source_files'])
        if args.output.resolve() in {p.resolve() for p in inputs if p is not None}:
            raise ValueError('Selection output must not overwrite its source evidence')
        from .compare import write_report
        write_report(args.output, selection)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    print(json.dumps(dict(selection_sha256=selection["selection_sha256"],
        class_counts={name:dict(Counter(train.labels[i] for i in part["indices"]))
                      for name,part in selection["partitions"].items()}), indent=2))


if __name__ == "__main__":
    main()
