"""Audit paired saved classifiers without fitting or sampling raw images."""

import argparse
import json
from pathlib import Path

import torch

from .cifar10 import CIFAR10Binary, make_manifest
from .compare import write_report
from .diagnostics import paired_diagnostics
from .features import load_features
from .probe_artifacts import load_probe, probe_context
from .selection import validate_pilot


def compare_saved_probes(baseline, candidate, baseline_path, candidate_path):
    for pair in (baseline, candidate):
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise ValueError("Each classifier requires its verified train/validation pair")
        probe_context(*pair)
    for left, right in zip(baseline, candidate):
        if (left.indices != right.indices or not torch.equal(left.labels, right.labels)
                or left.manifest_sha256 != right.manifest_sha256
                or left.selection_sha256 != right.selection_sha256):
            raise ValueError("Paired classifiers require identical ordered examples and labels")
    before, before_summary = load_probe(baseline_path, *baseline)
    after, after_summary = load_probe(candidate_path, *candidate)
    return dict(version=1, task="Paired saved-classifier validation audit",
        baseline_probe=before_summary, candidate_probe=after_summary,
        paired_validation=paired_diagnostics(baseline[1].labels,
            before.logits(baseline[1].features).argmax(1), after.logits(candidate[1].features).argmax(1)),
        refitted=False, test_evaluated=False,
        limitation="Reproduces descriptive errors on the original validation examples; not new evaluation evidence")


def run_audit(root, manifest_path, selection_path, baseline_paths, candidate_paths):
    if any(not isinstance(paths, (tuple, list)) or len(paths) != 3 for paths in (baseline_paths, candidate_paths)):
        raise ValueError("Each audit source requires train features, validation features, and a probe path")
    train, test = CIFAR10Binary(root), CIFAR10Binary(root, train=False)
    manifest = json.loads(Path(manifest_path).read_text())
    if manifest != make_manifest(train, test, seed=manifest["seed"]):
        raise ValueError("Dataset files differ from the locked manifest")
    selection = validate_pilot(json.loads(Path(selection_path).read_text()), manifest, train.labels)
    pairs = [tuple(load_features(path, manifest=manifest, selection=selection,
                training_labels=train.labels, partition=partition)
                for path, partition in zip(paths[:2], ("train", "validation")))
             for paths in (baseline_paths, candidate_paths)]
    return compare_saved_probes(*pairs, baseline_paths[2], candidate_paths[2])


def main():
    parser = argparse.ArgumentParser(description="Compare saved probes on bound validation features; never refit")
    parser.add_argument("root", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, nargs=3, required=True, metavar=("TRAIN", "VALIDATION", "PROBE"))
    parser.add_argument("--candidate", type=Path, nargs=3, required=True, metavar=("TRAIN", "VALIDATION", "PROBE"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    try:
        report = run_audit(args.root, args.manifest, args.selection, args.baseline, args.candidate)
        write_report(args.output, report)
    except (ValueError, OSError, KeyError, TypeError, RuntimeError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
