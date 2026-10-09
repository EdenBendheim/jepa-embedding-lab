"""Reproduce a saved probe's validation score without fitting a classifier."""

import argparse
import json
from pathlib import Path

import torch

from .cifar10 import CIFAR10Binary, make_manifest
from .compare import write_report
from .features import load_features
from .probe_artifacts import load_probe
from .selection import validate_pilot


def score_saved_probe(root: Path, manifest_path: Path, selection_path: Path,
                      train_path: Path, validation_path: Path, probe_path: Path) -> dict:
    train, test = CIFAR10Binary(root), CIFAR10Binary(root, train=False)
    manifest = json.loads(Path(manifest_path).read_text())
    if manifest != make_manifest(train, test, seed=manifest["seed"]):
        raise ValueError("Dataset files differ from the locked manifest")
    selection = validate_pilot(json.loads(Path(selection_path).read_text()), manifest, train.labels)
    sources = tuple(load_features(path, manifest=manifest, selection=selection,
                                  training_labels=train.labels, partition=partition)
                    for path, partition in ((train_path, "train"), (validation_path, "validation")))
    _, summary = load_probe(probe_path, *sources)
    return dict(version=1, task="Reproduce bound fitted-probe scores", probe=summary,
                refitted=False, test_evaluated=False,
                limitation="Rechecks the original validation-selected result; not independent confirmation or test accuracy")


def main():
    parser = argparse.ArgumentParser(description="Recheck saved validation scores without refitting or test access")
    parser.add_argument("root", type=Path)
    parser.add_argument("probe", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--train-features", type=Path, required=True)
    parser.add_argument("--validation-features", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    try:
        report = score_saved_probe(args.root, args.manifest, args.selection,
                                   args.train_features, args.validation_features, args.probe)
        if args.output:
            write_report(args.output, report)
    except (ValueError, OSError, KeyError, TypeError, RuntimeError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
