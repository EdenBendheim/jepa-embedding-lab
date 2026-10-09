"""Matched-data, validation-selected checkpoint/random/pixel pilot comparisons."""

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import tempfile
from time import perf_counter

import torch

from .cifar10 import CIFAR10Binary, make_manifest
from .embeddings import extract_cifar10, load_encoder
from .features import FeatureSet, load_features
from .model import ModelConfig
from .probes import ProbeConfig, select_probe, validate_candidates, validate_probe_data
from .probe_artifacts import STATE_FIELDS, load_probe, probe_context, save_probe, state_fingerprint
from .selection import validate_pilot


REPRESENTATIONS = ("checkpoint", "random", "pixels")


def compare_features(sets: dict[str, tuple[FeatureSet, FeatureSet]], candidates: tuple[ProbeConfig, ...], *,
                     probe_dir: Path | None = None) -> dict:
    validate_candidates(candidates)
    if set(sets) != set(REPRESENTATIONS):
        raise ValueError("A fair pilot requires checkpoint, random, and pixel representations")
    for name, pair in sets.items():
        if not isinstance(pair, (tuple,list)) or len(pair) != 2 or any(not isinstance(item,FeatureSet) for item in pair):
            raise ValueError("Every representation needs verified train and validation feature sets")
        train, validation = pair
        if (train.partition != "train" or validation.partition != "validation"
                or set(train.indices).intersection(validation.indices)
                or train.manifest_sha256 != validation.manifest_sha256
                or train.selection_sha256 != validation.selection_sha256 or train.metadata != validation.metadata
                or train.metadata.get("representation") != name):
            raise ValueError("Feature pair splits, provenance, or representation identity differ")
        validate_probe_data(train.features, train.labels, fitting=True)
        validate_probe_data(validation.features, validation.labels, dimension=train.features.shape[1])
        if train.metadata.get("normalization") != "x / 127.5 - 1":
            raise ValueError("All representations must retain the same fixed input normalization")
        if name == "pixels" and (train.features.shape[1] != 3072 or train.metadata.get("pooling") != "flatten NCHW RGB values"):
            raise ValueError("The pixel baseline must retain the declared NCHW flattening contract")
        if len(train.indices) != len(train.labels) or len(validation.indices) != len(validation.labels):
            raise ValueError("Every feature row must retain its selected example index")
    reference = sets["checkpoint"]
    for pair in sets.values():
        for item, expected in zip(pair,reference):
            if (item.manifest_sha256 != expected.manifest_sha256 or item.selection_sha256 != expected.selection_sha256
                    or item.indices != expected.indices or not torch.equal(item.labels,expected.labels)):
                raise ValueError("All representations must use the exact same ordered examples and labels")
    learned, random = reference[0].metadata, sets["random"][0].metadata
    if (not isinstance(learned.get("config"),dict) or learned["config"] != random.get("config")
            or learned.get("pooling") != "mean of all 64 patch tokens"
            or learned.get("encoder") not in ("context","target")
            or reference[0].features.shape[1] != learned["config"].get("embedding_dim")
            or sets["random"][0].features.shape[1] != reference[0].features.shape[1]
            or learned.get("encoder") != random.get("encoder") or learned.get("pooling") != random.get("pooling")):
        raise ValueError("The random baseline must match checkpoint architecture, encoder choice, and pooling")
    reports = {}
    for name in REPRESENTATIONS:
        train, validation = sets[name]
        start = perf_counter()
        probe, result = select_probe(train.features, train.labels, validation.features, validation.labels, candidates)
        selected = result["candidates"][result["selected_candidate"]]
        artifact = None
        if probe_dir is not None:
            state = {field: getattr(probe, field) for field in STATE_FIELDS}
            fingerprint = state_fingerprint(state, asdict(probe.config), probe_context(train, validation))
            path = Path(probe_dir) / f"{name}-{fingerprint}.pt"
            # Content-addressed names preserve files referenced by earlier reports.
            if path.exists():
                _, artifact = load_probe(path, train, validation)
                if artifact["state_sha256"] != fingerprint:
                    raise ValueError("Existing content-addressed probe differs from selected state")
            else:
                artifact = save_probe(path, probe, train, validation)
            artifact = dict(path=str(path), **artifact)
        reports[name] = dict(train_artifact=train.summary(), validation_artifact=validation.summary(),
                            probe_selection=result, selected_config=selected["config"],
                            training=selected["training"], validation=selected["validation"],
                            fitting_and_scoring_seconds=perf_counter()-start, selected_probe_artifact=artifact)
    return dict(version=1, task="Frozen ten-class CIFAR-10 pilot classification",
                manifest_sha256=reference[0].manifest_sha256, selection_sha256=reference[0].selection_sha256,
                counts=dict(train=len(reference[0].indices),validation=len(reference[1].indices)),
                candidate_grid=[asdict(config) for config in candidates], representations=reports,
                runtime=dict(torch=str(torch.__version__),cpu_threads=torch.get_num_threads()),
                test_evaluated=False,
                limitation="Small single-seed validation-selected pilot; not official test accuracy, independent confirmation, or a full-training result")


def run_comparison(root: Path, manifest_path: Path, selection_path: Path, feature_dir: Path, *,
                   candidates: tuple[ProbeConfig, ...], checkpoint: Path | None = None,
                   encoder: str = "context", random_seed: int = 29, probe_dir: Path | None = None) -> dict:
    validate_candidates(candidates)
    train, test = CIFAR10Binary(root), CIFAR10Binary(root,train=False)
    manifest = json.loads(Path(manifest_path).read_text())
    if manifest != make_manifest(train,test,seed=manifest["seed"]):
        raise ValueError("Current dataset files differ from the locked manifest")
    selection = validate_pilot(json.loads(Path(selection_path).read_text()),manifest,train.labels)
    feature_dir = Path(feature_dir)
    if checkpoint is not None:
        _, metadata = load_encoder(checkpoint,manifest["manifest_sha256"])
        config = ModelConfig(**metadata["config"])
        for representation in REPRESENTATIONS:
            for partition in ("train","validation"):
                extract_cifar10(root,manifest_path,partition=partition,representation=representation,
                    checkpoint=checkpoint if representation == "checkpoint" else None,
                    config=config if representation == "random" else None,encoder=encoder,seed=random_seed,
                    selection_path=selection_path,output=feature_dir/f"{representation}-{partition}.pt")
    sets = {name:tuple(load_features(feature_dir/f"{name}-{partition}.pt",manifest=manifest,
                        selection=selection,training_labels=train.labels,partition=partition)
                       for partition in ("train","validation")) for name in REPRESENTATIONS}
    # All six artifacts pass identity/fairness checks before any classifier is fitted.
    return compare_features(sets,candidates,probe_dir=probe_dir)


def write_report(path: Path, report: dict):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.",dir=path.parent)
    try:
        with os.fdopen(descriptor,"w") as stream:
            json.dump(report,stream,indent=2,allow_nan=False); stream.write("\n")
        os.replace(temporary,path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)


def main():
    parser = argparse.ArgumentParser(description="Compare matched balanced pilot features without test evaluation")
    parser.add_argument("root",type=Path)
    parser.add_argument("--manifest",type=Path,required=True)
    parser.add_argument("--selection",type=Path,required=True)
    parser.add_argument("--feature-dir",type=Path,required=True,help="Ignored local directory for six .pt feature artifacts")
    parser.add_argument("--probe-dir",type=Path,help="Save selected fitted probes in an ignored local directory")
    parser.add_argument("--checkpoint",type=Path,help="Extract fresh matched features; omit to verify/reuse existing artifacts")
    parser.add_argument("--encoder",choices=("context","target"),default="context")
    parser.add_argument("--random-seed",type=int,default=29)
    parser.add_argument("--probe-seed",type=int,default=29)
    parser.add_argument("--steps",type=int,default=200)
    parser.add_argument("--learning-rates",type=float,nargs="+",default=[0.0001,0.001,0.01,0.05])
    parser.add_argument("--weight-decays",type=float,nargs="+",default=[0.01,0.1])
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    try:
        candidates = tuple(ProbeConfig(args.steps,lr,wd,args.probe_seed)
                           for lr in args.learning_rates for wd in args.weight_decays)
        report = run_comparison(args.root,args.manifest,args.selection,args.feature_dir,candidates=candidates,
                                 checkpoint=args.checkpoint,encoder=args.encoder,random_seed=args.random_seed,probe_dir=args.probe_dir)
        write_report(args.output,report)
    except (ValueError,OSError,KeyError,TypeError,RuntimeError) as exc:
        parser.error(str(exc))
    print(json.dumps(report,indent=2))


if __name__ == "__main__":
    main()
