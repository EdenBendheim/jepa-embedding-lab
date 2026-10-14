"""Execute the committed frozen-transfer protocol and audit cached retries."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import torch

from jepa_lab.cifar10 import CIFAR10Binary,make_manifest
from jepa_lab.compare import write_report
from jepa_lab.features import load_features
from jepa_lab.pretrained import WEIGHTS_SHA256,pretrained_metadata
from jepa_lab.pretrained_extract import extract_pretrained
from jepa_lab.probes import ProbeConfig
from jepa_lab.selection import validate_pilot
from jepa_lab.transfer_audit import audit_transfer_report
from jepa_lab.transfer_compare import ARMS,compare_transfer,load_transfer_sets

ROOT=Path("data/cifar-10-batches-bin")
PROTOCOL=Path("experiments/pretrained-sources-20261013.json")
PROTOCOL_SHA="e8b3bcd9e10a19dbef6473a44807af97c9fac07473a7b08eb4622aa29a3a941e"


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(directory=Path("runs/pretrained-20261013"),*,reuse_only=False):
    if sha(PROTOCOL)!=PROTOCOL_SHA:
        raise ValueError("Committed transfer protocol changed")
    declared=json.loads(PROTOCOL.read_text())
    for source in declared["sources"]:
        if sha(source["path"])!=source["sha256"]:
            raise ValueError("Declared historical source bytes changed")
    train,test=CIFAR10Binary(ROOT),CIFAR10Binary(ROOT,train=False)
    manifest=json.loads(Path(declared["manifest_path"]).read_text())
    if (manifest!=make_manifest(train,test,seed=manifest["seed"])
            or manifest["manifest_sha256"]!=declared["manifest_sha256"]):
        raise ValueError("Declared dataset or split changed")
    selection=validate_pilot(json.loads(Path(declared["selection_path"]).read_text()),manifest,train.labels)
    if selection["selection_sha256"]!=declared["selection_sha256"] or selection.get("kind","pilot")!="pilot":
        raise ValueError("Declared tuning selection changed")
    if declared["weights_sha256"]!=WEIGHTS_SHA256:
        raise ValueError("Declared pretrained weights changed")
    weights=Path(declared["weights_path"])
    if weights.exists() and sha(weights)!=WEIGHTS_SHA256:
        raise ValueError("Local pretrained weights changed")
    directory=Path(directory);features=directory/"features"
    # Validate every existing cache before extracting any missing partition.
    for partition in ("train","validation"):
        path=features/f"pretrained-{partition}.pt"
        if path.exists():
            verified=load_features(path,manifest=manifest,selection=selection,
                training_labels=train.labels,partition=partition)
            if verified.metadata!=pretrained_metadata():raise ValueError("Cached pretrained recipe differs")
        elif reuse_only:
            raise ValueError("Missing pretrained features in reuse-only mode")
    for partition in ("train","validation"):
        path=features/f"pretrained-{partition}.pt"
        if not path.exists():
            extract_pretrained(ROOT,declared["manifest_path"],declared["selection_path"],weights,
                partition=partition,batch_size=8,output=path)
    sets=load_transfer_sets(ROOT,declared["manifest_path"],declared["selection_path"],declared["feature_dir"],features)
    candidates=tuple(ProbeConfig(declared["steps"],lr,wd) for lr in declared["learning_rates"] for wd in declared["weight_decays"])
    comparison=directory/"comparison.json"
    if comparison.exists():
        report=json.loads(comparison.read_text())
        if (report["candidate_grid"]!=[asdict(c) for c in candidates]
                or report["seeds"]!=declared["seeds"] or report["ridge_alpha_grid"]!=declared["ridge_alphas"]):
            raise ValueError("Cached classifier budget differs from the committed protocol")
    elif reuse_only:
        raise ValueError("Missing comparison in reuse-only mode")
    else:
        report=compare_transfer(sets,candidates,declared["seeds"],probe_dir=directory/"probes",ridge_alphas=declared["ridge_alphas"])
    audit=audit_transfer_report(report,sets)
    if not comparison.exists():write_report(comparison,report)
    summary=dict(protocol_sha256=PROTOCOL_SHA,manifest_sha256=manifest["manifest_sha256"],
        selection_sha256=selection["selection_sha256"],counts=report["counts"],seeds=report["seeds"],
        candidate_grid=report["candidate_grid"],ridge_alpha_grid=report["ridge_alpha_grid"],
        summary=audit["summary"],ridge_accuracy={name:report["ridge_baseline"][name]["validation"]["accuracy"] for name in ARMS},
        verified_classifiers=audit["verified_classifiers"],paired_errors_reproduced=True,test_evaluated=False,
        sources={name:{part:{"artifact_sha256":item.artifact_sha256,"feature_metadata":item.metadata}
            for part,item in zip(("train","validation"),pair)} for name,pair in sets.items()},
        limitation=report["limitation"])
    write_report(directory/"summary.json",summary)
    return summary


def main():
    parser=argparse.ArgumentParser(description="Run the predeclared offline frozen-transfer pilot")
    parser.add_argument("--directory",type=Path,default=Path("runs/pretrained-20261013"))
    parser.add_argument("--reuse-only",action="store_true")
    args=parser.parse_args();torch.set_num_threads(1)
    try:summary=run(args.directory,reuse_only=args.reuse_only)
    except (ValueError,OSError,KeyError,TypeError,RuntimeError) as exc:parser.error(str(exc))
    print(json.dumps({key:summary[key] for key in ("counts","summary","ridge_accuracy","verified_classifiers","test_evaluated")},indent=2))


if __name__=="__main__":main()
