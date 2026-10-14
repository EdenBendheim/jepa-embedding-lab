"""Matched downstream probes for JEPA controls and a frozen transfer reference."""

from dataclasses import asdict, replace
import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean, stdev

import torch

from .compare import REPRESENTATIONS, fit_representation, validate_comparison_inputs, validate_probe_seeds
from .compare import write_report
from .cifar10 import CIFAR10Binary, make_manifest
from .diagnostics import paired_diagnostics
from .features import FeatureSet, load_features
from .pretrained import pretrained_metadata
from .probes import ProbeConfig, validate_candidates, validate_probe_data
from .ridge import validate_ridge_candidates
from .selection import validate_pilot

ARMS = (*REPRESENTATIONS,"pretrained")
LIMITATION = ("Exploratory validation-selected transfer comparison, not official test accuracy or "
              "independent confirmation. ImageNet supervision, architecture, parameter count, input "
              "resolution and feature width differ; a score gap cannot isolate the learning objective.")


def validate_transfer_inputs(sets,candidates):
    if set(sets) != set(ARMS):
        raise ValueError("Transfer comparison requires checkpoint, matching random, pixels and pretrained")
    reference = validate_comparison_inputs({name:sets[name] for name in REPRESENTATIONS},candidates)
    pair = sets["pretrained"]
    if (not isinstance(pair,(tuple,list)) or len(pair)!=2
            or any(not isinstance(item,FeatureSet) for item in pair)):
        raise ValueError("Pretrained reference requires verified training and validation artifacts")
    for item,expected in zip(pair,reference):
        if (item.partition!=expected.partition or item.selection_kind!="pilot"
                or item.indices!=expected.indices or not torch.equal(item.labels,expected.labels)
                or item.manifest_sha256!=expected.manifest_sha256
                or item.selection_sha256!=expected.selection_sha256
                or item.metadata!=pretrained_metadata() or item.features.shape[1]!=512):
            raise ValueError("Transfer reference examples, provenance, dimension or fitting boundary differ")
        validate_probe_data(item.features,item.labels,dimension=512,fitting=item.partition=="train")
    # A frozen dataclass still contains mutable tensors. Recheck all verified content before any fit.
    for items in sets.values():
        for item in items:
            values_sha=hashlib.sha256(item.features.contiguous().numpy().tobytes()).hexdigest()
            labels_sha=hashlib.sha256(item.labels.contiguous().numpy().tobytes()).hexdigest()
            if item.features_sha256!=values_sha or item.labels_sha256!=labels_sha:
                raise ValueError("Verified feature content was changed before transfer comparison")
    return reference


def compare_transfer(sets,candidates,seeds,*,probe_dir=None,ridge_alphas=None):
    validate_probe_seeds(seeds)
    reference=validate_transfer_inputs(sets,candidates)
    if ridge_alphas is not None:validate_ridge_candidates(ridge_alphas)
    runs=[]
    for index,seed in enumerate(seeds):
        configs=tuple(replace(config,seed=seed) for config in candidates)
        reports,predictions={},{}
        for name in ARMS:
            reports[name],predictions[name]=fit_representation(sets[name],configs,name,
                Path(probe_dir)/f"seed-{seed}" if probe_dir is not None else None,
                ridge_alphas if index==0 else None)
        paired={head:{f"checkpoint_vs_{name}":paired_diagnostics(reference[1].labels,
            predictions[name][head],predictions["checkpoint"][head])
            for name in ("random","pixels","pretrained")} for head in predictions["checkpoint"]}
        runs.append(dict(seed=seed,representations=reports,paired_validation=paired))
    summary={}
    for name in ARMS:
        scores=[run["representations"][name]["validation"]["accuracy"] for run in runs]
        summary[name]=dict(validation_accuracies=scores,mean_accuracy=mean(scores),
            sample_standard_deviation=stdev(scores) if len(scores)>1 else 0.0,
            feature_dim=sets[name][0].features.shape[1])
    return dict(version=1,task="Frozen transfer reference on matched CIFAR-10 pilot examples",
        manifest_sha256=reference[0].manifest_sha256,selection_sha256=reference[0].selection_sha256,
        counts=dict(train=len(reference[0].indices),validation=len(reference[1].indices)),
        seeds=list(seeds),candidate_grid=[asdict(c) for c in candidates],ridge_alpha_grid=ridge_alphas,
        runs=runs,summary=summary,ridge_baseline={name:runs[0]["representations"][name]["ridge"]
            for name in ARMS} if ridge_alphas is not None else None,
        runtime=dict(torch=str(torch.__version__),cpu_threads=torch.get_num_threads()),
        test_evaluated=False,limitation=LIMITATION)


def load_transfer_sets(root,manifest_path,selection_path,feature_dir,pretrained_dir):
    """Read all eight artifacts before any fitting; do not sample raw images."""
    train,test=CIFAR10Binary(root),CIFAR10Binary(root,train=False)
    manifest=json.loads(Path(manifest_path).read_text())
    if manifest!=make_manifest(train,test,seed=manifest["seed"]):
        raise ValueError("Dataset differs from the fixed manifest")
    selection=validate_pilot(json.loads(Path(selection_path).read_text()),manifest,train.labels)
    if selection.get("kind","pilot")!="pilot":
        raise ValueError("Transfer tuning cannot use confirmation examples")
    sets={}
    for name in ARMS:
        directory=Path(pretrained_dir if name=="pretrained" else feature_dir)
        sets[name]=tuple(load_features(directory/f"{name}-{partition}.pt",manifest=manifest,
            selection=selection,training_labels=train.labels,partition=partition)
            for partition in ("train","validation"))
    return sets


def run_transfer(root,manifest_path,selection_path,feature_dir,pretrained_dir,*,
                 candidates,seeds,probe_dir=None,ridge_alphas=None):
    validate_candidates(candidates);validate_probe_seeds(seeds)
    if ridge_alphas is not None:validate_ridge_candidates(ridge_alphas)
    sets=load_transfer_sets(root,manifest_path,selection_path,feature_dir,pretrained_dir)
    return compare_transfer(sets,candidates,seeds,probe_dir=probe_dir,ridge_alphas=ridge_alphas)


def main():
    parser=argparse.ArgumentParser(description="Compare verified saved JEPA controls and a frozen transfer reference")
    parser.add_argument("root",type=Path)
    parser.add_argument("--manifest",type=Path,required=True)
    parser.add_argument("--selection",type=Path,required=True)
    parser.add_argument("--feature-dir",type=Path,required=True)
    parser.add_argument("--pretrained-dir",type=Path,required=True)
    parser.add_argument("--probe-dir",type=Path,required=True)
    parser.add_argument("--probe-seeds",type=int,nargs="+",default=[29,31,37])
    parser.add_argument("--steps",type=int,default=200)
    parser.add_argument("--learning-rates",type=float,nargs="+",default=[.0001,.001,.01,.05])
    parser.add_argument("--weight-decays",type=float,nargs="+",default=[.01,.1])
    parser.add_argument("--ridge-alphas",type=float,nargs="+",default=[.01,.1,1,10])
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args();torch.set_num_threads(1)
    try:
        candidates=tuple(ProbeConfig(args.steps,lr,wd) for lr in args.learning_rates for wd in args.weight_decays)
        report=run_transfer(args.root,args.manifest,args.selection,args.feature_dir,args.pretrained_dir,
            candidates=candidates,seeds=args.probe_seeds,probe_dir=args.probe_dir,ridge_alphas=args.ridge_alphas)
        write_report(args.output,report)
    except (ValueError,OSError,KeyError,TypeError,RuntimeError) as exc:
        parser.error(str(exc))
    print(json.dumps(report,indent=2))


if __name__=="__main__":main()
