"""Reproduce selected transfer classifiers and reported aggregates without fitting."""

from copy import deepcopy
from dataclasses import asdict,replace
from statistics import mean,stdev

import torch

from .diagnostics import paired_diagnostics
from .probe_artifacts import load_probe
from .probes import ProbeConfig
from .ridge import validate_ridge_candidates
from .transfer_compare import ARMS,LIMITATION,validate_transfer_inputs
from .compare import validate_probe_seeds


def _selected(record,pair,configs,metric):
    grid=record["probe_selection"]["candidates"]
    if [row["config"] for row in grid]!=configs:
        raise ValueError("Reported classifier candidates differ from the declared grid")
    chosen=record["probe_selection"]["selected_candidate"]
    if type(chosen) is not int or not 0<=chosen<len(grid):
        raise ValueError("Selected classifier index is invalid")
    expected=min(range(len(grid)),key=lambda i:(-grid[i]["validation"]["accuracy"],grid[i]["validation"][metric],i))
    if chosen!=expected or record["selected_config"]!=configs[chosen]:
        raise ValueError("Reported classifier selection does not follow the declared rule")
    saved=record["selected_probe_artifact"]
    probe,artifact=load_probe(saved["path"],*pair)
    if saved!={"path":saved["path"],**artifact} or asdict(probe.config)!=configs[chosen]:
        raise ValueError("Saved classifier identity/configuration changed")
    scores=artifact["scores"]
    if any(record[k]!=scores[k] or grid[chosen][k]!=scores[k] for k in ("training","validation")):
        raise ValueError("Reported selected scores do not reproduce")
    return probe.logits(pair[1].features).argmax(1),scores["validation"]["accuracy"]


def audit_transfer_report(report,sets):
    try:
        candidates=tuple(ProbeConfig(**value) for value in report["candidate_grid"])
        reference=validate_transfer_inputs(sets,candidates)
        seeds=report["seeds"];validate_probe_seeds(seeds)
        ridge=report["ridge_alpha_grid"]
        ridge_configs=[asdict(c) for c in validate_ridge_candidates(ridge)] if ridge is not None else None
        if (report["version"]!=1 or report["test_evaluated"] is not False or report["limitation"]!=LIMITATION
                or report["manifest_sha256"]!=reference[0].manifest_sha256
                or report["selection_sha256"]!=reference[0].selection_sha256
                or report["counts"]!=dict(train=len(reference[0].indices),validation=len(reference[1].indices))
                or report["runtime"]!=dict(torch=str(torch.__version__),cpu_threads=torch.get_num_threads())
                or len(report["runs"])!=len(seeds)):
            raise ValueError("Transfer report identity, runtime or evaluation boundary changed")
        accuracies={name:[] for name in ARMS};verified=0
        for index,(seed,run) in enumerate(zip(seeds,report["runs"])):
            if run["seed"]!=seed or set(run["representations"])!=set(ARMS):
                raise ValueError("Transfer run arms or seed differ")
            configs=[asdict(replace(c,seed=seed)) for c in candidates]
            predictions={};ridge_records={}
            for name,pair in sets.items():
                record=run["representations"][name]
                if record["train_artifact"]!=pair[0].summary() or record["validation_artifact"]!=pair[1].summary():
                    raise ValueError("Reported source artifacts differ from verified inputs")
                pred,accuracy=_selected(record,pair,configs,"cross_entropy")
                predictions[name]={"adamw":pred};accuracies[name].append(accuracy);verified+=1
                if index==0 and ridge_configs is not None:
                    predictions[name]["ridge"],_=_selected(record["ridge"],pair,ridge_configs,"mean_squared_error")
                    ridge_records[name]=record["ridge"];verified+=1
                elif "ridge" in record:
                    raise ValueError("Ridge classifiers should be fitted once per representation")
            expected={head:{f"checkpoint_vs_{name}":paired_diagnostics(reference[1].labels,
                predictions[name][head],predictions["checkpoint"][head]) for name in ("random","pixels","pretrained")}
                for head in predictions["checkpoint"]}
            if run["paired_validation"]!=expected:
                raise ValueError("Reported paired errors do not reproduce")
            if index==0 and report["ridge_baseline"]!=(ridge_records if ridge_configs is not None else None):
                raise ValueError("Reported ridge summary differs from verified states")
        expected_summary={name:dict(validation_accuracies=scores,mean_accuracy=mean(scores),
            sample_standard_deviation=stdev(scores) if len(scores)>1 else 0.0,
            feature_dim=sets[name][0].features.shape[1]) for name,scores in accuracies.items()}
        if report["summary"]!=expected_summary:
            raise ValueError("Reported classifier-seed aggregate does not reproduce")
        return dict(verified_classifiers=verified,summary=deepcopy(expected_summary),
            test_evaluated=False,no_refitting=True,paired_errors_reproduced=True,
            limitation="Selected states/scores and reported selection rules reproduced; losing classifier states were not saved or refitted")
    except (KeyError,TypeError,IndexError) as exc:
        raise ValueError("Malformed transfer comparison report") from exc
