"""Audit a repeated matched-feature comparison from saved classifier states."""
from dataclasses import asdict,replace
from statistics import mean,stdev

import torch

from .compare import validate_comparison_inputs,validate_probe_seeds
from .diagnostics import paired_diagnostics
from .ridge import validate_ridge_candidates
from .transfer_audit import _selected


def audit_repeated_report(report,sets,candidates,seeds,*,ridge_alphas):
    reference=validate_comparison_inputs(sets,candidates);validate_probe_seeds(seeds)
    ridge_configs=[asdict(c) for c in validate_ridge_candidates(ridge_alphas)]
    try:
        if (report['version']!=1 or report['test_evaluated'] is not False or report['seeds']!=list(seeds)
                or len(report['runs'])!=len(seeds) or report['manifest_sha256']!=reference[0].manifest_sha256
                or report['selection_sha256']!=reference[0].selection_sha256
                or report['counts']!=dict(train=len(reference[0].indices),validation=len(reference[1].indices))):
            raise ValueError('Repeated comparison identity or declared seeds differ')
        values={name:[] for name in sets};verified=0
        for index,(seed,run) in enumerate(zip(seeds,report['runs'])):
            configs=[asdict(replace(c,seed=seed)) for c in candidates]
            if (run['candidate_grid']!=configs or set(run['representations'])!=set(sets)
                    or run['ridge_alpha_grid']!=(ridge_alphas if index==0 else None)
                    or run['runtime']!=dict(torch=str(torch.__version__),cpu_threads=torch.get_num_threads())
                    or run['test_evaluated'] is not False):
                raise ValueError('Repeated comparison grid, arms, runtime or boundary differ')
            predictions={};ridge_records={}
            for name,pair in sets.items():
                row=run['representations'][name]
                if row['train_artifact']!=pair[0].summary() or row['validation_artifact']!=pair[1].summary():
                    raise ValueError('Selected classifier sources changed')
                pred,accuracy=_selected(row,pair,configs,'cross_entropy')
                predictions[name]=dict(adamw=pred);values[name].append(accuracy);verified+=1
                if index==0:
                    predictions[name]['ridge'],_=_selected(row['ridge'],pair,ridge_configs,'mean_squared_error')
                    ridge_records[name]=row['ridge'];verified+=1
                elif 'ridge' in row:raise ValueError('Ridge must be selected once per representation')
            expected={head:{f'checkpoint_vs_{name}':paired_diagnostics(reference[1].labels,
                predictions[name][head],predictions['checkpoint'][head]) for name in ('random','pixels')}
                for head in predictions['checkpoint']}
            if run['paired_validation']!=expected:raise ValueError('Selected paired errors do not reproduce')
            if index==0 and report['ridge_baseline']!=ridge_records:raise ValueError('Ridge summary changed')
        summary={name:dict(validation_accuracies=scores,mean_accuracy=mean(scores),
            sample_standard_deviation=stdev(scores) if len(scores)>1 else 0.) for name,scores in values.items()}
        if report['summary']!=summary:raise ValueError('Classifier-seed summary does not reproduce')
        return dict(verified_classifiers=verified,summary=summary,paired_errors_reproduced=True,
            no_refitting=True,test_evaluated=False,
            limitation='Selected states and reported grid reproduce; losing candidate states were not saved or refitted')
    except (KeyError,TypeError,IndexError) as exc:raise ValueError('Malformed repeated comparison report') from exc
