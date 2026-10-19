"""Controlled-width comparisons with shared pixels and exact initialization controls."""
from dataclasses import asdict,replace
from pathlib import Path
from statistics import mean,stdev

import torch

from .compare import fit_representation,validate_comparison_inputs,validate_probe_seeds
from .diagnostics import paired_diagnostics
from .features import _sha
from .ridge import validate_ridge_candidates
from .transfer_audit import _selected


def validate_widths(arms,candidates):
    if not isinstance(arms,dict) or not 2<=len(arms)<=3:raise ValueError('Use two or three distinct width arms')
    first=next(iter(arms.values()));reference=validate_comparison_inputs(first,candidates)
    anchor=reference[0].metadata;fixed={k:anchor.get(k) for k in ('completed_steps','training_settings','pooling','encoder','normalization')}
    if (type(fixed['completed_steps']) is not int or fixed['completed_steps']<1
            or not isinstance(fixed['training_settings'],dict)):raise ValueError('Widths require recorded training identity')
    widths=[];checkpoints=[]
    for name,data in arms.items():
        pair=validate_comparison_inputs(data,candidates);metadata=pair[0].metadata;config=metadata['config']
        width=config['embedding_dim'];widths.append(width);checkpoints.append(metadata.get('checkpoint_sha256'))
        if (name!=f'width{width}' or {k:metadata.get(k) for k in fixed}!=fixed
                or {k:v for k,v in config.items() if k!='embedding_dim'}!={k:v for k,v in anchor['config'].items() if k!='embedding_dim'}
                or data['random'][0].metadata.get('seed')!=fixed['training_settings'].get('seed')):
            raise ValueError('Width arms must vary only width and match exact initialization seeds')
        for item,expected in zip(pair,reference):
            if (item.indices!=expected.indices or not torch.equal(item.labels,expected.labels)
                    or item.manifest_sha256!=expected.manifest_sha256 or item.selection_sha256!=expected.selection_sha256):
                raise ValueError('Width arms require identical ordered examples')
        for item,expected in zip(data['pixels'],first['pixels']):
            if item.artifact_sha256!=expected.artifact_sha256 or item.metadata!=expected.metadata or not torch.equal(item.features,expected.features):
                raise ValueError('Widths must share the exact pixel artifacts')
    if widths!=sorted(set(widths)) or any(not _sha(h) for h in checkpoints) or len(set(checkpoints))!=len(checkpoints):
        raise ValueError('Use increasing unique widths and distinct checkpoint identities')
    return reference


def summarize(runs,arms):
    result={}
    for name in [*[f'{width}-{rep}' for width in arms for rep in ('checkpoint','random')],'pixels']:
        scores=[r['pixels']['validation']['accuracy'] if name=='pixels' else
            r['arms'][name.rsplit('-',1)[0]][name.rsplit('-',1)[1]]['validation']['accuracy'] for r in runs]
        result[name]=dict(accuracies=scores,mean_accuracy=mean(scores),sample_sd=stdev(scores) if len(scores)>1 else 0.)
    return result


def compare_arms(arms,candidates,seeds,*,probe_dir,ridge_alphas,validator,axis,limitation):
    reference=validator(arms,candidates);validate_probe_seeds(seeds);validate_ridge_candidates(ridge_alphas)
    first=next(iter(arms));runs=[]
    for index,seed in enumerate(seeds):
        configs=tuple(replace(c,seed=seed) for c in candidates);directory=Path(probe_dir)/f'seed{seed}'
        alphas=ridge_alphas if index==0 else None
        pixels,pixel_predictions=fit_representation(arms[first]['pixels'],configs,'pixels',directory,alphas)
        reports={};predictions={}
        for width,data in arms.items():
            reports[width]={};predictions[width]={}
            for name in ('checkpoint','random'):
                reports[width][name],predictions[width][name]=fit_representation(data[name],configs,f'{width}-{name}',directory,alphas)
        paired={head:{width:{f'vs_{name}':paired_diagnostics(reference[1].labels,pred,predictions[width]['checkpoint'][head])
            for name,pred in dict(random=predictions[width]['random'][head],pixels=pixel_predictions[head]).items()}
            for width in arms} for head in pixel_predictions}
        runs.append(dict(seed=seed,candidate_grid=[asdict(c) for c in configs],arms=reports,pixels=pixels,paired_validation=paired))
    return dict(version=1,**{axis:list(arms)},seeds=list(seeds),manifest_sha256=reference[0].manifest_sha256,
        selection_sha256=reference[0].selection_sha256,counts=dict(train=len(reference[0].labels),tuning=len(reference[1].labels)),
        ridge_alphas=list(ridge_alphas),runs=runs,summary=summarize(runs,arms),
        runtime=dict(torch=str(torch.__version__),cpu_threads=torch.get_num_threads()),test_evaluated=False,
        limitation=limitation)


def audit_arms(report,arms,candidates,seeds,*,ridge_alphas,validator,axis):
    reference=validator(arms,candidates);first=next(iter(arms));validate_probe_seeds(seeds)
    ridge_configs=[asdict(c) for c in validate_ridge_candidates(ridge_alphas)]
    if (report['version']!=1 or report[axis]!=list(arms) or report['seeds']!=list(seeds)
            or report['manifest_sha256']!=reference[0].manifest_sha256 or report['selection_sha256']!=reference[0].selection_sha256
            or report['counts']!=dict(train=len(reference[0].labels),tuning=len(reference[1].labels))
            or report['runtime']!=dict(torch=str(torch.__version__),cpu_threads=torch.get_num_threads())
            or report['test_evaluated'] is not False or report['ridge_alphas']!=list(ridge_alphas) or len(report['runs'])!=len(seeds)):
        raise ValueError('Width report identity/grid/runtime differs')
    verified=0
    for index,(seed,run) in enumerate(zip(seeds,report['runs'])):
        configs=[asdict(replace(c,seed=seed)) for c in candidates]
        if run['seed']!=seed or run['candidate_grid']!=configs or list(run['arms'])!=list(arms):raise ValueError('Width run configuration differs')
        predictions={};pixel={}
        sources={'pixels':(run['pixels'],arms[first]['pixels'])}|{f'{width}-{rep}':(run['arms'][width][rep],data[rep]) for width,data in arms.items() for rep in ('checkpoint','random')}
        for name,(row,pair) in sources.items():
            if row['train_artifact']!=pair[0].summary() or row['validation_artifact']!=pair[1].summary():raise ValueError('Width source artifacts differ')
            pred,_=_selected(row,pair,configs,'cross_entropy');heads=dict(adamw=pred);verified+=1
            if index==0:heads['ridge'],_=_selected(row['ridge'],pair,ridge_configs,'mean_squared_error');verified+=1
            elif 'ridge' in row:raise ValueError('Ridge fitted more than once')
            if name=='pixels':pixel=heads
            else:width,rep=name.rsplit('-',1);predictions.setdefault(width,{})[rep]=heads
        expected={head:{width:{f'vs_{name}':paired_diagnostics(reference[1].labels,pred,predictions[width]['checkpoint'][head])
            for name,pred in dict(random=predictions[width]['random'][head],pixels=pixel[head]).items()} for width in arms} for head in pixel}
        if expected!=run['paired_validation']:raise ValueError('Width paired errors do not reproduce')
    if report['summary']!=summarize(report['runs'],arms):raise ValueError('Width aggregates differ')
    return dict(verified_classifiers=verified,no_refitting=True,test_evaluated=False)


WIDTH_LIMITATION='Width changes capacity, feature dimensions and compute. Shared pixel heads are fitted once per classifier seed; initializations match within each width.'


def compare_widths(arms,candidates,seeds,*,probe_dir,ridge_alphas):
    return compare_arms(arms,candidates,seeds,probe_dir=probe_dir,ridge_alphas=ridge_alphas,
        validator=validate_widths,axis='widths',limitation=WIDTH_LIMITATION)


def audit_widths(report,arms,candidates,seeds,*,ridge_alphas):
    return audit_arms(report,arms,candidates,seeds,ridge_alphas=ridge_alphas,validator=validate_widths,axis='widths')
