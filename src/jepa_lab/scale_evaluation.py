"""Frozen evaluation of pinned scale runs, with conservative exposure recording."""
import json
from pathlib import Path
from statistics import mean, stdev

from .compare import write_report
from .confirmation import score_confirmation
from .confirmation_compare import compare_confirmed_probes
from .embeddings import extract_cifar10
from .exposure import extend_exposure_bundle, exposure_summary, load_exposure_bundle
from .features import load_features
from .model import config_dict
from .neighbor_workflow import neighbor_record
from .neighbor_confirmation import score_neighbor_confirmation
from .neighbors import NeighborConfig
from .pooling import POOLING
from .probes import ProbeConfig
from .scale_compare import compare_scales, audit_scales
from .scale_run import file_sha


def source_key(scale,seed,rep):
    return 'pixels' if rep=='pixels' else f'scale{scale}-seed{seed}-{rep}'


def register_observation(run,d,manifest,fresh,labels,lock_path,reuse_only):
    """An interrupted score attempt must not make potentially seen rows fresh again."""
    selected = run.directory/'confirmation-selection.json'
    updated = extend_exposure_bundle(d['exposure_bundle_path'],manifest,labels,
        [(selected,file_sha(selected))],expected_ledger_sha256=d['ledger_sha256'])
    bundle_path = run.directory/'observed-exposure.json'
    receipt_path = run.directory/'observation-receipt.json'
    receipt = dict(version=1,protocol_sha256=run.expected_sha,
        selection_sha256=fresh['selection_sha256'],selection_lock_sha256=file_sha(lock_path),
        bundle_sha256=updated['bundle_sha256'],
        policy='Conservatively consume the reservation before fresh extraction or scoring; retain even after an interrupted attempt')
    for path,expected in ((bundle_path,updated),(receipt_path,receipt)):
        if path.exists():
            if json.loads(path.read_text())!=expected:
                raise ValueError('Existing scale observation evidence differs')
        elif reuse_only:
            raise ValueError('Scale observation evidence missing in reuse-only mode')
    if not bundle_path.exists():write_report(bundle_path,updated)
    if not receipt_path.exists():write_report(receipt_path,receipt)
    restored=load_exposure_bundle(bundle_path,manifest,labels)
    return exposure_summary(restored['ledger'],manifest,labels)|dict(
        registered_before_fresh_access=True,conservative_on_interruption=True)


def feature(run,d,manifest,selection,labels,scale,seed,rep,partition,reuse_only):
    directory=run.directory
    key=source_key(scale,seed,rep);suffix='fresh' if selection.get('kind')=='confirmation' else partition
    path=Path(directory)/'features'/f'{key}-{suffix}.pt'
    if not path.exists():
        if reuse_only:raise ValueError('Cached scale feature is missing')
        extract_cifar10(run.root,d['manifest_path'],representation=rep,partition=partition,
            checkpoint=run.checkpoint(d,scale,seed) if rep=='checkpoint' else None,
            config=run.config(d,scale) if rep=='random' else None,seed=seed if seed is not None else 29,
            selection_path=Path(directory)/'confirmation-selection.json' if suffix=='fresh' else d['selection_path'],
            pooling=d['pooling'],encoder=d['encoder'],output=path)
    value=load_features(path,manifest=manifest,selection=selection,training_labels=labels,partition=partition)
    if rep!='pixels':
        if value.metadata['config']!=config_dict(run.config(d,scale)) or value.metadata['encoder']!=d['encoder']:
            raise ValueError('Cached scale config/encoder differs')
        if rep=='random' and value.metadata['seed']!=seed:raise ValueError('Random initialization control differs')
        if rep=='checkpoint' and value.metadata['checkpoint_sha256']!=file_sha(run.checkpoint(d,scale,seed)):
            raise ValueError('Cached features refer to a different scale checkpoint')
    if rep!='pixels' and value.metadata['pooling'] != POOLING[d['pooling']]:
        raise ValueError('Cached scale pooling differs')
    return value


def evaluation(run, *, reuse_only=False):
    directory=run.directory;d,manifest,ref,fresh,labels=run.inputs()
    completed=run.training(reuse_only=True)
    candidates=tuple(ProbeConfig(d['probe_steps'],lr,wd,d['classifier_seeds'][0]) for lr in d['learning_rates'] for wd in d['weight_decays'])
    neighbor_configs=[NeighborConfig(k) for k in d['neighbor_counts']]
    sources={};selected_linear=[];selected_neighbors={};verified=0
    pixel_pair=tuple(feature(run,d,manifest,ref,labels,None,None,'pixels',p,reuse_only) for p in ('train','validation'))
    sources['pixels']=pixel_pair
    for seed in d['training_seeds']:
        arms={}
        for scale in d['position_scales']:
            data=dict(pixels=pixel_pair)
            for rep in ('checkpoint','random'):
                key=source_key(scale,seed,rep)
                pair=tuple(feature(run,d,manifest,ref,labels,scale,seed,rep,p,reuse_only) for p in ('train','validation'))
                data[rep]=pair;sources[key]=pair
            arms[f'scale{scale}']=data
        path=directory/'comparisons'/f'seed{seed}.json'
        if not path.exists():
            if reuse_only:raise ValueError('Selected scale comparison missing')
            report=compare_scales(arms,candidates,d['classifier_seeds'],ridge_alphas=d['ridge_alphas'],probe_dir=directory/'probes'/f'encoder{seed}')
            write_report(path,report)
        report=json.loads(path.read_text());verified+=audit_scales(report,arms,candidates,d['classifier_seeds'],ridge_alphas=d['ridge_alphas'])['verified_classifiers']
        for probe_run in report['runs']:
            rows=[(None,'pixels',probe_run['pixels'])]+[(w,rep,probe_run['arms'][f'scale{w}'][rep]) for w in d['position_scales'] for rep in ('checkpoint','random')]
            for scale,rep,row in rows:
                for head,scored in [('adamw',row)]+([('ridge',row['ridge'])] if 'ridge' in row else []):
                    selected_linear.append(dict(scale=scale,training_seed=seed,representation=rep,head=head,classifier_seed=probe_run['seed'] if head=='adamw' else None,
                        source_key=source_key(scale,seed,rep),selected_config=scored['selected_config'],original_tuning=scored['validation'],
                        path=scored['selected_probe_artifact']['path'],state_sha256=scored['selected_probe_artifact']['state_sha256'],artifact_sha256=scored['selected_probe_artifact']['artifact_sha256']))
    for key,pair in sources.items():
        row=neighbor_record(directory/'neighbors'/f'{key}.json',pair,neighbor_configs,reuse_only)
        selected_neighbors[key]=row
    lock=dict(linear=selected_linear,neighbors=selected_neighbors)
    lock_path=directory/'selection-lock.json'
    if lock_path.exists() and json.loads(lock_path.read_text())!=lock:raise ValueError('Selected scale/classifier lock differs')
    if not lock_path.exists():
        if reuse_only:raise ValueError('Scale selection lock missing')
        write_report(lock_path,lock)
    # Persist conservative observation history before any fresh extraction/scoring.
    exposure = register_observation(run,d,manifest,fresh,labels,lock_path,reuse_only)
    fresh_sources={'pixels':feature(run,d,manifest,fresh,labels,None,None,'pixels','validation',reuse_only)}
    for scale in d['position_scales']:
        for seed in d['training_seeds']:
            for rep in ('checkpoint','random'):
                fresh_sources[source_key(scale,seed,rep)]=feature(run,d,manifest,fresh,labels,scale,seed,rep,'validation',reuse_only)
    scored_rows=[]
    for row in selected_linear:
        key=row['source_key'];scored=score_confirmation(row['path'],*sources[key],fresh_sources[key],fresh)
        scored_rows.append({k:v for k,v in row.items() if k!='path'}|dict(confirmation=scored['confirmation']))
    for key,row in selected_neighbors.items():
        scored=score_neighbor_confirmation(row['path'],*sources[key],fresh_sources[key],fresh)
        if key=='pixels':scale,seed,rep=None,None,'pixels'
        else:
            w,s,rep=key.split('-');scale,seed=float(w[len('scale'):]),int(s[4:])
        scored_rows.append(dict(scale=scale,training_seed=seed,representation=rep,head='knn',classifier_seed=None,source_key=key,
            selected_config=row['index']['config'],state_sha256=row['index']['state_sha256'],artifact_sha256=row['index']['artifact_sha256'],
            original_tuning=row['index']['scores']['validation'],confirmation=scored['confirmation']))
    paired_scales=[]
    for seed in d['training_seeds']:
        for head,classifier_seed in [('adamw',value) for value in d['classifier_seeds']]+[('ridge',None)]:
            rows=[next(r for r in selected_linear if r['training_seed']==seed and r['scale']==scale
                and r['representation']=='checkpoint' and r['head']==head and r['classifier_seed']==classifier_seed)
                for scale in d['position_scales']]
            after,before=rows  # Primary contrast: small scale minus scale one.
            baseline=(*sources[before['source_key']],fresh_sources[before['source_key']])
            candidate=(*sources[after['source_key']],fresh_sources[after['source_key']])
            paired=compare_confirmed_probes(baseline,candidate,before['path'],after['path'],fresh)
            paired_scales.append(dict(baseline_scale=1.,candidate_scale=.02,training_seed=seed,head=head,classifier_seed=classifier_seed,
                baseline_state_sha256=before['state_sha256'],candidate_state_sha256=after['state_sha256'],
                baseline_accuracy=paired['baseline_accuracy'],candidate_accuracy=paired['candidate_accuracy'],
                paired_confirmation=paired['paired_confirmation']))
    aggregates={}
    for scale,rep in [(w,r) for w in d['position_scales'] for r in ('checkpoint','random')]+[(None,'pixels')]:
        group='pixels' if rep=='pixels' else f'scale{scale}-{rep}';outcomes={}
        for head in ('adamw','ridge','knn'):
            rows=[r for r in scored_rows if r['scale']==scale and r['representation']==rep and r['head']==head]
            if rep=='pixels':
                unique={}
                for row in rows:
                    key=row['classifier_seed'];signature=(row['state_sha256'],row['original_tuning'],row['confirmation'])
                    if key in unique and unique[key][1]!=signature:raise ValueError('Shared pixel states/results differ')
                    unique[key]=(row,signature)
                rows=[value[0] for value in unique.values()]
            scores={}
            groups=[rows] if rep=='pixels' else [[r for r in rows if r['training_seed']==s] for s in d['training_seeds']]
            for phase in ('original_tuning','confirmation'):
                values=[mean(r[phase]['accuracy'] for r in group_rows) for group_rows in groups]
                scores[phase]=dict(encoder_means=values,mean_accuracy=mean(values),encoder_mean_sample_sd=stdev(values) if len(values)>1 else 0.)
            outcomes[head]=scores
        aggregates[group]=outcomes
    training_metadata={key:{k:v for k,v in row.items() if k not in ('history','limitation')} for key,row in completed.items()}
    excluded=set(fresh['exposure_ledger']['observed_validation_indices'])
    result=dict(version=1,date='2026-10-22',protocol_sha256=run.expected_sha,manifest_sha256=manifest['manifest_sha256'],
        reference_selection_sha256=ref['selection_sha256'],confirmation_selection_sha256=fresh['selection_sha256'],
        counts=dict(train=len(ref['partitions']['train']['indices']),tuning=len(ref['partitions']['validation']['indices']),confirmation=len(fresh['partitions']['validation']['indices']),excluded_observed_validation=len(excluded)),
        exposure=exposure,selection_lock_sha256=file_sha(lock_path),
        primary_delta=aggregates['scale0.02-checkpoint']['adamw']['confirmation']['mean_accuracy']-aggregates['scale1.0-checkpoint']['adamw']['confirmation']['mean_accuracy'],
        summary=aggregates,paired_scales=paired_scales,training=training_metadata,classifiers=scored_rows,verified_linear_states=verified,verified_neighbor_states=len(selected_neighbors),
        unique_linear_state_count=len({r['state_sha256'] for r in selected_linear}),actual_image_order_matched=True,
        all_states_locked_before_confirmation=True,confirmation_refitting=False,test_evaluated=False,
        limitation='Shared confirmation rows; seed variation is descriptive. Only positional amplitude changes at equal trainable capacity. Exact-initialization random controls and frozen heads are matched. No winner selected or official test evaluated.')
    write_report(directory/'summary.json',result)
    return result


