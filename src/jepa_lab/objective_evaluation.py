"""Frozen objective evaluation with shared controls and no-refit cache replay."""
import json
from statistics import mean, stdev

from .compare import write_report
from .confirmation import score_confirmation
from .confirmation_compare import compare_confirmed_probes
from .diagnostics import paired_diagnostics
from .neighbor_artifacts import load_neighbors
from .neighbor_confirmation import score_neighbor_confirmation
from .objective_features import feature, source_specs
from .objective_observation import observation_gate
from .objective_run import file_sha
from .objective_selection import prepare_selection


def summarize(rows, d, specs):
    result = {}
    for group in ('prediction', 'context_variance', 'random', 'pixels'):
        keys = [k for k, spec in specs.items() if (spec['arm']['name'] if spec['arm'] else spec['representation']) == group]
        result[group] = {}
        for head in ('adamw', 'ridge', 'knn'):
            selected = [r for r in rows if r['source_key'] in keys and r['head'] == head]
            groups = [selected] if group == 'pixels' else [[r for r in selected if specs[r['source_key']]['seed'] == seed]
                                                         for seed in d['training_seeds']]
            result[group][head] = {}
            for phase in ('original_tuning', 'confirmation'):
                scores = [mean(r[phase]['accuracy'] for r in values) for values in groups]
                result[group][head][phase] = dict(encoder_means=scores, mean_accuracy=mean(scores),
                    encoder_mean_sample_sd=stdev(scores) if len(scores) > 1 else 0.)
    return result


def evaluation(run, *, reuse_only=False):
    state = prepare_selection(run, reuse_only=reuse_only)
    d, manifest, fresh, labels = [state[k] for k in ('declaration', 'manifest', 'fresh', 'labels')]
    sources, lock = state['sources'], state['lock']
    exposure = observation_gate(run, d, manifest, fresh, labels, register=not reuse_only)
    specs = source_specs(d)
    fresh_sources = {key: feature(run, d, manifest, fresh, labels, key, 'validation', reuse_only=reuse_only)
                     for key in specs}
    rows = []
    for row in lock['linear']:
        key = row['source_key']
        scored = score_confirmation(row['path'], *sources[key], fresh_sources[key], fresh)
        rows.append(dict(source_key=key, head=row['head'], classifier_seed=row['classifier_seed'],
            selected_config=row['config'], state_sha256=row['state_sha256'], artifact_sha256=row['artifact_sha256'],
            original_tuning=row['scores']['validation'], confirmation=scored['confirmation']))
    indices = {}
    for key, row in lock['neighbors'].items():
        scored = score_neighbor_confirmation(row['path'], *sources[key], fresh_sources[key], fresh)
        indices[key] = load_neighbors(row['path'], *sources[key])[0]
        rows.append(dict(source_key=key, head='knn', classifier_seed=None,
            selected_config=row['index']['config'], state_sha256=row['index']['state_sha256'],
            artifact_sha256=row['index']['artifact_sha256'], original_tuning=row['index']['scores']['validation'],
            confirmation=scored['confirmation']))
    paired = []
    for seed in d['training_seeds']:
        before, after = f'prediction-seed{seed}', f'context_variance-seed{seed}'
        for head, classifier_seed in [('adamw', s) for s in d['evaluation']['classifier_seeds']] + [('ridge', None), ('knn', None)]:
            if head == 'knn':
                diagnostics = paired_diagnostics(fresh_sources[before].labels,
                    indices[before].predict(fresh_sources[before].features), indices[after].predict(fresh_sources[after].features))
            else:
                selected = [next(r for r in lock['linear'] if r['source_key'] == key and r['head'] == head
                                 and r['classifier_seed'] == classifier_seed) for key in (before, after)]
                comparison = compare_confirmed_probes((*sources[before], fresh_sources[before]),
                    (*sources[after], fresh_sources[after]), selected[0]['path'], selected[1]['path'], fresh)
                diagnostics = comparison['paired_confirmation']
            paired.append(dict(training_seed=seed, head=head, classifier_seed=classifier_seed,
                               baseline='prediction', candidate='context_variance', paired_confirmation=diagnostics))
    summary = summarize(rows, d, specs)
    ref = state['reference']
    result = dict(version=1, protocol_sha256=run.expected_sha, manifest_sha256=manifest['manifest_sha256'],
        reference_selection_sha256=ref['selection_sha256'], confirmation_selection_sha256=fresh['selection_sha256'],
        selection_lock_sha256=file_sha(state['lock_path']),
        counts=dict(train=len(ref['partitions']['train']['indices']), tuning=len(ref['partitions']['validation']['indices']),
                    confirmation=len(fresh['partitions']['validation']['indices']),
                    excluded_observed_validation=len(fresh['exposure_ledger']['observed_validation_indices'])),
        primary_delta=summary['context_variance']['adamw']['confirmation']['mean_accuracy']
                    - summary['prediction']['adamw']['confirmation']['mean_accuracy'],
        summary=summary, paired_objectives=paired, classifiers=rows, exposure=exposure,
        training={key: {k: v for k, v in row.items() if k not in ('history', 'limitation')}
                  for key, row in state['completed'].items()},
        verified_candidate_states=state['verified_candidate_states'], verified_linear_states=len(lock['linear']),
        verified_neighbor_states=len(lock['neighbors']), actual_image_order_matched=True,
        all_states_locked_before_confirmation=True, confirmation_refitting=False, test_evaluated=False,
        limitation='Shared rows, known tuning data and seed variation are descriptive, not independent test uncertainty. '
                   'Only the declared objective changes; random and pixel controls are shared. '
                   'A variance penalty does not imply useful features; report negative results without selecting a winner.')
    path = run.directory/'summary.json'
    if path.exists():
        if json.loads(path.read_text()) != result:
            raise ValueError('Existing objective summary differs; preserve the prior file')
    elif reuse_only:
        raise ValueError('Objective summary missing in reuse-only mode')
    else:
        write_report(path, result)
    return result
