"""Persist and reproduce every tuning candidate, including losing linear states."""
from dataclasses import asdict, replace
import json
from pathlib import Path

import torch

from .compare import validate_probe_seeds, write_report
from .probe_artifacts import load_probe, probe_context, save_probe
from .probes import fit_probe, validate_candidates
from .ridge import fit_ridge, validate_ridge_candidates


def source_record(path, pair, candidates, seeds, ridge_alphas, *, reuse_only=False):
    path = Path(path)
    context = probe_context(*pair)
    validate_candidates(candidates)
    validate_probe_seeds(seeds)
    ridge = validate_ridge_candidates(ridge_alphas)
    specs = [('adamw', seed, tuple(replace(c, seed=seed) for c in candidates), 'cross_entropy') for seed in seeds]
    specs.append(('ridge', None, ridge, 'mean_squared_error'))
    identity = dict(version=1, context=context, runtime=dict(torch=str(torch.__version__), cpu_threads=torch.get_num_threads()),
                    test_evaluated=False)
    cached = json.loads(path.read_text()) if path.exists() else None
    if cached is None and reuse_only:
        raise ValueError('Objective tuning report is missing')
    if cached is not None and (any(cached.get(k) != v for k, v in identity.items())
                               or len(cached.get('grids', [])) != len(specs)):
        raise ValueError('Objective tuning identity or grids changed')
    # Check every existing state before creating missing states, including partial retries.
    restored = {}
    for family, seed, configs, metric in specs:
        for index, config in enumerate(configs):
            output = path.parent / path.stem / f'{family}-seed{seed}-candidate{index}.pt'
            if output.exists():
                probe, artifact = load_probe(output, *pair)
                if artifact['config'] != asdict(config):
                    raise ValueError('Objective candidate configuration differs')
                restored[output] = (probe, artifact)
            elif cached is not None:
                raise ValueError('Recorded objective candidate state is missing')
    grids = []
    for number, (family, seed, configs, metric) in enumerate(specs):
        records = []
        for index, config in enumerate(configs):
            output = path.parent / path.stem / f'{family}-seed{seed}-candidate{index}.pt'
            if output in restored:
                probe, artifact = restored[output]
            else:
                fit = fit_probe if family == 'adamw' else fit_ridge
                probe = fit(pair[0].features, pair[0].labels, config)
                artifact = save_probe(output, probe, *pair)
            records.append(dict(path=str(output), **artifact))
        chosen = min(range(len(records)), key=lambda i: (-records[i]['scores']['validation']['accuracy'],
                       records[i]['scores']['validation'][metric], i))
        grid = dict(family=family, classifier_seed=seed, candidates=records, selected_candidate=chosen)
        if cached is not None and cached['grids'][number] != grid:
            raise ValueError('Objective candidate scores, selection rule or state changed')
        grids.append(grid)
    result = identity | dict(grids=grids)
    if cached is None:
        write_report(path, result)
    selected = [dict(head=g['family'], classifier_seed=g['classifier_seed'], **g['candidates'][g['selected_candidate']]) for g in grids]
    return dict(selected=selected, verified_candidate_states=sum(len(g['candidates']) for g in grids),
                verified_selected_states=len(selected), report=result)


def prepare_selection(run, *, reuse_only=False):
    """Tune only on the original pilot; lock all states before fresh access."""
    from .neighbor_workflow import neighbor_record
    from .neighbors import NeighborConfig
    from .objective_compare import validate_objective_features
    from .objective_features import feature, seed_sources, source_specs
    from .objective_run import file_sha
    from .probes import ProbeConfig
    d, manifest, ref, fresh, labels, bundle = run.inputs()
    completed = run.training(reuse_only=True)
    settings = d['evaluation']
    candidates = tuple(ProbeConfig(settings['probe_steps'], lr, decay, settings['classifier_seeds'][0])
        for lr in settings['learning_rates'] for decay in settings['weight_decays'])
    validate_candidates(candidates)
    validate_probe_seeds(settings['classifier_seeds'])
    validate_ridge_candidates(settings['ridge_alphas'])
    neighbors = [NeighborConfig(k) for k in settings['neighbor_counts']]
    pairs = {key: tuple(feature(run, d, manifest, ref, labels, key, partition, reuse_only=reuse_only)
        for partition in ('train', 'validation')) for key in source_specs(d)}
    for seed in d['training_seeds']:
        validate_objective_features(seed_sources(pairs, seed), candidates)
    selected, selected_neighbors, reports, count = [], {}, {}, 0
    for key, pair in pairs.items():
        path = run.directory / 'tuning' / f'{key}.json'
        result = source_record(path, pair, candidates, settings['classifier_seeds'],
                               settings['ridge_alphas'], reuse_only=reuse_only)
        selected.extend(dict(source_key=key, **row) for row in result['selected'])
        count += result['verified_candidate_states']
        reports[key] = file_sha(path)
        selected_neighbors[key] = neighbor_record(run.directory/'neighbors'/f'{key}.json', pair, neighbors, reuse_only)
    lock = dict(version=1, protocol_sha256=run.expected_sha,
        training_lock_sha256=file_sha(run.directory/'training-lock.json'),
        tuning_report_sha256=reports, linear=selected, neighbors=selected_neighbors)
    path = run.directory/'selection-lock.json'
    if path.exists():
        if json.loads(path.read_text()) != lock:
            raise ValueError('Selected objective/classifier lock differs')
    elif reuse_only:
        raise ValueError('Objective selection lock missing')
    else:
        write_report(path, lock)
    return dict(declaration=d, manifest=manifest, reference=ref, fresh=fresh, labels=labels,
        sources=pairs, lock=lock, lock_path=path, completed=completed, verified_candidate_states=count)
