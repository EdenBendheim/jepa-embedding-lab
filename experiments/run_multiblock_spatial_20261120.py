"""Compare spatial readouts on the fixed multi-block CIFAR-10 fitting panel."""

import argparse
import json
from pathlib import Path

import torch

from experiments.multiblock_audit_sources import reopen_sources
from experiments.multiblock_frozen import _verify_or_save
from experiments.multiblock_spatial_comparison import compare_readouts
from experiments.multiblock_spatial_panel import spatial_panel
from experiments.multiblock_spatial_scoring import score_spatial
from experiments.run_multiblock_audit_20261119 import BUDGET_RESULTS, load_audit_protocol
from experiments.run_multiblock_budget_20261118 import load_protocol as load_budget_protocol
from jepa_lab.analysis.spatial_evaluation import stratified_folds
from jepa_lab.objective_run import file_sha
from jepa_lab.variants.cifar_batches import CifarTrainingData
from jepa_lab.variants.multiblock_train import _digest

EXPERIMENTS = Path(__file__).resolve().parent
PROTOCOL_PATH = EXPERIMENTS/'multiblock-spatial-protocol-20261120.json'
AUDIT_RESULTS = EXPERIMENTS/'multiblock-audit-results-20261119.json'


def load_spatial_protocol():
    protocol = json.loads(PROTOCOL_PATH.read_text())
    if (type(protocol['version']) is not int or protocol['version'] != 1
            or protocol['prior_audit_sha256'] != file_sha(AUDIT_RESULTS)):
        raise ValueError('The spatial protocol does not match the published fitting-panel audit')
    for filename, expected in protocol['experiment_sources'].items():
        if file_sha(EXPERIMENTS/filename) != expected:
            raise ValueError(f'Declared spatial audit code changed: {filename}')
    return protocol


def compare_global_anchor(prior, scores, fold_hash):
    if prior['fold_assignments_sha256'] != fold_hash:
        raise ValueError('Fitting folds differ from the earlier mean-readout audit')
    for source, row in prior['layout_transfer']['comparisons'].items():
        current = scores[f'{source}/mean']
        observed = current['original']['correct'], current['transfer']['patch_shuffle']['candidate_correct']
        expected = row['baseline_correct'], row['candidate_correct']
        if observed != expected:
            raise ValueError(f'Global readout did not reproduce its prior scores for {source}')


def execute(args, protocol, budget, sources, data, prior):
    target = args.directory/'aggregate.json'
    cache = args.directory/'spatial-panel.pt'
    if args.replay and not target.is_file():
        raise ValueError('Replay requires a completed spatial aggregate')
    settings = protocol['settings']
    protocol_sha = file_sha(PROTOCOL_PATH)
    features, labels = spatial_panel(data, sources, cache, protocol_sha=protocol_sha,
        batch_size=settings['batch_size'], patch_seed=settings['patch_seed'], replay=args.replay)
    folds = stratified_folds(labels, folds=settings['folds'], seed=settings['fold_seed'])
    fold_hash = _digest(folds.tolist())
    scores, predictions = score_spatial(features, labels, sources.indices, folds,
        alpha=settings['alpha'], quadrant_seed=settings['quadrant_seed'])
    compare_global_anchor(prior, scores, fold_hash)
    comparison = compare_readouts(labels, predictions, budget['seeds'])
    report = dict(version=1, kind='multiblock_spatial_readout_fitting_diagnostic',
        protocol_sha256=protocol_sha, sources=sources.identity, settings=settings,
        fold_assignments_sha256=fold_hash, feature_cache_sha256=file_sha(cache),
        rows=len(labels), partition='train', global_anchor_reproduced=True,
        scores=scores, **comparison, validation_images_indexed=False, test_images_indexed=False,
        limitation='Descriptive cross-fitting on a training-partition panel with encoder exposure. '
                   'Layout perturbations measure fixed-head sensitivity on these samples.')
    _verify_or_save(target, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('data/cifar-10-batches-bin'))
    parser.add_argument('--manifest', type=Path, default=Path('runs/cifar10-manifest.json'))
    parser.add_argument('--selection', type=Path, default=Path('runs/pilot-selection-20261009.json'))
    parser.add_argument('--exposure', type=Path,
        default=Path('runs/context-variance-20261031/observed-exposure.json'))
    parser.add_argument('--budget-directory', type=Path, default=Path('runs/multiblock-budget-20261118'))
    parser.add_argument('--directory', type=Path, default=Path('runs/multiblock-spatial-20261120'))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--preflight', action='store_true')
    mode.add_argument('--replay', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    try:
        destination = args.directory.resolve()
        protected_dirs = (args.root, args.budget_directory, EXPERIMENTS,
                          Path('runs/multiblock-audit-20261119'))
        if any(destination.is_relative_to(path.resolve()) for path in protected_dirs):
            raise ValueError('Place this audit outside the raw data and earlier experiment directories')
        protected = {p.resolve() for p in (args.manifest, args.selection, args.exposure)}
        for name in ('spatial-panel.pt', 'aggregate.json'):
            path = (destination/name).resolve()
            if path.parent != destination or path in protected:
                raise ValueError('Audit output must stay in its own directory and preserve the inputs')
        protocol = load_spatial_protocol()
        load_audit_protocol()
        budget = load_budget_protocol(args)
        budget_results = json.loads(BUDGET_RESULTS.read_text())
        prior = json.loads(AUDIT_RESULTS.read_text())
        data = CifarTrainingData(args.root, args.manifest)
        selection = json.loads(args.selection.read_text())
        sources = reopen_sources(data, selection, args.budget_directory, budget, budget_results['training'])
        if sources.identity != prior['sources'] or len(sources.indices) != protocol['fitting_rows']:
            raise ValueError('The panel order or frozen checkpoint identity differs from the prior audit')
        if args.preflight:
            print(json.dumps(dict(preflight=True, sources=len(sources.models), rows=len(sources.indices),
                                  protocol_sha256=file_sha(PROTOCOL_PATH))))
            return
        result = execute(args, protocol, budget, sources, data, prior)
        print(json.dumps(dict(output=str(args.directory/'aggregate.json'),
                              rows=result['rows'], replay=args.replay)))
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
