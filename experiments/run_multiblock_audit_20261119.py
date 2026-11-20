"""Audit fixed CIFAR-10 grouping checkpoints on the original fitting panel."""

import argparse
import json
from pathlib import Path

import torch

from experiments.multiblock_audit_color import measure_rgb_reconstruction, residual_classification
from experiments.multiblock_audit_geometry import measure_geometry
from experiments.multiblock_audit_layout import layout_transfer
from experiments.multiblock_audit_panel import feature_panel
from experiments.multiblock_audit_sources import reopen_sources
from experiments.multiblock_frozen import _verify_or_save
from experiments.run_multiblock_budget_20261118 import load_protocol as load_budget_protocol
from jepa_lab.analysis.spatial_evaluation import stratified_folds
from jepa_lab.objective_run import file_sha
from jepa_lab.variants.cifar_batches import CifarTrainingData
from jepa_lab.variants.multiblock_train import _digest

EXPERIMENTS = Path(__file__).resolve().parent
PROTOCOL_PATH = EXPERIMENTS/'multiblock-audit-protocol-20261119.json'
BUDGET_RESULTS = EXPERIMENTS/'multiblock-budget-results-20261118.json'


def load_audit_protocol():
    protocol = json.loads(PROTOCOL_PATH.read_text())
    if (type(protocol['version']) is not int or protocol['version'] != 1
            or protocol['budget_results_sha256'] != file_sha(BUDGET_RESULTS)):
        raise ValueError('The audit protocol does not match the published budget experiment')
    for filename, expected in protocol['experiment_sources'].items():
        if file_sha(EXPERIMENTS/filename) != expected:
            raise ValueError(f'Declared audit code changed: {filename}')
    return protocol


def execute(args, protocol, budget, sources, data):
    directory = args.directory
    target = directory/'aggregate.json'
    if args.replay and not target.is_file():
        raise ValueError('Replay requires an existing aggregate from the declared audit')
    cache = directory/'fitting-panel.pt'
    protocol_sha = file_sha(PROTOCOL_PATH)
    settings = protocol['audit']
    features, labels = feature_panel(data, sources, cache, protocol_sha=protocol_sha,
        batch_size=settings['batch_size'], shuffle_seed=settings['shuffle_seed'], replay=args.replay)
    folds = stratified_folds(labels, folds=settings['folds'], seed=settings['fold_seed'])
    reconstruction = measure_rgb_reconstruction(features, folds, alpha=settings['reconstruction_alpha'])
    residual = residual_classification(features, labels, folds,
        reconstruction_alpha=settings['reconstruction_alpha'], classifier_alpha=settings['classifier_alpha'])
    layout = layout_transfer(features, labels, folds, alpha=settings['classifier_alpha'])
    geometry = measure_geometry(features, budget['seeds'])
    report = dict(version=1, kind='multiblock_training_panel_representation_audit',
        protocol_sha256=protocol_sha, sources=sources.identity, audit=settings,
        fold_assignments_sha256=_digest(folds.tolist()), panel_file_sha256=file_sha(cache),
        rows=len(labels), partition='train', geometry=geometry, rgb_reconstruction=reconstruction,
        residual_classification=residual, layout_transfer=layout,
        validation_images_indexed=False, test_images_indexed=False,
        limitation='Exploratory fixed-fold analysis on encoder-training partition rows. '
                   'RGB residuals remove a fitted linear component, and shuffling shifts the input distribution. '
                   'Classification here does not establish generalization or explain the earlier held-out difference.')
    _verify_or_save(target, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('data/cifar-10-batches-bin'))
    parser.add_argument('--manifest', type=Path, default=Path('runs/cifar10-manifest.json'))
    parser.add_argument('--selection', type=Path, default=Path('runs/pilot-selection-20261009.json'))
    parser.add_argument('--exposure', type=Path,
        default=Path('runs/context-variance-20261031/observed-exposure.json'),
        help='Historical exposure file pinned by the budget experiment')
    parser.add_argument('--budget-directory', type=Path, default=Path('runs/multiblock-budget-20261118'))
    parser.add_argument('--directory', type=Path, default=Path('runs/multiblock-audit-20261119'))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--preflight', action='store_true')
    mode.add_argument('--replay', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    try:
        protected_dirs = (args.root, args.budget_directory, EXPERIMENTS)
        if any(args.directory.resolve().is_relative_to(path.resolve()) for path in protected_dirs):
            raise ValueError('Place audit artifacts outside the data, checkpoint and code directories')
        for name in ('fitting-panel.pt', 'aggregate.json'):
            path = (args.directory/name).resolve()
            protected_files = (args.manifest, args.selection, args.exposure)
            if path.parent != args.directory.resolve() or path in {p.resolve() for p in protected_files}:
                raise ValueError('Audit output cannot redirect outside its own directory or replace an input')
        protocol = load_audit_protocol()
        budget = load_budget_protocol(args)
        prior = json.loads(BUDGET_RESULTS.read_text())
        data = CifarTrainingData(args.root, args.manifest)
        selection = json.loads(args.selection.read_text())
        sources = reopen_sources(data, selection, args.budget_directory, budget, prior['training'])
        if len(sources.indices) != protocol['fitting_rows']:
            raise ValueError('The fixed fitting sample differs from the declared row count')
        if args.preflight:
            print(json.dumps(dict(preflight=True, encoders=len(sources.models),
                fitting_rows=len(sources.indices), protocol_sha256=file_sha(PROTOCOL_PATH))))
            return
        report = execute(args, protocol, budget, sources, data)
        print(json.dumps(dict(output=str(args.directory/'aggregate.json'), rows=report['rows'],
                              encoders=len(sources.models), replay=args.replay)))
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
