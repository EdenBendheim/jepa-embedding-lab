"""Run or replay the declared longer joint-versus-separate CIFAR-10 comparison."""

import argparse
import json
from pathlib import Path
import platform
from statistics import mean

import torch

from experiments.multiblock_budget import budget_schedule
from experiments.multiblock_frozen import (
    _verify_or_save, paired_scores, prepare_heads, register_validation, score_frozen,
)
from experiments.multiblock_milestones import verify_milestones
from experiments.multiblock_trajectory import run_trajectory
from jepa_lab.embeddings import random_encoder
from jepa_lab.exposure import exposure_summary, load_exposure_bundle
from jepa_lab.model import ModelConfig
from jepa_lab.objective_run import file_sha
from jepa_lab.selection import validate_pilot
from jepa_lab.variants.cifar_batches import CifarTrainingData
from jepa_lab.variants.multiblock_train import _digest, _sources

EXPERIMENT_DIR = Path(__file__).resolve().parent
PROTOCOL_PATH = EXPERIMENT_DIR/'multiblock-budget-protocol-20261118.json'


def runtime():
    return dict(torch=str(torch.__version__), python=platform.python_version(),
                cpu_threads=torch.get_num_threads(), architecture=platform.machine())


def load_protocol(args):
    protocol = json.loads(PROTOCOL_PATH.read_text())
    if (protocol['version'] != 1 or protocol['package_sha256'] != _digest(_sources())
            or protocol['runtime'] != runtime()):
        raise ValueError('Installed package or runtime differs from the declared experiment')
    for filename, expected in protocol['experiment_sources'].items():
        if file_sha(EXPERIMENT_DIR/filename) != expected:
            raise ValueError(f'Declared experiment source changed: {filename}')
    timing = EXPERIMENT_DIR/'multiblock-timing-results-20261114.json'
    for file, name in ((timing, 'timing'), (args.manifest, 'manifest'),
                       (args.selection, 'selection'), (args.exposure, 'exposure')):
        if file_sha(file) != protocol['input_sha256'][name]:
            raise ValueError(f'Declared experiment input changed: {name}')
    timing_report = json.loads(timing.read_text())
    schedule = budget_schedule(timing_report, seconds_per_arm=protocol['projected_seconds_per_arm'],
                               chunk_steps=protocol['schedule']['chunk_steps'])
    if schedule != protocol['schedule'] or timing_report['batch_size'] != protocol['batch_size']:
        raise ValueError('Training schedule differs from the declared timing projection')
    return protocol


def execute(args, protocol, data, selection):
    protocol_sha = file_sha(PROTOCOL_PATH)
    directory = args.directory
    output = directory/'summary.json'
    if args.replay and not output.is_file():
        raise ValueError('Replay requires a completed report')
    config = ModelConfig(**protocol['config'])
    models, milestones = {}, {}
    plan = protocol['schedule']['checkpoints']
    for seed, order in zip(protocol['seeds'], protocol['orders'], strict=True):
        folder = directory/f'seed-{seed}'
        receipts = {arm: run_trajectory(data, folder/arm, grouping=arm, ends=plan[arm],
            seed=seed, batch_size=protocol['batch_size'], config=config, replay=args.replay)
            for arm in order}
        frozen, costs = verify_milestones(data, folder, receipts)
        initial = random_encoder(config, seed)
        frozen['random'] = initial
        models.update({f'{seed}/{name}': model for name, model in frozen.items()})
        milestones[str(seed)] = costs | dict(arm_order=order)
    lock = dict(protocol_sha256=protocol_sha, milestones=milestones)
    _verify_or_save(directory/'training-lock.json', lock)
    print(json.dumps(dict(training_complete=True, seeds=protocol['seeds'])), flush=True)

    heads = prepare_heads(data, directory, selection, models, identity=_digest(lock),
                          alpha=protocol['ridge_alpha'], replay=args.replay)
    fresh, exposure = register_validation(data, directory, selection, args.exposure,
        per_class=protocol['validation_per_class'], seed=protocol['validation_seed'])
    scores, predictions, labels = score_frozen(data, directory, fresh, models, heads)
    extended = 'joint_extended' in milestones[str(protocol['seeds'][0])]['milestones']
    paired = {str(seed): paired_scores(labels, predictions, seed=seed, extended=extended)
              for seed in protocol['seeds']}
    names = list(frozen)
    average = {name: mean(scores[f'{seed}/{name}']['accuracy'] for seed in protocol['seeds'])
               for name in names}
    average.update({name: scores[name]['accuracy'] for name in ('rgb_statistics', 'pixels')})
    report = dict(version=1, kind='matched_multiblock_budget_validation', protocol_sha256=protocol_sha,
        runtime=protocol['runtime'], manifest_sha256=data.manifest['manifest_sha256'],
        training=milestones, planning=protocol['schedule']['planning'],
        fitting_rows=len(selection['partitions']['train']['indices']), scoring_rows=len(labels),
        scoring_selection_sha256=fresh['selection_sha256'], ridge_alpha=protocol['ridge_alpha'],
        heads_lock_sha256=file_sha(directory/'heads-lock.json'), exposure=exposure,
        scores=scores, paired=paired, mean_accuracy=average, test_images_indexed=False,
        limitation='Fixed budget projection and descriptive results on one reserved validation subset. '
                   'Elapsed times differ, seeds share validation rows, and the official test remains reserved.')
    _verify_or_save(output, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('data/cifar-10-batches-bin'))
    parser.add_argument('--manifest', type=Path, default=Path('runs/cifar10-manifest.json'))
    parser.add_argument('--selection', type=Path, default=Path('runs/pilot-selection-20261009.json'))
    parser.add_argument('--exposure', type=Path,
        default=Path('runs/context-variance-20261031/observed-exposure.json'))
    parser.add_argument('--directory', type=Path, default=Path('runs/multiblock-budget-20261118'))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--preflight', action='store_true')
    mode.add_argument('--replay', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    try:
        if args.directory.resolve().is_relative_to(args.root.resolve()):
            raise ValueError('Choose an experiment directory outside the dataset')
        protocol = load_protocol(args)
        data = CifarTrainingData(args.root, args.manifest)
        selection = validate_pilot(json.loads(args.selection.read_text()), data.manifest, data.labels)
        if args.preflight:
            previous = load_exposure_bundle(args.exposure, data.manifest, data.labels)
            exposure = exposure_summary(previous['ledger'], data.manifest, data.labels)
            if min(exposure['remaining_per_class'].values()) < protocol['validation_per_class']:
                raise ValueError('Insufficient unused rows for the declared validation sample')
            print(json.dumps(dict(preflight=True, protocol_sha256=file_sha(PROTOCOL_PATH),
                schedule=protocol['schedule'], prior_observed_rows=exposure['observed_validation_count'])))
            return
        result = execute(args, protocol, data, selection)
        print(json.dumps(dict(summary=str(args.directory/'summary.json'), mean_accuracy=result['mean_accuracy'])))
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
