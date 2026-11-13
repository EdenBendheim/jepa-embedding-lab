"""Run the predeclared predictor grouping comparison on CIFAR-10 training rows."""

import argparse
import json
from pathlib import Path
from statistics import mean

import torch

from jepa_lab.compare import write_report
from jepa_lab.model import ModelConfig
from jepa_lab.objective_run import file_sha
from jepa_lab.selection import validate_pilot
from jepa_lab.variants.cifar_batches import CifarTrainingData
from jepa_lab.variants.multiblock_evaluation import evaluate_pair
from jepa_lab.variants.multiblock_pair import pair_spec, train_pair
from jepa_lab.variants.multiblock_train import _digest, _sources


def declared_pairs(protocol, data, selection):
    if protocol['version'] != 1 or protocol['contrast'] != 'grouping':
        raise ValueError('Expected the predictor grouping comparison')
    seeds = protocol['seeds']
    if len(seeds) != 3 or len(set(seeds)) != 3:
        raise ValueError('Declare three distinct encoder seeds')
    if len(selection['partitions']['train']['indices']) != protocol['scoring_rows']:
        raise ValueError('Fitting row count differs from the declaration')
    if (protocol['folds'], protocol['fold_seed'], protocol['ridge_alpha']) != (5, 29, .1):
        raise ValueError('Classifier recipe differs from the declaration')

    specs = [pair_spec(data, config=ModelConfig(**protocol['config']), contrast=protocol['contrast'],
                       seed=seed, steps=protocol['steps'], batch_size=protocol['batch_size']) for seed in seeds]
    if any(arm['runtime'] != protocol['runtime'] for spec in specs for arm in spec['arms'].values()):
        raise ValueError('Current runtime differs from the declaration')
    return tuple(zip(seeds, specs, strict=True))


def run(protocol, data, selection, directory, pairs):
    rows = []
    for seed, spec in pairs:
        folder = directory/f'seed-{seed}'
        receipt = train_pair(data, folder, spec)
        report = evaluate_pair(data, folder, spec, selection)
        if receipt != report['training']:
            raise ValueError('Checkpoints changed before scoring')
        saved = folder/'aggregate.json'
        if saved.exists():
            if json.loads(saved.read_text()) != report:
                raise ValueError('Stored aggregate differs from replayed scores')
        else:
            write_report(saved, report)
        pair = report['paired']['joint_2x2-minus-separate_2x2']
        rows.append(dict(seed=seed, training=receipt, arms=report['arms'],
            scoring_overlap=report['scoring_rows_seen_in_encoder_training'],
            folds_sha256=report['fold_assignments_sha256'],
            accuracy={name: source['score']['accuracy'] for name, source in report['scores'].items()},
            joint_minus_separate={key: pair[key] for key in ('accuracy_delta', 'recovered', 'regressed')}))
    if len({row['folds_sha256'] for row in rows}) != 1:
        raise ValueError('Seed pairs use different classifier folds')
    return dict(version=1, contrast=protocol['contrast'], source_sha256=protocol['source_sha256'],
                manifest_sha256=data.manifest['manifest_sha256'], selection_sha256=selection['selection_sha256'],
                scoring_rows=protocol['scoring_rows'], partition='train', seeds=rows,
                mean_accuracy={key: mean(row['accuracy'][key] for row in rows) for key in rows[0]['accuracy']},
                mean_joint_minus_separate=mean(row['joint_minus_separate']['accuracy_delta'] for row in rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('data/cifar-10-batches-bin'))
    parser.add_argument('--manifest', type=Path, default=Path('runs/cifar10-manifest.json'))
    parser.add_argument('--selection', type=Path, default=Path('runs/pilot-selection-20261009.json'))
    parser.add_argument('--exposure', type=Path, default=Path('runs/context-variance-20261031/observed-exposure.json'))
    parser.add_argument('--directory', type=Path, default=Path('runs/multiblock-grouping-20261112'))
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    try:
        recipe = Path(__file__).with_name('multiblock-grouping-protocol-20261112.json')
        protocol = json.loads(recipe.read_text())
        if _digest(_sources()) != protocol['source_sha256']:
            raise ValueError('Package sources differ from the declaration')
        if args.directory.resolve().is_relative_to(args.root.resolve()):
            raise ValueError('Use a working directory outside the CIFAR-10 inputs')
        for path, field in ((args.manifest, 'manifest_file_sha256'),
                            (args.selection, 'pilot_selection_file_sha256'),
                            (args.exposure, 'existing_exposure_file_sha256')):
            if file_sha(path) != protocol[field]:
                raise ValueError(f'Input bytes differ from the declaration: {path.name}')
        data = CifarTrainingData(args.root, args.manifest)
        selection = validate_pilot(json.loads(args.selection.read_text()), data.manifest, data.labels)
        pairs = declared_pairs(protocol, data, selection)
        if args.preflight:
            print(json.dumps(dict(preflight=True, protocol_sha256=file_sha(recipe))))
            return
        result = run(protocol, data, selection, args.directory, pairs)
        if file_sha(args.exposure) != protocol['existing_exposure_file_sha256']:
            raise ValueError('Exposure receipt changed during the run')
        result.update(protocol_sha256=file_sha(recipe),
                      existing_exposure_file_sha256=protocol['existing_exposure_file_sha256'])
        output = args.directory/'summary.json'
        if output.exists():
            if json.loads(output.read_text()) != result:
                raise ValueError('Stored summary differs from replayed scores')
        else:
            write_report(output, result)
        print(json.dumps(dict(output=str(output), mean_accuracy=result['mean_accuracy'])))
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
