"""Execute the predeclared training-only CIFAR-10 multi-block pilot."""

import argparse
import json
from pathlib import Path

import torch

from jepa_lab.compare import write_report
from jepa_lab.model import ModelConfig
from jepa_lab.objective_run import file_sha
from jepa_lab.selection import validate_pilot
from jepa_lab.variants.cifar_batches import CifarTrainingData
from jepa_lab.variants.multiblock_evaluation import evaluate_pair
from jepa_lab.variants.multiblock_pair import pair_spec, train_pair
from jepa_lab.variants.multiblock_train import _digest, _sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('data/cifar-10-batches-bin'))
    parser.add_argument('--manifest', type=Path, default=Path('runs/cifar10-manifest.json'))
    parser.add_argument('--selection', type=Path, default=Path('runs/pilot-selection-20261009.json'))
    parser.add_argument('--exposure', type=Path, default=Path('runs/context-variance-20261031/observed-exposure.json'))
    parser.add_argument('--directory', type=Path, default=Path('runs/multiblock-cifar-20261111'))
    parser.add_argument('--preflight', action='store_true', help='Verify inputs before indexing images or writing outputs')
    args = parser.parse_args()
    torch.set_num_threads(1)
    try:
        protocol_path = Path(__file__).with_name('multiblock-cifar-protocol-20261111.json')
        protocol = json.loads(protocol_path.read_text())
        if protocol['source_sha256'] != _digest(_sources()):
            raise ValueError('Current package sources differ from the declared pilot')
        if args.directory.resolve().is_relative_to(args.root.resolve()):
            raise ValueError('Use a working directory outside the CIFAR-10 inputs')
        for path, field in ((args.manifest, 'manifest_file_sha256'),
                            (args.selection, 'pilot_selection_file_sha256'),
                            (args.exposure, 'existing_exposure_file_sha256')):
            if file_sha(path) != protocol[field]:
                raise ValueError(f'Input bytes differ from the declared pilot: {path.name}')
        data = CifarTrainingData(args.root, args.manifest)
        selection = validate_pilot(json.loads(args.selection.read_text()), data.manifest, data.labels)
        spec = pair_spec(data, config=ModelConfig(**protocol['config']),
                         **{key: protocol[key] for key in ('steps', 'batch_size', 'seed')})
        if (len(selection['partitions']['train']['indices']) != protocol['scoring_rows']
                or any(identity['runtime'] != protocol['runtime'] for identity in spec['arms'].values())
                or (protocol['partition'], protocol['folds'], protocol['fold_seed'], protocol['ridge_alpha'])
                != ('train', 5, 29, .1)):
            raise ValueError('Current fitting selection, runtime or evaluation recipe differs from the declaration')
        if args.preflight:
            print(json.dumps(dict(preflight=True, protocol_sha256=file_sha(protocol_path))))
            return
        training = train_pair(data, args.directory, spec)
        report = evaluate_pair(data, args.directory, spec, selection)
        if report['training'] != training:
            raise ValueError('Verified checkpoints changed between training and scoring')
        report['protocol_sha256'] = file_sha(protocol_path)
        report['existing_exposure_file_sha256'] = file_sha(args.exposure)
        if report['existing_exposure_file_sha256'] != protocol['existing_exposure_file_sha256']:
            raise ValueError('Existing exposure history changed during the diagnostic')
        output = args.directory/'aggregate.json'
        if output.exists():
            if json.loads(output.read_text()) != report:
                raise ValueError('Recomputed scores differ from the existing aggregate report')
        else:
            write_report(output, report)
        print(json.dumps(dict(output=str(output), accuracy={
            name: result['score']['accuracy'] for name, result in report['scores'].items()})))
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
