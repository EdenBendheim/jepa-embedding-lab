"""Time the declared predictor grouping comparison on local CIFAR-10 data."""

import argparse
import json
from pathlib import Path
import platform

import torch

from jepa_lab.objective_run import file_sha
from jepa_lab.variants.cifar_batches import CifarTrainingData
from jepa_lab.variants.multiblock_timing import benchmark_grouping, estimate_step_budget
from jepa_lab.variants.multiblock_train import _digest, _sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('data/cifar-10-batches-bin'))
    parser.add_argument('--manifest', type=Path, default=Path('runs/cifar10-manifest.json'))
    parser.add_argument('--exposure', type=Path, default=Path('runs/context-variance-20261031/observed-exposure.json'))
    parser.add_argument('--output', type=Path)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    try:
        recipe_path = Path(__file__).with_name('multiblock-timing-protocol-20261114.json')
        recipe = json.loads(recipe_path.read_text())
        runtime = dict(torch=str(torch.__version__), python=platform.python_version(), cpu_threads=torch.get_num_threads())
        if recipe['version'] != 1 or recipe['source_sha256'] != _digest(_sources()) or recipe['runtime'] != runtime:
            raise ValueError('Package sources or runtime differ from the timing declaration')
        for path, field in ((args.manifest, 'manifest_file_sha256'),
                            (args.exposure, 'existing_exposure_file_sha256')):
            if file_sha(path) != recipe[field]:
                raise ValueError(f'Input bytes differ from the timing declaration: {path.name}')
        if not args.preflight and (args.output is None or args.output.exists()
                or args.output.resolve().is_relative_to(args.root.resolve())):
            raise ValueError('Choose a new output file outside the CIFAR-10 data')
        data = CifarTrainingData(args.root, args.manifest)
        protocol_sha256 = file_sha(recipe_path)
        if args.preflight:
            print(json.dumps(dict(preflight=True, protocol_sha256=protocol_sha256)))
            return

        report = benchmark_grouping(data, **recipe['benchmark'])
        if file_sha(args.exposure) != recipe['existing_exposure_file_sha256']:
            raise ValueError('Exposure evidence changed during timing')
        report.update(protocol_sha256=protocol_sha256, runtime=runtime,
            platform=dict(os=platform.system(), architecture=platform.machine()),
            projected_budget=estimate_step_budget(report, seconds_per_arm=recipe['projected_seconds_per_arm']))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x') as handle:
            json.dump(report, handle, indent=2, allow_nan=False)
            handle.write('\n')
        print(json.dumps(dict(output=str(args.output),
            median_joint_over_separate_step_time=report['median_joint_over_separate_step_time'],
            projected_budget=report['projected_budget'])))
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
