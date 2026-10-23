"""Source-pinned, bounded execution of a prospectively declared scale contrast."""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from .cifar10 import CIFAR10Binary, make_manifest
from .compare import write_report
from .exposure import load_exposure_bundle, make_ledger_confirmation
from .model import ModelConfig, config_dict
from .model_size import model_size
from .selection import validate_pilot
from .training_budget import run_training_budget, verify_budget_state

PROTOCOL = Path('experiments/scale-sources-20261020.json')
PROTOCOL_SHA = '26c3f7facd70e1b2593ecd500345798d5583f33a31a93ab3d413d5da56146a1d'
DEFAULT_DIRECTORY = Path('runs/scale-20261020')


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class ScaleRun:
    def __init__(self, root, directory=DEFAULT_DIRECTORY, *, protocol=PROTOCOL, expected_sha=PROTOCOL_SHA):
        self.root, self.directory, self.protocol = Path(root), Path(directory), Path(protocol)
        self.expected_sha = expected_sha

    def inputs(self):
        if file_sha(self.protocol) != self.expected_sha:
            raise ValueError('Scale protocol bytes changed')
        d = json.loads(self.protocol.read_text())
        pins = {Path(row['path']).resolve(): row['sha256'] for row in d['sources']}
        required = [d[key] for key in ('manifest_path', 'selection_path', 'exposure_bundle_path')]
        required.extend(d['reused_scale_one_checkpoints'].values())
        if any(Path(path).resolve() not in pins for path in required):
            raise ValueError('Scale inputs need original source-file pins')
        for path, expected in pins.items():
            if file_sha(path) != expected:
                raise ValueError('Declared scale source bytes changed')
        if (d['position_scales'] != [.02, 1.] or d['position_encoding'] != 'sincos2d'
                or len(d['training_seeds']) != len(set(d['training_seeds']))):
            raise ValueError('Scale run requires the declared .02 / 1.0 fixed-position contrast')
        manifest = json.loads(Path(d['manifest_path']).read_text())
        train, test = CIFAR10Binary(self.root), CIFAR10Binary(self.root, train=False)
        if (manifest != make_manifest(train, test, seed=manifest['seed'])
                or manifest['manifest_sha256'] != d['manifest_sha256']):
            raise ValueError('Scale dataset or fixed split changed')
        reference = validate_pilot(json.loads(Path(d['selection_path']).read_text()), manifest, train.labels)
        bundle = load_exposure_bundle(d['exposure_bundle_path'], manifest, train.labels)
        if bundle['ledger']['ledger_sha256'] != d['ledger_sha256']:
            raise ValueError('Scale exposure history changed')
        fresh = make_ledger_confirmation(manifest, train.labels, reference, bundle['ledger'],
            validation_per_class=d['confirmation_per_class'], seed=d['confirmation_seed'])
        if fresh['selection_sha256'] != d['confirmation_selection_sha256']:
            raise ValueError('Scale confirmation reservation changed')
        # Never permit an output directory to replace pinned historical evidence.
        protected = set(pins) | {self.protocol.resolve()}
        protected.update(Path(row['path']).resolve() for row in bundle['source_files'])
        if any(self.directory.resolve() == path or self.directory.resolve() in path.parents for path in protected):
            raise ValueError('Scale output directory contains protected experiment evidence')
        selected = self.directory/'confirmation-selection.json'
        if selected.exists() and json.loads(selected.read_text()) != fresh:
            raise ValueError('Existing scale reservation differs; preserve the prior file')
        return d, manifest, reference, fresh, train.labels

    @staticmethod
    def config(d, scale):
        return ModelConfig(patch_dim=48, embedding_dim=d['embedding_dim'], heads=d['heads'],
            encoder_depth=d['encoder_depth'], predictor_depth=d['predictor_depth'],
            position_encoding=d['position_encoding'], position_scale=scale)

    def checkpoint(self, d, scale, seed):
        old = d['reused_scale_one_checkpoints'].get(str(seed)) if scale == 1. else None
        return Path(old) if old else self.directory/'checkpoints'/f'scale{scale}-seed{seed}.pt'

    def training(self, *, reuse_only=False):
        d, manifest, reference, fresh, labels = self.inputs()
        settings = lambda seed: dict(seed=seed, batch_size=d['batch_size'],
            learning_rate=d['learning_rate'], momentum=d['momentum'])
        mask = d['mask']
        for scale in d['position_scales']:
            for seed in d['training_seeds']:
                path = self.checkpoint(d, scale, seed)
                recipe = settings(seed)
                if mask != dict(target_rows=2, target_cols=2, context_fraction=.75):
                    recipe['mask'] = mask
                identity = dict(format_version=1, dataset='CIFAR-10',
                    manifest_sha256=manifest['manifest_sha256'], config=config_dict(self.config(d, scale)),
                    settings=recipe, runtime=dict(torch=str(torch.__version__), cpu_threads=torch.get_num_threads()))
                if path.exists():
                    verify_budget_state(path, identity, manifest['partitions']['train']['indices'], d['total_steps'])
                elif reuse_only or (scale == 1. and str(seed) in d['reused_scale_one_checkpoints']):
                    raise ValueError('Pinned or reuse-only scale checkpoint is missing')
        selected = self.directory/'confirmation-selection.json'
        if not selected.exists():
            if reuse_only:
                raise ValueError('Reserved scale selection is missing')
            write_report(selected, fresh)
        reports = {}
        for scale in d['position_scales']:
            for seed in d['training_seeds']:
                name = f'scale{scale}-seed{seed}'
                result = run_training_budget(self.root, d['manifest_path'], self.checkpoint(d, scale, seed),
                    total_steps=d['total_steps'], chunk_steps=1000, checkpoint_every=50,
                    **settings(seed), config=self.config(d, scale), **mask,
                    reuse_only=reuse_only or (scale == 1. and str(seed) in d['reused_scale_one_checkpoints']))
                result['model_size'] = model_size(self.config(d, scale))
                output = self.directory/'training'/f'{name}.json'
                if output.exists() and json.loads(output.read_text()) != result:
                    raise ValueError('Completed scale training report changed')
                if not output.exists():
                    if reuse_only:
                        raise ValueError('Completed scale training report is missing')
                    write_report(output, result)
                reports[name] = result
        for seed in d['training_seeds']:
            rows = [reports[f'scale{scale}-seed{seed}'] for scale in d['position_scales']]
            if len({row['train_image_order_sha256'] for row in rows}) != 1:
                raise ValueError('Actual training image order differs across scales')
            if len({row['model_size']['trainable_parameters'] for row in rows}) != 1:
                raise ValueError('Scale arms have unequal trainable capacity')
        return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('data/cifar-10-batches-bin'))
    parser.add_argument('--directory', type=Path, default=DEFAULT_DIRECTORY)
    parser.add_argument('--reuse-only', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    try:
        reports = ScaleRun(args.root, args.directory).training(reuse_only=args.reuse_only)
        print(json.dumps({key: {k: row[k] for k in ('completed_steps', 'train_image_order_sha256', 'model_size')}
                          for key, row in reports.items()}, indent=2))
    except (ValueError, OSError, KeyError, TypeError, RuntimeError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
