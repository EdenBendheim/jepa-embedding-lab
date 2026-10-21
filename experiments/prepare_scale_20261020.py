"""Recheck pinned scale-experiment inputs and reserve unscored validation rows."""
import argparse
import hashlib
import json
from pathlib import Path

from jepa_lab.cifar10 import CIFAR10Binary, make_manifest
from jepa_lab.compare import write_report
from jepa_lab.exposure import load_exposure_bundle, make_ledger_confirmation, exposure_summary
from jepa_lab.model import ModelConfig
from jepa_lab.model_size import model_size
from jepa_lab.selection import validate_pilot

PROTOCOL = Path('experiments/scale-sources-20261020.json')
PROTOCOL_SHA = '26c3f7facd70e1b2593ecd500345798d5583f33a31a93ab3d413d5da56146a1d'


def prepare(root, protocol_path, expected_sha, output):
    protocol_path, output = Path(protocol_path), Path(output)
    if hashlib.sha256(protocol_path.read_bytes()).hexdigest() != expected_sha:
        raise ValueError('Scale protocol bytes changed')
    d = json.loads(protocol_path.read_text())
    pins = {Path(row['path']).resolve(): row['sha256'] for row in d['sources']}
    required = [d[key] for key in ('manifest_path', 'selection_path', 'exposure_bundle_path')]
    required.extend(d['reused_scale_one_checkpoints'].values())
    if any(Path(name).resolve() not in pins for name in required):
        raise ValueError('Scale experiment inputs need source-file pins')
    for path, expected in pins.items():
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('Declared scale source bytes changed')
    manifest = json.loads(Path(d['manifest_path']).read_text())
    train, test = CIFAR10Binary(root), CIFAR10Binary(root, train=False)
    if (manifest != make_manifest(train, test, seed=manifest['seed'])
            or manifest['manifest_sha256'] != d['manifest_sha256']):
        raise ValueError('Scale dataset or fixed split changed')
    reference = validate_pilot(json.loads(Path(d['selection_path']).read_text()), manifest, train.labels)
    bundle = load_exposure_bundle(d['exposure_bundle_path'], manifest, train.labels)
    ledger = bundle['ledger']
    if ledger['ledger_sha256'] != d['ledger_sha256']:
        raise ValueError('Scale exposure history changed')
    fresh = make_ledger_confirmation(manifest, train.labels, reference, ledger,
                                    validation_per_class=d['confirmation_per_class'], seed=d['confirmation_seed'])
    if fresh['selection_sha256'] != d['confirmation_selection_sha256']:
        raise ValueError('Scale confirmation reservation changed')
    protected = set(pins) | {protocol_path.resolve()}
    protected.update(Path(row['path']).resolve() for row in bundle['source_files'])
    if output.resolve() in protected:
        raise ValueError('Reservation cannot overwrite experiment evidence')
    sizes = [model_size(ModelConfig(patch_dim=48, embedding_dim=d['embedding_dim'], heads=d['heads'],
             encoder_depth=d['encoder_depth'], predictor_depth=d['predictor_depth'],
             position_encoding=d['position_encoding'], position_scale=scale)) for scale in d['position_scales']]
    if output.exists():
        if json.loads(output.read_text()) != fresh:
            raise ValueError('Existing scale reservation differs; preserve the prior file')
    else:
        write_report(output, fresh)
    return dict(protocol_sha256=expected_sha, selection_sha256=fresh['selection_sha256'],
                ledger_sha256=ledger['ledger_sha256'], exposure=exposure_summary(ledger, manifest, train.labels),
                reserved_count=len(fresh['partitions']['validation']['indices']), position_scales=d['position_scales'],
                trainable_parameters=[size['trainable_parameters'] for size in sizes],
                raw_images_sampled=False, models_trained=False, classifiers_fitted=False,
                confirmation_scored=False, test_evaluated=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('data/cifar-10-batches-bin'))
    parser.add_argument('--output', type=Path, default=Path('runs/scale-20261020/confirmation-selection.json'))
    args = parser.parse_args()
    try:
        print(json.dumps(prepare(args.root, PROTOCOL, PROTOCOL_SHA, args.output), indent=2))
    except (ValueError, KeyError, TypeError, OSError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
