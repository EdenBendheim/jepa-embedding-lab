"""Inspect frozen scale-run encoders on the original fitting images only."""

import hashlib
import json

import torch

from .cifar10 import CIFAR10Binary, patchify_rgb
from .embeddings import load_encoder, pixel_features, random_encoder
from .feature_geometry import feature_geometry
from .model import config_dict
from .stability import paired_features, parameter_lag, token_spread


@torch.inference_mode()
def _tokens(model, images, batch_size):
    model.eval()
    context, target = [], []
    for start in range(0, len(images), batch_size):
        patches = patchify_rgb(images[start:start+batch_size])
        positions = torch.arange(64).expand(len(patches), -1)
        context.append(model.context_encoder(patches, positions))
        target.append(model.target_encoder(patches, positions))
    return torch.cat(context), torch.cat(target)


def inspect_training(run, *, batch_size=16):
    """Reuse completed training, then compare fixed full-image encoder endpoints.

    No optimization, masked prediction, class scoring, validation extraction or
    test extraction occurs. Labels are used by existing split/selection replay,
    never by these representation diagnostics.
    """
    if type(batch_size) is not int or not 1 <= batch_size <= 32:
        raise ValueError('Audit batch size must be 1-32')
    reports = run.training(reuse_only=True)
    d, manifest, reference, _, _ = run.inputs()
    indices = reference['partitions']['train']['indices']
    if not 1 <= len(indices) <= 1000:
        raise ValueError('Audit needs 1-1000 original fitting images')
    if not set(indices).issubset(manifest['partitions']['train']['indices']):
        raise ValueError('Audit selection must be training-only')
    train = CIFAR10Binary(run.root)
    images = torch.stack([train[index][0] for index in indices])
    arms = {}
    for scale in d['position_scales']:
        for seed in d['training_seeds']:
            name = f'scale{scale}-seed{seed}'
            model, metadata = load_encoder(run.checkpoint(d, scale, seed), manifest['manifest_sha256'])
            if (metadata['checkpoint_sha256'] != reports[name]['checkpoint_sha256']
                    or metadata['config'] != config_dict(run.config(d, scale))
                    or metadata['completed_steps'] != d['total_steps']):
                raise ValueError('Audit checkpoint differs from the verified completed run')
            random = random_encoder(run.config(d, scale), seed)
            context, target = _tokens(model, images, batch_size)
            random_context, random_target = _tokens(random, images, batch_size)
            arms[name] = dict(checkpoint=metadata,
                context=token_spread(context), target=token_spread(target),
                random_context=token_spread(random_context), random_target=token_spread(random_target),
                context_target=paired_features(context.mean(1), target.mean(1)),
                context_random=paired_features(context.mean(1), random_context.mean(1)),
                parameter_lag=parameter_lag(model.context_encoder, model.target_encoder))
    return dict(version=1, protocol_sha256=run.expected_sha, manifest_sha256=manifest['manifest_sha256'],
        training_selection_sha256=reference['selection_sha256'],
        ordered_training_indices_sha256=hashlib.sha256(json.dumps(indices, separators=(',', ':')).encode()).hexdigest(),
        count=len(indices), batch_size=batch_size, runtime=dict(torch=str(torch.__version__), cpu_threads=torch.get_num_threads()),
        arms=arms, pixels=feature_geometry(pixel_features(images)),
        images_sampled=True, partition='train', classifiers_fitted=False, models_trained=False,
        validation_images_read=False, test_evaluated=False,
        limitation='Exploratory training-only endpoint diagnostics, not downstream quality, temporal stability or a new held-out comparison')


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _code_fingerprints():
    from pathlib import Path
    # Include transitive local helpers, so changed extraction/diagnostic code
    # cannot silently reuse an older calculation. This is not authentication.
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(__file__).parent.glob('*.py'))}


def cached_audit(run, output, *, batch_size=16, reuse_only=False):
    """Return a byte-bound saved diagnostic report, or create it atomically.

    Cache replay rechecks the dataset, source pins, completed journals, weights,
    runtime and code. It does not read images or recalculate token diagnostics.
    A checksum detects accidental corruption, not a deliberately forged report.
    """
    from pathlib import Path
    from .compare import write_report
    from .exposure import load_exposure_bundle
    if type(batch_size) is not int or not 1 <= batch_size <= 32:
        raise ValueError('Audit batch size must be 1-32')
    reports = run.training(reuse_only=True)
    d, manifest, reference, _, labels = run.inputs()
    output = Path(output)
    protected = {Path(row['path']).resolve() for row in d['sources']} | {run.protocol.resolve()}
    bundle = load_exposure_bundle(d['exposure_bundle_path'], manifest, labels)
    protected.update(Path(row['path']).resolve() for row in bundle['source_files'])
    code_directory = Path(__file__).resolve().parent
    if (output.resolve() in protected or output.resolve().is_relative_to(run.root.resolve())
            or output.resolve().is_relative_to(run.directory.resolve())
            or output.resolve().is_relative_to(code_directory)):
        raise ValueError('Stability cache cannot replace experiment evidence, code or raw data')
    identity = dict(version=1, protocol_sha256=run.expected_sha,
        manifest_sha256=manifest['manifest_sha256'], selection_sha256=reference['selection_sha256'],
        checkpoints={name: row['checkpoint_sha256'] for name, row in reports.items()},
        batch_size=batch_size, runtime=dict(torch=str(torch.__version__), cpu_threads=torch.get_num_threads()),
        code_sha256=_code_fingerprints())
    if output.exists():
        saved = json.loads(output.read_text())
        if (not isinstance(saved, dict) or set(saved) != {'identity', 'report', 'report_sha256'}
                or saved['identity'] != identity or saved['report_sha256'] != _digest(saved['report'])):
            raise ValueError('Existing stability cache differs; preserve it and use a new output path')
        return saved['report'], True
    if reuse_only:
        raise ValueError('Reuse-only stability cache is missing')
    result = inspect_training(run, batch_size=batch_size)
    write_report(output, dict(identity=identity, report=result, report_sha256=_digest(result)))
    return result, False


def main():
    import argparse
    from pathlib import Path
    from .scale_run import DEFAULT_DIRECTORY, ScaleRun
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('data/cifar-10-batches-bin'))
    parser.add_argument('--directory', type=Path, default=DEFAULT_DIRECTORY)
    parser.add_argument('--output', type=Path, required=True, help='Local JSON cache outside the scale-run directory')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--reuse-only', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    try:
        result, reused = cached_audit(ScaleRun(args.root, args.directory), args.output,
            batch_size=args.batch_size, reuse_only=args.reuse_only)
        print(json.dumps(dict(count=result['count'], runs=len(result['arms']), cache_reused=reused,
            images_sampled_this_invocation=not reused, validation_images_read=False, test_evaluated=False)))
    except (ValueError, OSError, KeyError, TypeError, RuntimeError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
