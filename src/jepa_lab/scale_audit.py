"""Publish aggregate diagnostics from completed scale budgets without image access."""
import argparse
import json
from pathlib import Path

import torch

from .compare import write_report
from .exposure import load_exposure_bundle
from .scale_run import DEFAULT_DIRECTORY, ScaleRun
from .training_diagnostics import summarize_training


def audit_training(run):
    reports = run.training(reuse_only=True)
    arms = {}
    for name, report in reports.items():
        diagnostics = summarize_training(report, window=100)
        arms[name] = dict(config=report['config'], settings=report['settings'],
            checkpoint_sha256=report['checkpoint_sha256'],
            train_image_order_sha256=report['train_image_order_sha256'],
            model_size=report['model_size'], diagnostics=diagnostics)
    return dict(version=1, protocol_sha256=run.expected_sha, training=arms,
        actual_image_order_matched=True, trainable_capacity_matched=True,
        images_sampled=False, models_trained=False, classifiers_fitted=False,
        confirmation_scored=False, test_evaluated=False,
        limitation='Replayed training diagnostics only. Finite loss and token variance do not establish useful embeddings or rule out collapse.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('data/cifar-10-batches-bin'))
    parser.add_argument('--directory', type=Path, default=DEFAULT_DIRECTORY)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    try:
        run = ScaleRun(args.root, args.directory)
        result = audit_training(run)
        # The publication path must not replace any original input or local state.
        d, manifest, ref, fresh, labels = run.inputs()
        protected = {Path(row['path']).resolve() for row in d['sources']} | {run.protocol.resolve()}
        bundle = load_exposure_bundle(d['exposure_bundle_path'], manifest, labels)
        protected.update(Path(row['path']).resolve() for row in bundle['source_files'])
        for path in run.directory.rglob('*'):
            if path.is_file():
                protected.add(path.resolve())
        if args.output.resolve() in protected or args.output.resolve().is_relative_to(run.root.resolve()):
            raise ValueError('Training audit cannot replace experiment evidence')
        write_report(args.output, result)
        print(json.dumps(dict(runs=len(result['training']), images_sampled=False,
                              confirmation_scored=False, test_evaluated=False)))
    except (ValueError, OSError, KeyError, TypeError, RuntimeError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
