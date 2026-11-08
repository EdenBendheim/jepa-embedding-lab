"""Declare, run and replay a fitting-only frozen spatial-pooling comparison."""

import argparse
import json
import os
from pathlib import Path
import tempfile

import torch

from ..objective_run import ObjectiveRun, file_sha
from .color_audit import digest
from .color_cache import code_fingerprints
from .spatial_cache import cached_features
from .spatial_evaluation import evaluate, stratified_folds
from .spatial_sources import prepare_sources


def _declaration(inputs):
    assignments = stratified_folds(inputs.labels)
    return dict(version=1, kind='fitting_only_spatial_pooling',
        feature_identity=dict(version=1, inputs=inputs.identity, code_sha256=code_fingerprints()),
        evaluation=dict(classifier='ridge', alpha=.1, folds=5, fold_seed=29,
            fold_assignments_sha256=digest(assignments.tolist()), hyperparameter_search=False,
            encoder_training_overlap=True))


def declare_protocol(run):
    """Validate inputs and folds without image indexing, fitting or inference."""
    return _declaration(prepare_sources(run))


def load_protocol(path, expected_sha):
    path = Path(path)
    if file_sha(path) != expected_sha:
        raise ValueError('Spatial protocol checksum differs')
    declared = json.loads(path.read_text())
    if (not isinstance(declared, dict) or set(declared) != {'version', 'kind', 'feature_identity', 'evaluation'}
            or type(declared['version']) is not int or declared['version'] != 1
            or declared['kind'] != 'fitting_only_spatial_pooling'):
        raise ValueError('Invalid spatial protocol')
    return declared


def _read_report(path, identity):
    if path.stat().st_size > 20 * 1024 * 1024:
        raise ValueError('Spatial aggregate report exceeds 20 MiB')
    saved = json.loads(path.read_text())
    if (not isinstance(saved, dict) or set(saved) != {'identity', 'report', 'report_sha256'}
            or saved['identity'] != identity or saved['report_sha256'] != digest(saved['report'])):
        raise ValueError('Existing spatial report differs; preserve it and choose a new path')
    return saved['report']


def run_analysis(run, features_path, output, declared, *, reuse_only=False):
    if type(reuse_only) is not bool:
        raise ValueError('Reuse-only must be boolean')
    output = Path(output)
    package = Path(__file__).resolve().parent.parent
    if (output.suffix != '.json' or any(output.resolve().is_relative_to(folder.resolve())
                                      for folder in (run.root, run.directory, package))):
        raise ValueError('Spatial reports must be outside raw data, completed experiments and code')
    inputs, features, features_reused, features_sha = cached_features(run, features_path,
        reuse_only=reuse_only or output.exists(), expected_identity=declared['feature_identity'])
    if declared != _declaration(inputs):
        raise ValueError('Spatial evaluation differs from the declared recipe')
    identity = dict(version=1, declared=declared, feature_artifact_sha256=features_sha)
    if output.exists():
        return _read_report(output, identity), features_reused, True
    if reuse_only:
        raise ValueError('Reuse-only spatial report is missing')
    report = evaluate(inputs, features)
    state = dict(identity=identity, report=report, report_sha256=digest(report))
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', dir=output.parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(json.dumps(state, indent=2, allow_nan=False) + '\n')
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, output)
        except FileExistsError:
            return _read_report(output, identity), features_reused, True
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return report, features_reused, False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--objective-protocol', type=Path, required=True)
    parser.add_argument('--objective-sha256', required=True)
    parser.add_argument('--diagnostic-protocol', type=Path, required=True)
    parser.add_argument('--diagnostic-sha256', required=True)
    parser.add_argument('--features', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reuse-only', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    try:
        declared = load_protocol(args.diagnostic_protocol, args.diagnostic_sha256)
        run = ObjectiveRun(args.root, args.directory, protocol=args.objective_protocol,
                           expected_sha=args.objective_sha256)
        report, features_reused, report_reused = run_analysis(run, args.features, args.output, declared,
                                                             reuse_only=args.reuse_only)
        print(json.dumps(dict(count=report['count'], sources=len(report['sources']),
            feature_cache_reused=features_reused, report_cache_reused=report_reused,
            images_indexed_this_invocation=not features_reused, validation_images_read=False,
            test_images_read=False), allow_nan=False))
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
