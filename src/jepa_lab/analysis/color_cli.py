"""Run or replay a prospectively declared fitting-only RGB audit."""

import argparse
import json
from pathlib import Path

import torch

from ..objective_run import ObjectiveRun, file_sha
from .color_audit import verified_sources
from .color_cache import cached_audit, code_fingerprints


def declare_protocol(run):
    measurement, _, _ = verified_sources(run)
    return dict(version=1, kind='fitting_only_rgb_reconstruction',
                identity=dict(version=1, measurement=measurement, code_sha256=code_fingerprints()))


def load_protocol(path, expected_sha256):
    path = Path(path)
    if file_sha(path) != expected_sha256:
        raise ValueError('Declared color protocol checksum differs')
    value = json.loads(path.read_text())
    if (not isinstance(value, dict) or set(value) != {'version', 'kind', 'identity'}
            or type(value['version']) is not int or value['version'] != 1
            or value['kind'] != 'fitting_only_rgb_reconstruction' or not isinstance(value['identity'], dict)):
        raise ValueError('Invalid declared color protocol')
    return value['identity']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--objective-protocol', type=Path, required=True)
    parser.add_argument('--objective-sha256', required=True)
    parser.add_argument('--diagnostic-protocol', type=Path, required=True)
    parser.add_argument('--diagnostic-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reuse-only', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    try:
        expected = load_protocol(args.diagnostic_protocol, args.diagnostic_sha256)
        run = ObjectiveRun(args.root, args.directory, protocol=args.objective_protocol,
                           expected_sha=args.objective_sha256)
        report, reused = cached_audit(run, args.output, reuse_only=args.reuse_only, expected_identity=expected)
        print(json.dumps(dict(count=report['identity']['count'], sources=len(report['arms']),
            cache_reused=reused, images_indexed_this_invocation=not reused,
            validation_images_read=False, test_images_read=False), allow_nan=False))
    except (ValueError, OSError, KeyError, TypeError, RuntimeError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
