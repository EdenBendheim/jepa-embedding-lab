"""Immutable aggregate caches for the fitting-only RGB audit."""

import json
import os
from pathlib import Path
import tempfile

from ..objective_run import file_sha
from .color_audit import digest, measure, verified_sources


def code_fingerprints():
    directory = Path(__file__).resolve().parent.parent
    return {str(path.relative_to(directory)): file_sha(path)
            for path in sorted(directory.rglob('*.py'))}


def _read(path, identity):
    if path.stat().st_size > 10 * 1024 * 1024:
        raise ValueError('Color cache exceeds the aggregate-report limit')
    saved = json.loads(path.read_text())
    if (not isinstance(saved, dict) or set(saved) != {'identity', 'report', 'report_sha256'}
            or saved['identity'] != identity or not isinstance(saved['report'], dict)
            or saved['report'].get('identity') != identity['measurement']
            or saved['report_sha256'] != digest(saved['report'])):
        raise ValueError('Existing color cache differs; preserve it and use a new output path')
    return saved['report']


def cached_audit(run, output, *, reuse_only=False, expected_identity=None):
    if type(reuse_only) is not bool:
        raise ValueError('Reuse-only must be boolean')
    output = Path(output)
    package = Path(__file__).resolve().parent.parent
    recipe = run.inputs()[0]
    protected = {Path(row['path']).resolve() for row in recipe['sources']} | {run.protocol.resolve()}
    if (output.suffix != '.json' or output.resolve() in protected
            or any(output.resolve().is_relative_to(folder.resolve())
                   for folder in (run.root, run.directory, package))):
        raise ValueError('Color cache cannot replace experiment evidence, code or raw data')
    measurement, indices, sources = verified_sources(run)
    identity = dict(version=1, measurement=measurement, code_sha256=code_fingerprints())
    if expected_identity is not None and expected_identity != identity:
        raise ValueError('Declared color protocol differs from the current inputs, recipe or code')
    if output.exists():
        return _read(output, identity), True
    if reuse_only:
        raise ValueError('Reuse-only color cache is missing')
    report = measure(run, measurement, indices, sources)
    payload = json.dumps(dict(identity=identity, report=report, report_sha256=digest(report)),
                         indent=2, allow_nan=False) + '\n'
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', dir=output.parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        # An exclusive atomic link never overwrites another run's evidence.
        try:
            os.link(temporary, output)
        except FileExistsError:
            return _read(output, identity), True
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return report, False
