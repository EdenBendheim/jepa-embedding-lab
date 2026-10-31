"""Consume reserved rows before fresh access, including interrupted attempts."""
import json

from .compare import write_report
from .exposure import extend_exposure_bundle, exposure_summary, load_exposure_bundle
from .objective_run import file_sha


def _evidence(run, d, manifest, fresh, labels):
    lock_path = run.directory/'selection-lock.json'
    if not lock_path.exists():
        raise ValueError('Fresh objective features require the observation gate and selection lock')
    lock = json.loads(lock_path.read_text())
    if (lock.get('protocol_sha256') != run.expected_sha or lock.get('training_lock_sha256')
            != file_sha(run.directory/'training-lock.json')):
        raise ValueError('Objective selection lock refers to different training/protocol evidence')
    for key, expected in lock['tuning_report_sha256'].items():
        path = run.directory/'tuning'/f'{key}.json'
        if file_sha(path) != expected:
            raise ValueError('Locked objective tuning report changed')
        report = json.loads(path.read_text())
        for grid in report['grids']:
            for candidate in grid['candidates']:
                if file_sha(candidate['path']) != candidate['artifact_sha256']:
                    raise ValueError('Locked objective candidate artifact changed')
    for record in lock['neighbors'].values():
        if file_sha(record['path']) != record['index']['artifact_sha256']:
            raise ValueError('Locked objective neighbor artifact changed')
    selected = run.directory/'confirmation-selection.json'
    if json.loads(selected.read_text()) != fresh:
        raise ValueError('Objective reservation changed')
    updated = extend_exposure_bundle(d['exposure_bundle_path'], manifest, labels,
        [(selected, file_sha(selected))], expected_ledger_sha256=d['ledger_sha256'])
    receipt = dict(version=1, protocol_sha256=run.expected_sha,
        selection_sha256=fresh['selection_sha256'], selection_lock_sha256=file_sha(lock_path),
        bundle_sha256=updated['bundle_sha256'],
        policy='Consume before fresh image indexing or scoring; retain on interruption')
    return updated, receipt


def observation_gate(run, d, manifest, fresh, labels, *, register=False):
    updated, receipt = _evidence(run, d, manifest, fresh, labels)
    paths = [(run.directory/'observed-exposure.json', updated),
             (run.directory/'observation-receipt.json', receipt)]
    for path, expected in paths:
        if path.exists():
            if json.loads(path.read_text()) != expected:
                raise ValueError('Existing objective observation evidence differs')
        elif not register:
            raise ValueError('Fresh objective features require the observation gate receipt')
    if register:
        for path, expected in paths:
            if not path.exists():
                write_report(path, expected)
    restored = load_exposure_bundle(paths[0][0], manifest, labels)
    return exposure_summary(restored['ledger'], manifest, labels) | dict(
        registered_before_fresh_access=True, conservative_on_interruption=True)
