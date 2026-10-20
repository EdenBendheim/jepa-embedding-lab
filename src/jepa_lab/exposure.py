"""Replayable, flat evidence of validation selections already observed.

Fingerprints establish internal consistency, not who produced a source or
whether the caller disclosed every experiment. Keep the complete local history.
"""

from copy import deepcopy
import argparse
import hashlib
import json
from pathlib import Path

from .cifar10 import CIFAR10Binary, make_manifest
from .selection import document_hash, validate_pilot

MAX_SOURCES = 128


def _closure(records, roots):
    pending, found = list(roots), set()
    while pending:
        key = pending.pop()
        if key in found:
            continue
        if key not in records:
            raise ValueError("Exposure ledger is missing a source dependency")
        found.add(key)
        node = records[key]
        pending.extend(node['parents'])
    return found


def _assemble(records, roots, manifest_sha256):
    roots = sorted(set(roots))
    keys = _closure(records, roots)
    nodes = [deepcopy(records[key]) for key in sorted(keys)]
    rows = sorted({i for node in nodes for i in node['selection']['partitions']['validation']['indices']})
    result = dict(version=1, kind='validation_exposure_ledger', dataset='CIFAR-10',
                  manifest_sha256=manifest_sha256, roots=roots, sources=nodes,
                  observed_validation_indices=rows)
    return result | dict(ledger_sha256=document_hash(result, 'ledger_sha256'))


def make_exposure_ledger(manifest, labels, selections):
    """Validate original selections, then retain each source once in a flat DAG."""
    if not isinstance(selections, (list, tuple)) or not 1 <= len(selections) <= MAX_SOURCES:
        raise ValueError('Exposure ledger requires 1-128 original source selections')
    records = {}

    def add(selected):
        validate_pilot(selected, manifest, labels)
        key = selected['selection_sha256']
        if key in records:
            return key
        wire = deepcopy(selected)
        parents = []
        if selected.get('kind') == 'confirmation':
            parents = [add(selected['reference_selection'])]
            parents.extend(add(parent) for parent in selected['excluded_selections'])
            wire.pop('reference_selection')
            wire.pop('excluded_selections')
        records[key] = dict(selection=wire, parents=parents)
        if len(records) > MAX_SOURCES:
            raise ValueError('Exposure ledger exceeds 128 unique sources')
        return key

    roots = [add(selected) for selected in selections]
    result = _assemble(records, roots, manifest['manifest_sha256'])
    return validate_exposure_ledger(result, manifest, labels)


def validate_exposure_ledger(ledger, manifest, labels):
    """Replay all declared source selections; never trust a self-hashed row union."""
    try:
        if (not isinstance(ledger, dict) or type(ledger.get('version')) is not int
                or ledger['version'] != 1 or ledger.get('kind') != 'validation_exposure_ledger'
                or ledger.get('manifest_sha256') != manifest['manifest_sha256']
                or ledger.get('ledger_sha256') != document_hash(ledger, 'ledger_sha256')
                or not isinstance(ledger.get('sources'), list)
                or not 1 <= len(ledger['sources']) <= MAX_SOURCES
                or not isinstance(ledger.get('roots'), list) or not ledger['roots']):
            raise ValueError('Invalid exposure ledger identity or source bound')
        records = {}
        for node in ledger['sources']:
            if (not isinstance(node, dict) or set(node) != {'selection', 'parents'}
                    or not isinstance(node['selection'], dict) or not isinstance(node['parents'], list)
                    or any(not isinstance(key, str) for key in node['parents'])):
                raise ValueError('Invalid exposure source record')
            key = node['selection']['selection_sha256']
            if not isinstance(key, str) or key in records:
                raise ValueError('Exposure source IDs must be unique strings')
            records[key] = node
        if any(not isinstance(key, str) for key in ledger['roots']):
            raise ValueError('Exposure roots must be source IDs')
        if _closure(records, ledger['roots']) != set(records):
            raise ValueError('Exposure sources must all be reachable from declared roots')
        restored, pending = {}, set(records)
        while pending:
            progress = False
            for key in sorted(pending):
                node = records[key]
                if any(parent not in restored for parent in node['parents']):
                    continue
                selected = deepcopy(node['selection'])
                if selected.get('kind') == 'confirmation':
                    if not node['parents']:
                        raise ValueError('Confirmation source is missing its reference')
                    selected['reference_selection'] = deepcopy(restored[node['parents'][0]])
                    selected['excluded_selections'] = [deepcopy(restored[p]) for p in node['parents'][1:]]
                elif node['parents']:
                    raise ValueError('Pilot source cannot have confirmation parents')
                validate_pilot(selected, manifest, labels)
                restored[key] = selected
                pending.remove(key)
                progress = True
            if not progress:
                raise ValueError('Exposure source dependencies contain a cycle or missing parent')
        expected = _assemble(records, ledger['roots'], manifest['manifest_sha256'])
        if ledger != expected:
            raise ValueError('Exposure ledger differs from its canonical source-derived union')
    except (KeyError, TypeError, AttributeError, RecursionError) as exc:
        raise ValueError('Invalid exposure ledger schema') from exc
    return ledger


def exposure_summary(ledger, manifest, labels):
    """Aggregate only: avoid printing exact rows or source paths to public logs."""
    validate_exposure_ledger(ledger, manifest, labels)
    used = set(ledger['observed_validation_indices'])
    available = set(manifest['partitions']['validation']['indices']) - used
    return dict(ledger_sha256=ledger['ledger_sha256'], source_count=len(ledger['sources']),
                observed_validation_count=len(used), remaining_validation_count=len(available),
                remaining_per_class={str(c): sum(labels[i] == c for i in available) for c in range(10)},
                test_selected=False,
                limitation='Covers declared source selections only; fingerprints do not prove complete disclosure or producer authenticity')


def make_exposure_bundle(manifest, labels, sources):
    """Pin original local file bytes as well as replaying their selection content."""
    if not isinstance(sources, (list, tuple)) or not 1 <= len(sources) <= MAX_SOURCES:
        raise ValueError('Provide 1-128 source file/hash pairs')
    evidence, selections, seen = [], [], set()
    for filename, expected in sources:
        path = Path(filename).resolve()
        if (not isinstance(expected, str) or len(expected) != 64
                or any(c not in '0123456789abcdef' for c in expected) or str(path) in seen):
            raise ValueError('Source paths must be distinct and each needs a lowercase SHA-256 pin')
        seen.add(str(path))
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != expected:
            raise ValueError('Exposure source file bytes differ from their pin')
        selected = json.loads(payload)
        validate_pilot(selected, manifest, labels)
        selections.append(selected)
        evidence.append(dict(path=str(path), file_sha256=expected,
                             selection_sha256=selected['selection_sha256']))
    ledger = make_exposure_ledger(manifest, labels, selections)
    result = dict(version=1, kind='validation_exposure_bundle', ledger=ledger,
                  source_files=sorted(evidence, key=lambda row: row['path']))
    return result | dict(bundle_sha256=document_hash(result, 'bundle_sha256'))


def load_exposure_bundle(path, manifest, labels):
    """Re-read pinned original files before trusting a cached exposure ledger."""
    try:
        bundle = json.loads(Path(path).read_text())
        if (not isinstance(bundle, dict) or type(bundle.get('version')) is not int
                or bundle['version'] != 1 or bundle.get('kind') != 'validation_exposure_bundle'
                or bundle.get('bundle_sha256') != document_hash(bundle, 'bundle_sha256')):
            raise ValueError('Invalid exposure bundle fingerprint')
        expected = make_exposure_bundle(manifest, labels,
            [(row['path'], row['file_sha256']) for row in bundle['source_files']])
        if bundle != expected:
            raise ValueError('Exposure bundle differs from its pinned source evidence')
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError('Invalid exposure bundle schema') from exc
    return bundle


def main():
    parser = argparse.ArgumentParser(description='Build or recheck validation exposure history; never select or score test')
    parser.add_argument('root', type=Path)
    parser.add_argument('--manifest', type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--source', nargs=2, action='append', metavar=('FILE', 'SHA256'))
    group.add_argument('--bundle', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    try:
        manifest = json.loads(args.manifest.read_text())
        train, test = CIFAR10Binary(args.root), CIFAR10Binary(args.root, train=False)
        if manifest != make_manifest(train, test, seed=manifest['seed']):
            raise ValueError('Manifest differs from the current dataset and split')
        if args.source:
            if args.output is None:
                raise ValueError('Building exposure history requires --output')
            bundle = make_exposure_bundle(manifest, train.labels, args.source)
        else:
            if args.output is not None:
                raise ValueError('Bundle inspection is read-only; omit --output')
            bundle = load_exposure_bundle(args.bundle, manifest, train.labels)
        summary = exposure_summary(bundle['ledger'], manifest, train.labels)
        summary['source_file_count'] = len(bundle['source_files'])
        if args.output:
            if args.output.resolve() in {args.manifest.resolve(), *(Path(row['path']) for row in bundle['source_files'])}:
                raise ValueError('Exposure output must not overwrite its manifest or original sources')
            from .compare import write_report
            write_report(args.output, bundle)
    except (ValueError, OSError, KeyError, TypeError, RuntimeError) as exc:
        parser.error(str(exc))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
