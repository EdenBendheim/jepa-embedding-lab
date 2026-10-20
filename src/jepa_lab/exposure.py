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
        ledger_roots = None
        if selected.get('version') == 3 and selected.get('kind') == 'confirmation':
            reference = add(selected['reference_selection'])
            old = selected['exposure_ledger']
            for node in old['sources']:
                old_key = node['selection']['selection_sha256']
                if old_key in records and records[old_key] != node:
                    raise ValueError('Exposure source identity has conflicting evidence')
                records[old_key] = deepcopy(node)
            ledger_roots = deepcopy(old['roots'])
            parents = [reference, *(p for p in ledger_roots if p != reference)]
            wire.pop('reference_selection')
            wire.pop('exposure_ledger')
        elif selected.get('kind') == 'confirmation':
            parents = [add(selected['reference_selection'])]
            parents.extend(add(parent) for parent in selected['excluded_selections'])
            wire.pop('reference_selection')
            wire.pop('excluded_selections')
        records[key] = dict(selection=wire, parents=parents)
        if ledger_roots is not None:
            records[key]['ledger_roots'] = ledger_roots
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
            if (not isinstance(node, dict) or set(node) not in (
                    {'selection', 'parents'}, {'selection', 'parents', 'ledger_roots'})
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
                if type(selected.get('version')) is not int:
                    raise ValueError('Exposure selection versions must be integers')
                if selected.get('version') == 3 and selected.get('kind') == 'confirmation':
                    roots = node.get('ledger_roots')
                    if (not isinstance(roots, list) or not roots or not node['parents']
                            or any(not isinstance(p, str) for p in roots)):
                        raise ValueError('Ledger confirmation requires its original ledger roots')
                    ref = restored[node['parents'][0]]
                    if node['parents'] != [node['parents'][0], *(p for p in sorted(set(roots)) if p != node['parents'][0])]:
                        raise ValueError('Ledger confirmation dependencies differ from its evidence')
                    old = _assemble(records, roots, manifest['manifest_sha256'])
                    selected['reference_selection'] = deepcopy(ref)
                    selected['exposure_ledger'] = old
                    expected = _reserve_from_ledger(manifest, labels, ref, old,
                        validation_per_class=selected['per_class']['validation'], seed=selected['seed'])
                    if selected != expected:
                        raise ValueError('Ledger confirmation source differs from its replayed reservation')
                elif selected.get('kind') == 'confirmation':
                    if not node['parents']:
                        raise ValueError('Confirmation source is missing its reference')
                    selected['reference_selection'] = deepcopy(restored[node['parents'][0]])
                    selected['excluded_selections'] = [deepcopy(restored[p]) for p in node['parents'][1:]]
                elif node['parents']:
                    raise ValueError('Pilot source cannot have confirmation parents')
                if selected.get('version') != 3:
                    if 'ledger_roots' in node:
                        raise ValueError('Only ledger confirmations may retain ledger roots')
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


def _reserve_from_ledger(manifest, labels, reference, ledger, *, validation_per_class, seed):
    """Construct canonical v3 using already replayed sources (internal only)."""
    if (type(validation_per_class) is not int or not 1 <= validation_per_class <= 100
            or type(seed) is not int or not 0 <= seed < 2**63):
        raise ValueError('Ledger confirmation requires 1-100 rows/class and a valid seed')
    if (type(reference.get('version')) is not int or reference['version'] != 1
            or 'kind' in reference):
        raise ValueError('Ledger confirmation requires its original fitting pilot')
    ref_id = reference['selection_sha256']
    nodes = {n['selection']['selection_sha256']: n for n in ledger['sources']}
    if ref_id not in nodes or nodes[ref_id] != dict(selection=reference, parents=[]):
        raise ValueError('Exposure ledger must retain the exact original fitting pilot')
    used = set(ledger['observed_validation_indices'])
    chosen = []
    for label in range(10):
        group = [i for i in manifest['partitions']['validation']['indices']
                 if labels[i] == label and i not in used]
        if len(group) < validation_per_class:
            raise ValueError('Not enough unseen validation examples for every class')
        group.sort(key=lambda i: (hashlib.sha256(f'cifar10-confirmation-v1:{seed}:{i}'.encode()).digest(), i))
        chosen.extend(group[:validation_per_class])
    result = dict(version=3, kind='confirmation', dataset='CIFAR-10',
        manifest_sha256=manifest['manifest_sha256'], seed=seed,
        algorithm='cifar10-confirmation-ledger-v1 SHA-256 within each fixed validation class, excluding source-replayed history',
        reference_selection=deepcopy(reference), exposure_ledger=deepcopy(ledger),
        per_class=dict(train=reference['per_class']['train'], validation=validation_per_class),
        partitions=dict(train=deepcopy(reference['partitions']['train']),
                        validation=dict(official_split='train', indices=sorted(chosen))))
    return result | dict(selection_sha256=document_hash(result, 'selection_sha256'))


def make_ledger_confirmation(manifest, labels, reference, ledger, *, validation_per_class=50, seed=20261019):
    """Reserve another balanced round without embedding a deeper ancestry tree."""
    validate_pilot(reference, manifest, labels)
    validate_exposure_ledger(ledger, manifest, labels)
    return _reserve_from_ledger(manifest, labels, reference, ledger,
                               validation_per_class=validation_per_class, seed=seed)


def validate_ledger_confirmation(selection, manifest, labels):
    try:
        if (not isinstance(selection, dict) or type(selection.get('version')) is not int
                or selection['version'] != 3 or selection.get('kind') != 'confirmation'):
            raise ValueError('Ledger confirmation requires integer version three')
        expected = make_ledger_confirmation(manifest, labels, selection['reference_selection'],
            selection['exposure_ledger'], validation_per_class=selection['per_class']['validation'],
            seed=selection['seed'])
        if selection != expected:
            raise ValueError('Ledger confirmation differs from its source-replayed protocol')
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError('Invalid ledger confirmation schema') from exc
    return selection


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


def extend_exposure_bundle(path, manifest, labels, new_sources, *, expected_ledger_sha256):
    """Add observations while retaining and rechecking every original file pin."""
    previous = load_exposure_bundle(path, manifest, labels)
    if previous['ledger']['ledger_sha256'] != expected_ledger_sha256:
        raise ValueError('Previous exposure ledger differs from its predeclared fingerprint')
    added = make_exposure_bundle(manifest, labels, new_sources)
    evidence = {row['path']: row['file_sha256'] for row in previous['source_files']}
    for row in added['source_files']:
        if row['path'] in evidence and evidence[row['path']] != row['file_sha256']:
            raise ValueError('Original exposure evidence cannot be replaced')
        evidence[row['path']] = row['file_sha256']
    updated = make_exposure_bundle(manifest, labels, list(evidence.items()))
    old_ids = {row['selection']['selection_sha256'] for row in previous['ledger']['sources']}
    new_ids = {row['selection']['selection_sha256'] for row in updated['ledger']['sources']}
    if (not old_ids <= new_ids or not set(previous['ledger']['observed_validation_indices'])
            <= set(updated['ledger']['observed_validation_indices'])):
        raise ValueError('Exposure extension cannot drop earlier sources or observed rows')
    return updated


def main():
    parser = argparse.ArgumentParser(description='Build or recheck validation exposure history; never select or score test')
    parser.add_argument('root', type=Path)
    parser.add_argument('--manifest', type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--source', nargs=2, action='append', metavar=('FILE', 'SHA256'))
    group.add_argument('--bundle', type=Path)
    parser.add_argument('--extend-bundle', type=Path, help='Retain this bundle and add --source evidence into a new output')
    parser.add_argument('--expected-ledger-sha256', help='Required fingerprint of --extend-bundle')
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
            if args.extend_bundle:
                if not args.expected_ledger_sha256:
                    raise ValueError('Extending history requires --expected-ledger-sha256')
                bundle = extend_exposure_bundle(args.extend_bundle, manifest, train.labels, args.source,
                                                expected_ledger_sha256=args.expected_ledger_sha256)
            else:
                if args.expected_ledger_sha256:
                    raise ValueError('--expected-ledger-sha256 requires --extend-bundle')
                bundle = make_exposure_bundle(manifest, train.labels, args.source)
        else:
            if args.output is not None or args.extend_bundle or args.expected_ledger_sha256:
                raise ValueError('Bundle inspection is read-only; omit --output')
            bundle = load_exposure_bundle(args.bundle, manifest, train.labels)
        summary = exposure_summary(bundle['ledger'], manifest, train.labels)
        summary['source_file_count'] = len(bundle['source_files'])
        if args.output:
            protected = {args.manifest.resolve(), *(Path(row['path']) for row in bundle['source_files'])}
            if args.extend_bundle:
                protected.add(args.extend_bundle.resolve())
            if args.output.resolve() in protected:
                raise ValueError('Exposure output must not overwrite its manifest or original sources')
            from .compare import write_report
            write_report(args.output, bundle)
    except (ValueError, OSError, KeyError, TypeError, RuntimeError) as exc:
        parser.error(str(exc))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
