"""Replayable, flat evidence of validation selections already observed.

Fingerprints establish internal consistency, not who produced a source or
whether the caller disclosed every experiment. Keep the complete local history.
"""

from copy import deepcopy

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
