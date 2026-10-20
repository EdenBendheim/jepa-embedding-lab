from copy import deepcopy
import random
import unittest

from jepa_lab.cifar10 import make_manifest
from jepa_lab.exposure import make_exposure_ledger, validate_exposure_ledger, exposure_summary
from jepa_lab.selection import make_pilot, make_confirmation, document_hash, excluded_validation_indices
from test_embeddings import Fixture


class ExposureTests(unittest.TestCase):
    def setUp(self):
        self.labels = Fixture('unused').labels
        self.manifest = make_manifest(Fixture('unused'), Fixture('unused', train=False))
        self.pilot = make_pilot(self.manifest, self.labels, train_per_class=2, validation_per_class=2)
        self.other = make_pilot(self.manifest, self.labels, seed=9, validation_per_class=1)
        self.fresh = make_confirmation(self.manifest, self.labels, self.pilot,
                                       exclusions=(self.other,), validation_per_class=2)

    def test_flat_source_replay_deduplicates_preserves_history_and_rng(self):
        before = deepcopy(self.fresh)
        rng = random.getstate()
        ledger = make_exposure_ledger(self.manifest, self.labels, [self.fresh, self.pilot, self.fresh])
        self.assertEqual(3, len(ledger['sources']))
        self.assertEqual(excluded_validation_indices(self.fresh), set(ledger['observed_validation_indices']))
        self.assertEqual(ledger, make_exposure_ledger(self.manifest, self.labels, [self.pilot, self.fresh]))
        self.assertEqual(before, self.fresh)
        self.assertEqual(rng, random.getstate())
        report = exposure_summary(ledger, self.manifest, self.labels)
        self.assertEqual(5000, report['observed_validation_count'] + report['remaining_validation_count'])
        self.assertFalse(report['test_selected'])
        self.assertNotIn('observed_validation_indices', report)

    def test_rehashed_union_source_changes_cycles_or_unreachable_records_fail(self):
        ledger = make_exposure_ledger(self.manifest, self.labels, [self.fresh])
        changed = deepcopy(ledger)
        changed['observed_validation_indices'].pop()
        source = deepcopy(ledger)
        source['sources'][0]['selection']['seed'] += 1
        cycle = deepcopy(ledger)
        node = next(n for n in cycle['sources'] if n['parents'])
        node['parents'][0] = node['selection']['selection_sha256']
        omitted = deepcopy(ledger)
        omitted['sources'].pop()
        for value in (changed, source, cycle, omitted, ledger | {'version': True}):
            value['ledger_sha256'] = document_hash(value, 'ledger_sha256')
            with self.assertRaises(ValueError):
                validate_exposure_ledger(value, self.manifest, self.labels)
        bad = deepcopy(self.fresh)
        bad['reference_selection']['seed'] += 1
        bad['selection_sha256'] = document_hash(bad, 'selection_sha256')
        with self.assertRaises(ValueError): make_exposure_ledger(self.manifest, self.labels, [bad])
