from copy import deepcopy
import unittest

from jepa_lab.exposure import make_exposure_ledger, make_ledger_confirmation, validate_exposure_ledger
from jepa_lab.selection import document_hash, excluded_validation_indices, validate_pilot
import test_exposure


class LedgerReservationTests(unittest.TestCase):
    def setUp(self):
        fixture = test_exposure.ExposureTests(); fixture.setUp()
        self.labels, self.manifest, self.pilot, self.fresh = fixture.labels, fixture.manifest, fixture.pilot, fixture.fresh

    def test_many_rounds_keep_exact_ancestry_flat_and_fresh_without_depth_bypass(self):
        ledger = make_exposure_ledger(self.manifest, self.labels, [self.fresh])
        observed = excluded_validation_indices(self.fresh)
        initial_count = len(ledger['sources'])
        for round_no in range(6):
            fresh = make_ledger_confirmation(self.manifest, self.labels, self.pilot, ledger,
                                             validation_per_class=1, seed=round_no)
            rows = set(fresh['partitions']['validation']['indices'])
            self.assertFalse(rows & observed)
            self.assertEqual(self.pilot['partitions']['train'], fresh['partitions']['train'])
            self.assertEqual(fresh, validate_pilot(fresh, self.manifest, self.labels))
            self.assertEqual(observed | rows, excluded_validation_indices(fresh))
            observed |= rows
            ledger = make_exposure_ledger(self.manifest, self.labels, [fresh])
            self.assertEqual(initial_count + round_no + 1, len(ledger['sources']))
            self.assertEqual(observed, set(ledger['observed_validation_indices']))
            for node in ledger['sources']:
                self.assertNotIn('exposure_ledger', node['selection'])
                self.assertNotIn('excluded_selections', node['selection'])

    def test_forged_history_reference_rows_and_legacy_nesting_are_rejected(self):
        ledger = make_exposure_ledger(self.manifest, self.labels, [self.fresh])
        fresh = make_ledger_confirmation(self.manifest, self.labels, self.pilot, ledger, validation_per_class=1)
        edited = deepcopy(fresh); edited['partitions']['validation']['indices'][0] = ledger['observed_validation_indices'][0]
        edited['selection_sha256'] = document_hash(edited, 'selection_sha256')
        with self.assertRaises(ValueError): validate_pilot(edited, self.manifest, self.labels)
        flat = make_exposure_ledger(self.manifest, self.labels, [fresh])
        node = next(n for n in flat['sources'] if n['selection']['version'] == 3)
        node['ledger_roots'] = [self.pilot['selection_sha256']]
        flat['ledger_sha256'] = document_hash(flat, 'ledger_sha256')
        with self.assertRaises(ValueError): validate_exposure_ledger(flat, self.manifest, self.labels)
        with self.assertRaises(ValueError):
            make_ledger_confirmation(self.manifest, self.labels, self.fresh, ledger)
        from jepa_lab.selection import make_confirmation
        with self.assertRaisesRegex(ValueError, 'flattened'):
            make_confirmation(self.manifest, self.labels, self.pilot, exclusions=(fresh,))
