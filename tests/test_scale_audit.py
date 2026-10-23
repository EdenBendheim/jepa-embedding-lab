from unittest.mock import patch
import unittest

import test_scale_run as fixtures
from test_budget_driver import Fixture
from jepa_lab.scale_audit import audit_training
from jepa_lab import scale_audit


class ScaleAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.ScaleRunTests.setUpClass()

    def test_aggregate_audit_replays_without_training_images_or_row_level_exports(self):
        fixtures.ScaleRunTests.setUp(self)
        self.run.training()
        before = {p: p.read_bytes() for p in self.directory.rglob('*') if p.is_file()}
        with patch('jepa_lab.training_budget.train_cifar10', side_effect=AssertionError('training')), \
                patch.object(Fixture, '__getitem__', side_effect=AssertionError('images')):
            result = audit_training(self.run)
        self.assertEqual(2, len(result['training']))
        for row in result['training'].values():
            self.assertEqual(3, row['diagnostics']['completed_steps'])
            self.assertNotIn('history', row)
            self.assertNotIn('train_indices', str(row))
            self.assertNotIn('partitions', str(row))
        self.assertFalse(result['confirmation_scored'])
        self.assertEqual(before, {p: p.read_bytes() for p in before})
        with patch.object(scale_audit, 'summarize_training', side_effect=ValueError('bad diagnostics')):
            with self.assertRaisesRegex(ValueError, 'bad diagnostics'):
                audit_training(self.run)
