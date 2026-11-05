import unittest
from unittest.mock import patch

import torch

from jepa_lab import objective_features
from jepa_lab.analysis.color_audit import inspect_fitting
from test_cifar10_train import Fixture
import test_objective_workflow as fixtures


class ColorAuditTests(unittest.TestCase):
    repin = fixtures.ObjectiveWorkflowTests.repin

    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        fixtures.ObjectiveWorkflowTests.setUp(self)
        d, m, ref, _, labels, _ = self.run.inputs()
        for key in objective_features.source_specs(d):
            objective_features.feature(self.run, d, m, ref, labels, key, 'train')
        Fixture.accessed.clear()

    def test_only_exact_fitting_images_are_indexed_and_completed_evidence_is_preserved(self):
        indices = self.run.inputs()[2]['partitions']['train']['indices']
        before = {p: p.read_bytes() for p in self.directory.rglob('*') if p.is_file()}
        rng = torch.get_rng_state().clone()
        with patch('jepa_lab.analysis.color_audit.CIFAR10Binary', Fixture), \
                patch('jepa_lab.training_budget.train_cifar10', side_effect=AssertionError('training')), \
                patch.object(objective_features, 'extract_cifar10', side_effect=AssertionError('extraction')):
            report = inspect_fitting(self.run)
        self.assertEqual([(True, i) for i in indices], Fixture.accessed)
        self.assertEqual(4, len(report['arms']))
        for row in report['arms'].values():
            self.assertEqual(5, len(row['aligned']['folds']))
            self.assertEqual(10, row['residual_geometry']['count'])
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(before, {p: p.read_bytes() for p in before})

    def test_missing_or_corrupt_feature_fails_before_raw_images(self):
        path = self.directory / 'features/context_variance-seed29-train.pt'
        original = path.read_bytes()
        with patch('jepa_lab.analysis.color_audit.CIFAR10Binary', Fixture):
            path.unlink()
            with self.assertRaisesRegex(ValueError, 'missing'):
                inspect_fitting(self.run)
            path.write_bytes(original)
            state = torch.load(path, weights_only=True)
            state['features'][0, 0] += .1
            torch.save(state, path)
            with self.assertRaises(ValueError):
                inspect_fitting(self.run)
        self.assertEqual([], Fixture.accessed)
