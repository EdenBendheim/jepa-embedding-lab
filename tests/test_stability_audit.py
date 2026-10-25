import json
from pathlib import Path
import unittest
from unittest.mock import patch

import torch

import test_scale_run as fixtures
from test_budget_driver import Fixture
from jepa_lab import stability_audit


class StabilityAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): fixtures.ScaleRunTests.setUpClass()

    def setUp(self):
        fixtures.ScaleRunTests.setUp(self)
        self.run.training()
        p = patch('jepa_lab.stability_audit.CIFAR10Binary', Fixture)
        p.start(); self.addCleanup(p.stop)

    def test_only_original_training_images_and_no_state_mutation(self):
        before = {p: p.read_bytes() for p in self.directory.rglob('*') if p.is_file()}
        rng = torch.get_rng_state().clone(); Fixture.accessed.clear()
        with patch('jepa_lab.training_budget.train_cifar10', side_effect=AssertionError('training')):
            result = stability_audit.inspect_training(self.run, batch_size=4)
        d = json.loads(self.run.protocol.read_text())
        pilot = json.loads(Path(d['selection_path']).read_text())
        self.assertEqual([(True, i) for i in pilot['partitions']['train']['indices']], Fixture.accessed)
        self.assertEqual(10, result['count']); self.assertEqual(2, len(result['arms']))
        self.assertFalse(result['validation_images_read']); self.assertFalse(result['test_evaluated'])
        self.assertNotIn('train_indices', json.dumps(result)); self.assertNotIn('features', result)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(before, {p: p.read_bytes() for p in before})
        for arm in result['arms'].values():
            self.assertEqual(0., arm['random_context']['between_image_population_variance'] -
                             arm['random_target']['between_image_population_variance'])
            self.assertTrue(arm['parameter_lag']['buffers_identical'])

    def test_invalid_bounds_or_changed_weights_fail_before_images(self):
        with patch.object(Fixture, '__getitem__', side_effect=AssertionError('images')):
            for batch in (0, 33, True):
                with self.assertRaises(ValueError): stability_audit.inspect_training(self.run, batch_size=batch)
            path = self.directory/'checkpoints/scale0.02-seed29.pt'
            saved = torch.load(path, weights_only=True)
            saved['model']['context_encoder.projection.weight'][0, 0] += 1
            torch.save(saved, path)
            with self.assertRaises(ValueError): stability_audit.inspect_training(self.run)
