import unittest
from unittest.mock import patch

import torch

from jepa_lab.analysis import spatial_sources
from test_cifar10_train import Fixture
import test_color_audit as fixtures


class SpatialSourceTests(unittest.TestCase):
    setUp = fixtures.ColorAuditTests.setUp
    repin = fixtures.ColorAuditTests.repin
    setUpClass = fixtures.ColorAuditTests.setUpClass

    def test_preflight_and_exact_read_order_preserve_existing_sources(self):
        before = {p: p.read_bytes() for p in self.directory.rglob('*') if p.is_file()}
        rng = torch.get_rng_state().clone()
        with patch.object(Fixture, '__getitem__', side_effect=AssertionError('preflight images')):
            inputs = spatial_sources.prepare_sources(self.run)
        with patch.object(spatial_sources, 'CIFAR10Binary', Fixture), \
                patch('jepa_lab.training_budget.train_cifar10', side_effect=AssertionError('training')):
            values = spatial_sources.extract_fitting(self.root, inputs)
        self.assertEqual([(True, i) for i in inputs.indices], Fixture.accessed)
        self.assertEqual(7, len(values))
        for key in inputs.models:
            self.assertTrue(torch.allclose(values[f'{key}/mean'], inputs.references[key].features, atol=1e-6))
            self.assertEqual((10, 64), values[f'{key}/quadrants'].shape)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(before, {p: p.read_bytes() for p in before})

    def test_wrong_completed_encoder_is_rejected_before_images(self):
        original = spatial_sources.load_encoder
        def wrong(*args):
            model, metadata = original(*args)
            return model, metadata | {'completed_steps': 999}
        with patch.object(spatial_sources, 'load_encoder', side_effect=wrong), \
                self.assertRaisesRegex(ValueError, 'differs'):
            spatial_sources.prepare_sources(self.run)
        self.assertEqual([], Fixture.accessed)
