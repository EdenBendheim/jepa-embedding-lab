from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from jepa_lab.analysis.spatial_cache import cached_features
from test_cifar10_train import Fixture
import test_spatial_sources as fixtures


class SpatialCacheTests(unittest.TestCase):
    setUp = fixtures.SpatialSourceTests.setUp
    repin = fixtures.SpatialSourceTests.repin
    setUpClass = fixtures.SpatialSourceTests.setUpClass

    def output(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        return Path(folder.name) / 'features.pt'

    def test_exact_no_image_replay_and_changed_features_are_rejected_without_replacement(self):
        output = self.output()
        with patch('jepa_lab.analysis.spatial_sources.CIFAR10Binary', Fixture):
            _, features, reused, sha = cached_features(self.run, output)
        self.assertFalse(reused)
        before = output.read_bytes()
        with patch.object(Fixture, '__getitem__', side_effect=AssertionError('images')), \
                patch('jepa_lab.analysis.spatial_sources.paired_embeddings', side_effect=AssertionError('encoding')):
            _, again, reused, restored_sha = cached_features(self.run, output, reuse_only=True)
        self.assertTrue(reused)
        self.assertEqual(sha, restored_sha)
        self.assertTrue(all(torch.equal(value, again[key]) for key, value in features.items()))
        self.assertEqual(before, output.read_bytes())
        state = torch.load(output, weights_only=True)
        state['features']['prediction-seed29/quadrants'][0, 0] += .1
        torch.save(state, output)
        corrupt = output.read_bytes()
        with self.assertRaisesRegex(ValueError, 'content differs'):
            cached_features(self.run, output)
        self.assertEqual(corrupt, output.read_bytes())

    def test_missing_protected_and_failed_save_leave_no_partial_cache(self):
        output = self.output()
        with self.assertRaisesRegex(ValueError, 'missing'):
            cached_features(self.run, output, reuse_only=True)
        with patch('jepa_lab.analysis.spatial_sources.CIFAR10Binary', Fixture), \
                patch('jepa_lab.analysis.spatial_cache.os.link', side_effect=OSError('disk error')), \
                self.assertRaisesRegex(OSError, 'disk error'):
            cached_features(self.run, output)
        self.assertEqual([], list(output.parent.iterdir()))
        with self.assertRaisesRegex(ValueError, 'outside'):
            cached_features(self.run, self.directory/'features.pt')
