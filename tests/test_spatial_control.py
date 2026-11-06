import unittest

import torch

from jepa_lab.analysis.spatial_control import shuffle_quadrants


class SpatialControlTests(unittest.TestCase):
    def test_per_image_identity_is_batch_independent_and_preserves_cell_contents_and_rng(self):
        features = torch.arange(24.).reshape(3, 8)
        before = features.clone()
        rng = torch.get_rng_state().clone()
        result = shuffle_quadrants(features, [17, 9, 201])
        for row, index in enumerate([17, 9, 201]):
            self.assertTrue(torch.equal(result[row:row+1], shuffle_quadrants(features[row:row+1], [index])))
            self.assertEqual(sorted(features[row].reshape(4, 2).tolist()), sorted(result[row].reshape(4, 2).tolist()))
        reordered = shuffle_quadrants(features[[2, 0, 1]], [201, 17, 9])
        self.assertTrue(torch.equal(result[[2, 0, 1]], reordered))
        self.assertTrue(torch.equal(result.reshape(3, 4, 2).mean(1), features.reshape(3, 4, 2).mean(1)))
        self.assertTrue(torch.equal(features, before))
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertFalse(torch.equal(result, features))

    def test_invalid_identity_or_feature_contracts_are_rejected(self):
        for indices in ([1], [1, 1], [True, 2], [-1, 2], [50000, 2]):
            with self.assertRaises(ValueError):
                shuffle_quadrants(torch.zeros(2, 8), indices)
        for value in (torch.zeros(2, 7), torch.zeros(2, 8).double(), torch.full((2, 8), float('inf'))):
            with self.assertRaises(ValueError):
                shuffle_quadrants(value, [1, 2])
