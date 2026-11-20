from collections import Counter
import unittest

import torch

from experiments.multiblock_patch_control import shuffle_patches
from jepa_lab.analysis.color_statistics import rgb_statistics


class PatchControlTests(unittest.TestCase):
    def test_shuffle_preserves_whole_patches_and_color_without_perturbing_rng(self):
        images = torch.randint(0, 256, (2, 3, 32, 32), generator=torch.Generator().manual_seed(7),
                               dtype=torch.uint8)

        def patch_counts(image):
            return Counter(bytes(image[:, row:row+4, col:col+4].contiguous().numpy().tobytes())
                           for row in range(0, 32, 4) for col in range(0, 32, 4))

        rng = torch.get_rng_state().clone()
        shuffled = shuffle_patches(images, [3, 19], seed=29)
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))
        self.assertFalse(torch.equal(shuffled, images))
        self.assertTrue(torch.equal(rgb_statistics(shuffled), rgb_statistics(images)))
        self.assertTrue(torch.equal(shuffled.flip(0), shuffle_patches(images.flip(0), [19, 3], seed=29)))
        self.assertTrue(torch.equal(shuffled[0:1], shuffle_patches(images[0:1], [3], seed=29)))
        self.assertEqual([patch_counts(row) for row in images], [patch_counts(row) for row in shuffled])

    def test_ambiguous_rows_rejected(self):
        images = torch.zeros(2, 3, 32, 32, dtype=torch.uint8)
        with self.assertRaisesRegex(ValueError, 'distinct'):
            shuffle_patches(images, [3, 3], seed=29)
