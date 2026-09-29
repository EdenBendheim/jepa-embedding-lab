import math
import random
import unittest

from jepa_lab.masking import block_mask
from jepa_lab.patches import patchify


class FoundationTests(unittest.TestCase):
    def test_patch_values_and_positions_preserve_input(self):
        image = [[r * 4 + c for c in range(4)] for r in range(4)]
        patches = patchify(image, 2)
        self.assertEqual(patches[0].values, (0, 1, 4, 5))
        self.assertEqual(patches[-1].values, (10, 11, 14, 15))
        self.assertEqual([(p.row, p.col) for p in patches], [(0, 0), (0, 1), (1, 0), (1, 1)])
        self.assertEqual(sorted(value for p in patches for value in p.values), list(range(16)))

    def test_invalid_images_do_not_silently_crop(self):
        for image, size in (([], 2), ([[1, 2], [3]], 1), ([[1, 2, 3]], 2),
                             ([[1, math.nan]], 1), ([[1, math.inf]], 1), ([[1]], 0)):
            with self.assertRaises(ValueError):
                patchify(image, size)

    def test_masks_partition_grid_without_leaking_target(self):
        for seed in range(50):
            mask = block_mask(6, 8, target_rows=2, target_cols=3, seed=seed)
            context, target, unused = map(set, (mask.context, mask.target, mask.unused))
            self.assertFalse(context & target or context & unused or target & unused)
            self.assertEqual(context | target | unused, set(range(48)))
            self.assertEqual(len(target), 6)
            self.assertTrue(context)
            coordinates = [(i // 8, i % 8) for i in target]
            self.assertEqual(len({r for r, _ in coordinates}), 2)
            self.assertEqual(len({c for _, c in coordinates}), 3)

    def test_seed_is_reproducible_and_does_not_change_global_rng(self):
        random.seed(42)
        expected = random.random()
        random.seed(42)
        self.assertEqual(block_mask(8, 8, seed=9), block_mask(8, 8, seed=9))
        self.assertNotEqual(block_mask(8, 8, seed=9), block_mask(8, 8, seed=10))
        self.assertEqual(random.random(), expected)

    def test_invalid_masks_fail(self):
        for parameters in (dict(rows=0, cols=2), dict(rows=2, cols=2, target_rows=3),
                           dict(rows=2, cols=2), dict(rows=4, cols=4, context_fraction=0),
                           dict(rows=4, cols=4, context_fraction=math.nan),
                           dict(rows=4, cols=4, context_fraction=1.1)):
            with self.assertRaises(ValueError):
                block_mask(**parameters)

    def test_all_remaining_context_and_minimum_context(self):
        self.assertEqual(len(block_mask(3, 3, context_fraction=1).context), 5)
        self.assertEqual(len(block_mask(3, 3, context_fraction=0.001).context), 1)


if __name__ == "__main__":
    unittest.main()
