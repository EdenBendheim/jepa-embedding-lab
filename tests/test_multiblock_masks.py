from dataclasses import replace
import random
import unittest

from jepa_lab.variants.multiblock_masks import multi_block_mask


class MultiBlockMaskTests(unittest.TestCase):
    def test_seeded_tiles_partition_grid_and_never_leak_any_target_into_context(self):
        before = random.getstate()
        for seed in range(30):
            mask = multi_block_mask(8, 8, blocks=4, seed=seed)
            self.assertEqual(mask, multi_block_mask(8, 8, blocks=4, seed=seed))
            self.assertEqual([4]*4, [len(block) for block in mask.blocks])
            self.assertEqual(36, len(mask.context))  # .75 of the 48 non-target positions
            target = {i for block in mask.blocks for i in block}
            self.assertEqual(16, len(target))
            self.assertFalse(target & set(mask.context))
            self.assertEqual(set(range(64)), target | set(mask.context) | set(mask.unused))
        self.assertEqual(before, random.getstate())
        self.assertNotEqual(multi_block_mask(8, 8, seed=29), multi_block_mask(8, 8, seed=31))
        self.assertEqual((), multi_block_mask(4, 4, context_fraction=1).unused)

    def test_smallest_context_and_unplaceable_or_invalid_recipes(self):
        self.assertEqual(1, len(multi_block_mask(4, 4, blocks=3, context_fraction=.001).context))
        for kwargs in ({'blocks': 0}, {'blocks': 5}, {'target_rows': 3}, {'target_rows': 8},
                       {'blocks': True}, {'context_fraction': True}, {'context_fraction': float('nan')},
                       {'seed': -1}, {'seed': True}):
            with self.assertRaises(ValueError):
                multi_block_mask(4, 4, **kwargs)
        with self.assertRaises(ValueError):
            multi_block_mask(4, 4, blocks=4)

    def test_manually_constructed_overlap_or_nonrectangular_blocks_are_rejected(self):
        mask = multi_block_mask(4, 4)
        for changes in ({'context': mask.context + (mask.blocks[0][0],)},
                        {'blocks': (mask.blocks[0], mask.blocks[0])}, {'unused': ()},
                        {'context': list(mask.context)}, {'rows': True}):
            with self.assertRaises(ValueError):
                replace(mask, **changes)
        old, new = mask.blocks[0][-1], mask.unused[0]
        broken = mask.blocks[0][:-1] + (new,)
        with self.assertRaisesRegex(ValueError, 'rectangle'):
            replace(mask, blocks=(broken, mask.blocks[1]), unused=(old, *mask.unused[1:]))
