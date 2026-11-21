import unittest

import torch

from experiments.multiblock_spatial_comparison import TRAINED, compare_readouts


class SpatialComparisonTests(unittest.TestCase):
    def test_control_deltas_pair_same_rows_and_same_random_seed(self):
        labels = torch.tensor([0, 1, 2, 3])
        wrong = (labels+1) % 10
        fit = {}

        def add(source, global_readout, quadrant_readout, shuffled):
            for key in ('mean', 'repeated_mean'):
                fit[f'{source}/{key}'] = dict(original=global_readout, patch_shuffle=wrong)
            fit[f'{source}/quadrants'] = dict(original=quadrant_readout,
                patch_shuffle=shuffled, quadrant_shuffle=shuffled)

        add('rgb', wrong, labels, labels)
        for seed in (29, 31):
            correct_random = labels if seed == 29 else wrong
            add(f'{seed}/random', wrong, correct_random, wrong)
            for arm in TRAINED:
                add(f'{seed}/{arm}', wrong, labels, wrong)
        report = compare_readouts(labels, fit, [29, 31])
        self.assertEqual(0., report['paired']['29/joint_extended']['quadrant_gain_minus_random'])
        self.assertEqual(1., report['paired']['31/joint_extended']['quadrant_gain_minus_random'])
        self.assertEqual(4, report['paired']['31/separate']['quadrant_vs_random']['recovered'])
        self.assertEqual(-1., report['paired']['31/joint_matched']['shuffle_delta_minus_random']['patch_shuffle'])
        self.assertEqual(.5, report['mean_accuracy']['random']['quadrants']['original'])
        self.assertTrue(all(count == 0 for count in report['repeated_mean_prediction_disagreements'].values()))
