import unittest
from unittest.mock import patch

import torch

from experiments import multiblock_spatial_scoring as scoring
from jepa_lab.analysis.spatial_evaluation import stratified_folds


class SpatialScoringTests(unittest.TestCase):
    def test_original_fold_heads_are_reused_and_repeat_penalty_is_matched(self):
        torch.set_num_threads(1)
        labels = torch.arange(30) % 10
        original = torch.nn.functional.one_hot(labels, 10).float()
        shifted = torch.nn.functional.one_hot((labels+1) % 10, 10).float()
        cells = torch.cat((original, original.roll(1, 1), original.roll(2, 1), original.roll(3, 1)), dim=1)
        panel = {'rgb/original/mean': original, 'rgb/patch_shuffle/mean': shifted,
                 'rgb/original/quadrants': cells, 'rgb/patch_shuffle/quadrants': cells}
        folds = stratified_folds(labels, folds=3)
        with patch.object(scoring, 'fit_ridge', wraps=scoring.fit_ridge) as fit:
            report, predictions = scoring.score_spatial(panel, labels, tuple(range(30)), folds,
                                                        alpha=.1, quadrant_seed=43)
        self.assertEqual(9, fit.call_count)
        self.assertTrue(torch.equal(original[folds != 0], fit.call_args_list[0].args[0]))
        self.assertEqual(.4, report['rgb/repeated_mean']['alpha'])
        for view in ('original', 'patch_shuffle'):
            self.assertTrue(torch.equal(predictions['rgb/mean'][view], predictions['rgb/repeated_mean'][view]))
        patch_drop = report['rgb/mean']['transfer']['patch_shuffle']
        self.assertEqual((30, 0, 30), (patch_drop['baseline_correct'], patch_drop['candidate_correct'], patch_drop['regressed']))
        self.assertEqual(0, report['rgb/quadrants']['transfer']['patch_shuffle']['predictions_changed'])
        self.assertIn('quadrant_shuffle', report['rgb/quadrants']['transfer'])
