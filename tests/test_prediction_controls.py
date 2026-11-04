import unittest

import torch

from jepa_lab.analysis.prediction_controls import prediction_controls


class PredictionControlTests(unittest.TestCase):
    def test_own_image_matters_and_template_leaves_it_out(self):
        targets = torch.tensor([0., 2., 4.]).reshape(3, 1, 1)
        result = prediction_controls(targets, targets)
        self.assertEqual(0., result['matched_smooth_l1'])
        self.assertEqual([1, 2], result['reassignment_shifts'])
        self.assertAlmostEqual(6.5 / 3, result['reassigned_minus_matched'])
        self.assertAlmostEqual(5 / 3, result['leave_one_out_template_smooth_l1'])

    def test_image_independent_predictions_and_invalid_pairs(self):
        targets = torch.tensor([0., 2., 4.]).reshape(3, 1, 1)
        self.assertEqual(0., prediction_controls(
            torch.ones_like(targets), targets)['reassigned_minus_matched'])
        for predictions in (targets[:2], targets.double(), targets.reshape(3, 1),
                            torch.full_like(targets, float('inf'))):
            with self.assertRaises(ValueError):
                prediction_controls(predictions, targets)
