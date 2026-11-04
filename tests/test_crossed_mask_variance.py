import unittest

import torch

from jepa_lab.analysis.mask_variance import crossed_variance


class CrossedMaskVarianceTests(unittest.TestCase):
    def test_separates_mask_bias_from_image_dependent_mask_response(self):
        image = torch.tensor([-1., 1.])[:, None]
        mask = torch.tensor([-2., 2.])[None, :]
        values = (image + mask + 1.5 * image * mask + 7).unsqueeze(-1)
        report = crossed_variance(values)
        self.assertEqual({'image': 1., 'mask': 4., 'interaction': 9.}, report['variance'])
        self.assertEqual(14., report['total_variance'])
        self.assertAlmostEqual(1., sum(report['fraction'].values()))
        self.assertEqual(report, crossed_variance(values.flip((0, 1))))

    def test_constant_and_incomplete_inputs(self):
        self.assertEqual(dict.fromkeys(('image', 'mask', 'interaction')),
                         crossed_variance(torch.ones(2, 3, 1))['fraction'])
        for values in (torch.ones(2, 1, 1), torch.ones(1, 2, 1),
                       torch.full((2, 2, 1), float('nan'))):
            with self.assertRaises(ValueError):
                crossed_variance(values)
