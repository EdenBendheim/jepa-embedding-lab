import unittest

import torch

from jepa_lab.analysis.color_statistics import rgb_statistics


class ColorStatisticsTests(unittest.TestCase):
    def test_channel_order_population_variance_and_spatial_invariance(self):
        images = torch.zeros(1, 3, 32, 32, dtype=torch.uint8)
        images[:, 1, :16] = 255
        images[:, 2] = 255
        before = images.clone()
        expected = torch.tensor([[0., .5, 1., 0., .5, 0.]])
        self.assertTrue(torch.equal(expected, rgb_statistics(images)))
        self.assertTrue(torch.equal(expected, rgb_statistics(images.flip((2, 3)))))
        self.assertTrue(torch.equal(before, images))
        for invalid in (images.float(), images[:, :1], images[:, :, :16], images[:0]):
            with self.assertRaises(ValueError):
                rgb_statistics(invalid)
