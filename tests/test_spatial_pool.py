import unittest

import torch

from jepa_lab.analysis.spatial_pool import spatial_mean


class SpatialPoolTests(unittest.TestCase):
    def test_exact_quadrant_and_channel_order_and_global_mean(self):
        grid = torch.empty(2, 8, 8, 3)
        for row in range(2):
            for col in range(2):
                grid[:, row*4:(row+1)*4, col*4:(col+1)*4] = torch.tensor([row*2+col, 10, 20])
        tokens = grid.reshape(2, 64, 3).requires_grad_()
        before = tokens.detach().clone()
        result = spatial_mean(tokens)
        self.assertEqual((2, 12), result.shape)
        self.assertEqual([0, 10, 20, 1, 10, 20, 2, 10, 20, 3, 10, 20], result[0].tolist())
        self.assertTrue(torch.equal(spatial_mean(tokens, bins=1), tokens.mean(1)))
        self.assertTrue(torch.equal(result.reshape(2, 4, 3).mean(1), tokens.mean(1)))
        self.assertTrue(torch.equal(tokens.detach(), before))
        self.assertFalse(result.requires_grad)

    def test_shape_type_grid_and_finite_contracts(self):
        for tokens in (torch.zeros(0, 64, 2), torch.zeros(1, 63, 2), torch.zeros(1, 64, 0),
                       torch.zeros(1, 64, 513), torch.zeros(1, 64, 2).double(),
                       torch.full((1, 64, 2), float('nan'))):
            with self.assertRaises(ValueError):
                spatial_mean(tokens)
        for bins in (True, 0, 3, 2.):
            with self.assertRaises(ValueError):
                spatial_mean(torch.zeros(1, 64, 2), bins=bins)
