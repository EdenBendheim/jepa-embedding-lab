import math
import unittest
import torch
from jepa_lab.positions import sincos_2d


class PositionTests(unittest.TestCase):
    def test_exact_row_column_layout_origin_frequencies_and_rng(self):
        rng=torch.get_rng_state().clone();positions=sincos_2d(2,3,8)
        self.assertEqual((6,8),tuple(positions.shape));self.assertEqual(torch.float32,positions.dtype)
        self.assertEqual([0.,0.,1.,1.,0.,0.,1.,1.],positions[0].tolist())
        torch.testing.assert_close(positions[3,:4],torch.tensor([math.sin(1),math.sin(.01),math.cos(1),math.cos(.01)]))
        self.assertTrue(torch.equal(positions[0,:4],positions[2,:4]))
        self.assertTrue(torch.equal(positions[0,4:],positions[3,4:]))
        self.assertTrue(torch.equal(rng,torch.get_rng_state()));self.assertFalse(positions.requires_grad)
        self.assertTrue(torch.equal(positions,sincos_2d(2,3,8)))

    def test_shape_bounds_and_independent_returned_values(self):
        for args in ((True,2,8),(0,2,8),(2,2,6),(65,65,8),(2,2,4100)):
            with self.assertRaises(ValueError):sincos_2d(*args)
        a=sincos_2d(2,2,8);a.zero_();self.assertGreater(float(sincos_2d(2,2,8).abs().sum()),0)
