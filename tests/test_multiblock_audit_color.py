import unittest

import torch

from experiments.multiblock_audit_color import measure_rgb_reconstruction
from jepa_lab.analysis.crossfit import make_folds


class AuditColorTests(unittest.TestCase):
    def test_known_color_signal_is_found_only_when_fit_rows_are_aligned(self):
        random = torch.Generator().manual_seed(31)
        rgb = torch.randn(100, 6, generator=random)
        linear = rgb @ torch.randn(6, 8, generator=random)
        result = measure_rgb_reconstruction(
            {'rgb': rgb, '29/joint_extended/original': linear,
             '29/joint_extended/shuffled': linear.flip(0)}, make_folds(100), alpha=1e-6)
        self.assertEqual({'29/joint_extended'}, set(result))
        row = result['29/joint_extended']
        self.assertGreater(row['aligned']['reconstruction_r2'], .999)
        self.assertLess(row['permuted']['reconstruction_r2'], .2)
        self.assertLess(row['residual_geometry']['mean_feature_population_std'], .001)
