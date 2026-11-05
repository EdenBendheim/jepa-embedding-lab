import unittest

import torch

from jepa_lab.analysis.reconstruction import fit_reconstruction
from jepa_lab.ridge import RidgeConfig


class ReconstructionTests(unittest.TestCase):
    def test_predicts_new_rows_using_fitting_only_means_and_scale(self):
        source = torch.arange(8, dtype=torch.float32).reshape(-1, 1)
        target = torch.cat((source * 2 + 3, -source), dim=1)
        original = source.clone()
        model = fit_reconstruction(source, target, RidgeConfig(1e-6))
        prediction = model.predict(torch.tensor([[9.], [20.]]))
        self.assertTrue(torch.allclose(prediction, torch.tensor([[21., -9.], [43., -20.]]), atol=1e-4))
        self.assertEqual(3.5, float(model.source_mean[0]))
        self.assertTrue(torch.equal(original, source))

    def test_constant_sources_intercept_and_invalid_alignment(self):
        source = torch.ones(4, 2)
        target = torch.arange(4, dtype=torch.float32).reshape(-1, 1)
        model = fit_reconstruction(source, target)
        self.assertTrue(torch.equal(model.predict(source), torch.full((4, 1), 1.5)))
        for bad_source, bad_target in ((source[:1], target[:1]), (source[:3], target),
                                       (source.double(), target), (torch.ones(4, 65), target)):
            with self.assertRaises(ValueError):
                fit_reconstruction(bad_source, bad_target)
        with self.assertRaises(ValueError):
            model.predict(torch.ones(2, 3))
