import unittest
from unittest.mock import patch

import torch

from jepa_lab.analysis.crossfit import crossfit_reconstruction, make_folds
from jepa_lab.analysis.reconstruction import fit_reconstruction
from jepa_lab.ridge import RidgeConfig


class ColorCrossfitTests(unittest.TestCase):
    def test_held_rows_excluded_from_fits_and_training_mapping_control_breaks_signal(self):
        source = torch.randn(100, 3, generator=torch.Generator().manual_seed(99))
        target = source @ torch.tensor([[2., 1.], [1., -3.], [4., 2.]]) + 5
        rng = torch.get_rng_state().clone()
        folds = make_folds(100)
        fits = []
        def observe(x, y, config):
            fits.append((x.clone(), y.clone()))
            return fit_reconstruction(x, y, config)
        with patch('jepa_lab.analysis.crossfit.fit_reconstruction', side_effect=observe):
            prediction, _, report = crossfit_reconstruction(source, target, folds, config=RidgeConfig(1e-6))
        self.assertGreater(report['reconstruction_r2'], .999)
        for fold, (x, y) in enumerate(fits):
            self.assertTrue(torch.equal(source[folds != fold], x))
            self.assertTrue(torch.equal(target[folds != fold], y))
        shuffled = crossfit_reconstruction(source, target, folds, permute_training=True)[2]
        self.assertLess(shuffled['reconstruction_r2'], .2)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertTrue(torch.allclose(prediction, target, atol=1e-4))

    def test_constant_target_and_invalid_fold_maps(self):
        source = torch.ones(10, 2)
        target = torch.ones(10, 3)
        self.assertIsNone(crossfit_reconstruction(source, target, make_folds(10))[2]['reconstruction_r2'])
        for assignments in (torch.zeros(10, dtype=torch.long), make_folds(10).float(),
                            make_folds(10) * 2, make_folds(10)[:3]):
            with self.assertRaises(ValueError):
                crossfit_reconstruction(source, target, assignments)
