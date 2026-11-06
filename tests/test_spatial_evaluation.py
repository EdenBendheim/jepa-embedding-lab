import unittest
from unittest.mock import patch

import torch

from jepa_lab.analysis.spatial_control import shuffle_quadrants
from jepa_lab.analysis.spatial_evaluation import crossfit_ridge, stratified_folds
from jepa_lab.ridge import fit_ridge


class SpatialEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_layout_signal_survives_quadrants_but_not_mean_and_fits_exclude_scoring_rows(self):
        labels = torch.arange(10).repeat(20)
        signal = torch.nn.functional.one_hot(labels, 10).float()
        spatial = torch.cat((signal, -signal, signal, -signal), dim=1)
        mean = spatial.reshape(200, 4, 10).mean(1)
        rng = torch.get_rng_state().clone()
        assignments = stratified_folds(labels)
        fits = []
        def observe(x, y, config):
            fits.append((x.clone(), y.clone()))
            return fit_ridge(x, y, config)
        with patch('jepa_lab.analysis.spatial_evaluation.fit_ridge', side_effect=observe):
            _, report = crossfit_ridge(spatial, labels, assignments)
        self.assertEqual(1., report['accuracy'])
        for fold, (x, y) in enumerate(fits):
            self.assertTrue(torch.equal(x, spatial[assignments != fold]))
            self.assertTrue(torch.equal(y, labels[assignments != fold]))
            self.assertEqual([4]*10, report['folds'][fold]['scoring_class_counts'])
        self.assertEqual(.1, crossfit_ridge(mean, labels, assignments)[1]['accuracy'])
        shuffled = shuffle_quadrants(spatial, list(range(200)))
        self.assertLess(crossfit_ridge(shuffled, labels, assignments)[1]['accuracy'], .7)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))

    def test_invalid_strata_and_fold_ids_fail_before_fitting(self):
        labels = torch.arange(10).repeat(5)
        for bad in (labels.float(), labels[:20], torch.zeros(50, dtype=torch.long)):
            with self.assertRaises(ValueError):
                stratified_folds(bad)
        for assignments in (torch.zeros(50, dtype=torch.long), labels, stratified_folds(labels).float()):
            with patch('jepa_lab.analysis.spatial_evaluation.fit_ridge', side_effect=AssertionError('fit')), \
                    self.assertRaises(ValueError):
                crossfit_ridge(torch.zeros(50, 3), labels, assignments)
