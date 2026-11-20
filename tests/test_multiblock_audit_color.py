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

    def test_residualizer_and_heads_only_fit_on_the_current_fold_complement(self):
        from unittest.mock import patch
        from experiments import multiblock_audit_color as audit
        from jepa_lab.analysis.spatial_evaluation import stratified_folds

        labels = torch.arange(30) % 10
        random = torch.Generator().manual_seed(29)
        rgb = torch.randn(30, 6, generator=random)
        features = torch.nn.functional.one_hot(labels, 10).float() + rgb @ torch.randn(6, 10, generator=random)
        folds = stratified_folds(labels, folds=3)
        with patch.object(audit, 'fit_reconstruction', wraps=audit.fit_reconstruction) as reconstruction, \
             patch.object(audit, 'fit_ridge', wraps=audit.fit_ridge) as classifier:
            result = audit.residual_classification(
                {'rgb': rgb, '29/trained/original': features}, labels, folds,
                reconstruction_alpha=1e-6, classifier_alpha=.1)
        self.assertEqual(3, reconstruction.call_count)
        self.assertEqual(6, classifier.call_count)
        for fold, call in enumerate(reconstruction.call_args_list):
            self.assertTrue(torch.equal(call.args[0], rgb[folds != fold]))
            self.assertTrue(torch.equal(call.args[1], features[folds != fold]))
            for fit in classifier.call_args_list[2*fold:2*fold+2]:
                self.assertTrue(torch.equal(fit.args[1], labels[folds != fold]))
        contrast = result['comparisons']['29/trained']['rgb_plus_residual']
        self.assertEqual(30, contrast['candidate_correct'])
        self.assertGreater(contrast['accuracy_delta'], .5)
