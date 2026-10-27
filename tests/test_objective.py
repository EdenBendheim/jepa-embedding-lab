import unittest

import torch
from torch.nn import functional as F

from jepa_lab.model import JEPA, ModelConfig
from jepa_lab.objective import ObjectiveConfig, context_variance, objective_terms


class ObjectiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_between_image_variance_does_not_count_spatial_token_spread(self):
        # Large differences within each image cannot disguise identical image means.
        tokens = torch.tensor([[[-10., 10.], [10., -10.]]]*2)
        penalty, std = context_variance(tokens)
        self.assertAlmostEqual(float(penalty), .99, places=6)
        self.assertAlmostEqual(float(std), .01, places=6)
        separated = torch.tensor([[[-.5, -.5]], [[.5, .5]]], requires_grad=True)
        penalty, _ = context_variance(separated)
        self.assertAlmostEqual(float(penalty.detach()), 1-(.25+1e-4)**.5, places=6)
        penalty.backward()
        self.assertTrue(torch.isfinite(separated.grad).all())
        self.assertGreater(float(separated.grad[0, 0, 0]), 0)
        self.assertLess(float(separated.grad[1, 0, 0]), 0)

    def test_zero_weight_keeps_exact_prediction_loss_and_stops_target_gradients(self):
        pred = torch.tensor([[[1., 2.]], [[-1., 0.]]], requires_grad=True)
        target = torch.zeros_like(pred, requires_grad=True)
        terms = objective_terms(pred, target)
        self.assertTrue(torch.equal(terms['loss'], F.smooth_l1_loss(pred, target)))
        self.assertEqual({}, ObjectiveConfig().settings())
        terms['loss'].backward()
        self.assertIsNone(target.grad)
        self.assertIsNotNone(pred.grad)

    def test_enabled_objective_has_visible_context_gradients_without_hidden_pixels(self):
        torch.manual_seed(7)
        model = JEPA(ModelConfig(rows=4, cols=4, embedding_dim=16, encoder_depth=1))
        patches = torch.randn(2, 16, 4, requires_grad=True)
        context, target = [0, 1, 2, 3], [4, 5]
        ordinary = model(patches, context, target)
        state = torch.get_rng_state().clone()
        pred, teacher, encoded = model(patches, context, target, return_context=True)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        self.assertTrue(torch.equal(pred, ordinary[0]))
        self.assertTrue(torch.equal(teacher, ordinary[1]))
        changed = patches.detach().clone(); changed[:, 4:] += 100
        altered = model(changed, context, target, return_context=True)
        self.assertTrue(torch.equal(pred, altered[0]))
        self.assertTrue(torch.equal(encoded, altered[2]))
        terms = objective_terms(pred, teacher, encoded, config=ObjectiveConfig(.1))
        terms['loss'].backward()
        self.assertGreater(float(patches.grad[:, :4].abs().sum()), 0)
        self.assertEqual(0, float(patches.grad[:, 4:].abs().sum()))
        self.assertTrue(all(p.grad is None for p in model.target_encoder.parameters()))

    def test_bad_config_shapes_and_nonfinite_values_are_rejected(self):
        for weight, floor in [(True, 1), (-1, 1), (float('nan'), 1), (1, 0), (1, float('inf'))]:
            with self.subTest(weight=weight, floor=floor), self.assertRaises(ValueError):
                ObjectiveConfig(weight, floor)
        for bad in [torch.zeros(1, 2, 3), torch.zeros(2, 0, 3), torch.ones(2, 2, 3).long(),
                    torch.full((2, 2, 3), float('nan'))]:
            with self.assertRaises(ValueError): context_variance(bad)
        pred = torch.zeros(2, 2, 3)
        with self.assertRaises(ValueError): objective_terms(pred, pred[:, :1])
        with self.assertRaises(ValueError):
            objective_terms(pred, pred, torch.zeros(3, 2, 3), config=ObjectiveConfig(.1))
