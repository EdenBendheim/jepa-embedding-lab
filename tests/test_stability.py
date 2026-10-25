import unittest

import torch

from jepa_lab.stability import paired_features


class StabilityTests(unittest.TestCase):
    def test_pair_geometry_invariance_and_isolation(self):
        x = torch.tensor([[1., 0.], [-1., 0.], [0., 2.], [0., -2.]], requires_grad=True)
        before = x.detach().clone(); rng = torch.get_rng_state().clone()
        y = x.detach().flip(1) * 3 + 7
        report = paired_features(x, y)
        self.assertAlmostEqual(1., report['centered_linear_cka'])
        self.assertGreater(report['rmse'], 0)
        self.assertAlmostEqual(1., paired_features(x, x)['mean_paired_cosine'])
        self.assertEqual(0., paired_features(x, x)['rmse'])
        self.assertTrue(torch.equal(before, x)); self.assertIsNone(x.grad)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        shuffled = paired_features(x, y[[0, 2, 1, 3]])
        self.assertLess(shuffled['centered_linear_cka'], .99)

    def test_pair_degenerate_and_invalid(self):
        for x in (torch.zeros(3, 2), torch.ones(1, 2), torch.ones(4, 2)):
            self.assertIsNone(paired_features(x, x)['centered_linear_cka'])
        self.assertIsNone(paired_features(torch.zeros(2, 2), torch.ones(2, 2))['mean_paired_cosine'])
        for x in (torch.ones(0, 2), torch.ones(1001, 2), torch.ones(2, 4097),
                  torch.ones(2, 2).double(), torch.full((2, 2), float('nan'))):
            with self.assertRaises(ValueError): paired_features(x, x)
        with self.assertRaises(ValueError): paired_features(torch.ones(3, 2), torch.ones(2, 2))

    def test_spatial_variation_can_hide_identical_image_means(self):
        from jepa_lab.stability import token_spread
        spatial = torch.tensor([[[-1., 0.], [1., 0.]]]).repeat(3, 1, 1)
        result = token_spread(spatial)
        self.assertGreater(result['within_image_population_variance'], 0)
        self.assertEqual(0., result['between_fraction'])
        self.assertEqual(0, result['pooled_geometry']['centered_numerical_rank'])
        images = torch.tensor([-1., 0., 1.])[:, None, None].expand(3, 2, 2).clone()
        image_result = token_spread(images)
        self.assertEqual(1., image_result['between_fraction'])
        mixed = token_spread(spatial + images)
        self.assertAlmostEqual(mixed['total_population_variance'],
            mixed['within_image_population_variance'] + mixed['between_image_population_variance'])
        scaled = token_spread((spatial + images) * 3 + 2)
        self.assertAlmostEqual(mixed['between_fraction'], scaled['between_fraction'])
        self.assertAlmostEqual(9*mixed['total_population_variance'], scaled['total_population_variance'])

    def test_spatial_degenerate_invalid_and_unchanged(self):
        from jepa_lab.stability import token_spread
        x = torch.ones(1, 1, 3, requires_grad=True); rng = torch.get_rng_state().clone()
        self.assertIsNone(token_spread(x)['between_fraction'])
        self.assertIsNone(x.grad); self.assertTrue(torch.equal(x, torch.ones_like(x)))
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        for x in (torch.ones(2, 2), torch.ones(2, 65, 2), torch.ones(0, 2, 2),
                  torch.ones(2, 2, 2).double(), torch.full((2, 2, 2), float('inf'))):
            with self.assertRaises(ValueError): token_spread(x)

    def test_parameter_lag_matches_ema_interpolation_and_keeps_state(self):
        from copy import deepcopy
        from jepa_lab.model import JEPA, ModelConfig
        from jepa_lab.stability import parameter_lag
        model = JEPA(ModelConfig(embedding_dim=8, heads=2, encoder_depth=1,
                                  position_encoding='sincos2d'))
        with torch.no_grad():
            for p in model.context_encoder.parameters(): p.add_(.5)
        before = deepcopy(model.state_dict()); rng = torch.get_rng_state().clone()
        lag = parameter_lag(model.context_encoder, model.target_encoder)
        self.assertAlmostEqual(.5, lag['rmse'], places=6)
        self.assertTrue(lag['buffers_identical']); self.assertEqual(1, lag['buffer_count'])
        self.assertTrue(all(torch.equal(v, model.state_dict()[k]) for k, v in before.items()))
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        model.update_target(.75)
        after = parameter_lag(model.context_encoder, model.target_encoder)
        self.assertAlmostEqual(.75 * lag['rmse'], after['rmse'], places=6)
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_parameter_lag_rejects_mismatches_and_handles_zero_reference(self):
        from jepa_lab.stability import parameter_lag
        a, b = torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)
        with torch.no_grad():
            for p in b.parameters(): p.zero_()
        self.assertIsNone(parameter_lag(a, b)['relative_l2'])
        with self.assertRaises(ValueError): parameter_lag(a, torch.nn.Linear(3, 2))
        with self.assertRaises(ValueError): parameter_lag(a, b.double())
        with self.assertRaises(ValueError): parameter_lag(torch.nn.Identity(), torch.nn.Identity())
        with torch.no_grad(): a.weight[0, 0] = float('nan')
        with self.assertRaises(ValueError): parameter_lag(a, torch.nn.Linear(2, 2))
