from dataclasses import replace
import unittest

import torch

from jepa_lab.model import JEPA, ModelConfig, config_dict


class PositionScaleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_scale_changes_used_positions_without_changing_initialization_or_rng(self):
        for mode in ('learned', 'sincos2d'):
            config = ModelConfig(rows=4, cols=4, embedding_dim=16,
                                 encoder_depth=1, position_encoding=mode)
            torch.manual_seed(19)
            reference = JEPA(config)
            rng = torch.get_rng_state().clone()
            torch.manual_seed(19)
            scaled = JEPA(replace(config, position_scale=.02))
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            for key, value in reference.state_dict().items():
                self.assertTrue(torch.equal(value, scaled.state_dict()[key]), key)
            patches = torch.randn(2, 16, 4, requires_grad=True)
            indices = torch.tensor([[0, 1, 2], [4, 5, 6]])
            encoder = scaled.context_encoder
            expected = encoder.blocks(encoder.projection(patches.gather(
                1, indices.unsqueeze(-1).expand(-1, -1, 4))) + encoder.positions[indices] * .02)
            self.assertTrue(torch.equal(expected, encoder(patches, indices)))
            loss = scaled.loss(patches, [0, 1, 2], [3, 4])
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertEqual(0., float(patches.grad[:, 3:].abs().sum()))
            self.assertTrue(all(p.grad is None for p in scaled.target_encoder.parameters()))
            self.assertEqual(.02, config_dict(scaled.config)['position_scale'])

    def test_zero_scale_removes_target_position_signal_and_defaults_keep_old_identity(self):
        model = JEPA(ModelConfig(rows=4, cols=4, embedding_dim=16,
                                 encoder_depth=1, position_scale=0))
        predictions, _ = model(torch.randn(2, 16, 4), [0, 1, 2], [3, 4])
        torch.testing.assert_close(predictions[:, 0], predictions[:, 1])
        self.assertNotIn('position_scale', config_dict(ModelConfig()))
        self.assertEqual(config_dict(ModelConfig()), config_dict(ModelConfig(position_scale=1)))
        for invalid in (True, -1, float('nan'), float('inf'), '0.02', None):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                ModelConfig(position_scale=invalid)
