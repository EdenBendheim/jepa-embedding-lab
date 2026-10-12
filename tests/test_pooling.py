import unittest
from unittest.mock import patch

import torch

from jepa_lab.embeddings import image_embeddings, random_encoder
from jepa_lab.model import ModelConfig
from jepa_lab.pooling import POOLING, pool_tokens, pooling_dimension, pooling_mode


class PoolingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def test_spatial_moments_maxima_and_batch_isolation(self):
        values = torch.arange(64, dtype=torch.float32).reshape(1, 64, 1).repeat(2, 1, 3)
        values[1] += 100
        values.requires_grad_(True)
        before = values.detach().clone()
        self.assertEqual([[31.5] * 3, [131.5] * 3], pool_tokens(values).tolist())
        self.assertEqual([[63.] * 3, [163.] * 3], pool_tokens(values, "max").tolist())
        moments = pool_tokens(values, "mean_std")
        self.assertEqual((2, 6), tuple(moments.shape))
        self.assertTrue(torch.equal(moments[:, :3], pool_tokens(values)))
        self.assertTrue(torch.allclose(moments[:, 3:], torch.full((2, 3), (1365 / 4)**0.5)))
        self.assertTrue(torch.equal(moments, pool_tokens(values.flip(1), "mean_std")))
        self.assertFalse(moments.requires_grad)
        self.assertIsNone(values.grad)
        self.assertTrue(torch.equal(values.detach(), before))
        self.assertTrue((pool_tokens(torch.ones(1, 64, 4), "mean_std")[:, 4:] == 0).all())

    def test_invalid_shapes_values_and_contracts(self):
        for value in (torch.zeros(0, 64, 2), torch.zeros(1, 63, 2), torch.zeros(1, 64, 0),
                      torch.zeros(1, 64, 2, dtype=torch.long), torch.full((1, 64, 2), float("nan"))):
            with self.assertRaises(ValueError): pool_tokens(value)
        for mode in ("attention", None, []):
            with self.assertRaises(ValueError): pool_tokens(torch.ones(1, 64, 2), mode)
        for width in (True, 0, 2049):
            with self.assertRaises(ValueError): pooling_dimension(width, "mean")
        for mode, description in POOLING.items():
            self.assertEqual(mode, pooling_mode(description))
        with self.assertRaises(ValueError): pooling_mode("unknown")

    def test_frozen_encoder_pooling_preserves_state_and_validates_before_encoding(self):
        model = random_encoder(ModelConfig(patch_dim=48, embedding_dim=16, encoder_depth=1), 29).train()
        images = torch.zeros(2, 3, 32, 32, dtype=torch.uint8)
        before = {key: value.clone() for key, value in model.state_dict().items()}
        for mode in POOLING:
            self.assertEqual((2, pooling_dimension(16, mode)), tuple(image_embeddings(model, images, pooling=mode).shape))
            self.assertTrue(model.training)
        self.assertTrue(all(torch.equal(value, before[key]) for key, value in model.state_dict().items()))
        with patch.object(model.context_encoder, "forward", side_effect=AssertionError("encoder ran")):
            with self.assertRaises(ValueError): image_embeddings(model, images, pooling="unknown")
