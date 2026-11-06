import unittest
from unittest.mock import patch

import torch

from jepa_lab.analysis.spatial_embeddings import paired_embeddings
from jepa_lab.embeddings import image_embeddings, random_encoder
from jepa_lab.model import ModelConfig


class SpatialEmbeddingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.model = random_encoder(ModelConfig(patch_dim=48, embedding_dim=16, encoder_depth=1), 29).train()
        self.images = (torch.arange(2*3*32*32) % 256).to(torch.uint8).reshape(2, 3, 32, 32)

    def test_one_forward_shared_tokens_exact_mean_and_unchanged_model_rng_and_mode(self):
        before = {k: v.clone() for k, v in self.model.state_dict().items()}
        rng = torch.get_rng_state().clone()
        with patch.object(self.model.context_encoder, 'forward', wraps=self.model.context_encoder.forward) as encode:
            values = paired_embeddings(self.model, self.images, [17, 31])
        self.assertEqual(1, encode.call_count)
        self.assertTrue(torch.equal(values['mean'], image_embeddings(self.model, self.images)))
        self.assertEqual((2, 64), values['quadrants'].shape)
        self.assertTrue(torch.allclose(values['mean'], values['quadrants'].reshape(2, 4, 16).mean(1), atol=1e-6))
        self.assertTrue(self.model.training)
        self.assertFalse(self.model.target_encoder.training)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertTrue(all(torch.equal(v, before[k]) for k, v in self.model.state_dict().items()))
        self.assertTrue(all(not v.requires_grad for v in values.values()))

    def test_bad_inputs_fail_before_encoding_and_exception_restores_mode(self):
        with patch.object(self.model.context_encoder, 'forward', side_effect=AssertionError('encode')):
            for images, indices in ((self.images.float(), [17, 31]), (self.images, [17]),
                                    (self.images[:, :, :16], [17, 31])):
                with self.assertRaises(ValueError):
                    paired_embeddings(self.model, images, indices)
            with self.assertRaisesRegex(AssertionError, 'encode'):
                paired_embeddings(self.model, self.images, [17, 31])
        self.assertTrue(self.model.training)
