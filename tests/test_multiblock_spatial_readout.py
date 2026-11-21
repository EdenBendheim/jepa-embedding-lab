import unittest
from unittest.mock import patch

import torch

from experiments.multiblock_patch_control import shuffle_patches
from experiments.multiblock_spatial_readout import encoder_views
from jepa_lab.embeddings import image_embeddings, random_encoder
from jepa_lab.model import ModelConfig


class SpatialReadoutTests(unittest.TestCase):
    def test_views_recover_global_mean_without_extra_encoder_passes(self):
        torch.set_num_threads(1)
        model = random_encoder(ModelConfig(patch_dim=48, embedding_dim=16, encoder_depth=1), 29)
        generator = torch.Generator().manual_seed(2)
        images = torch.randint(0, 256, (3, 3, 32, 32), dtype=torch.uint8, generator=generator)
        indices = (2, 7, 10)
        with patch.object(model.context_encoder, 'forward', wraps=model.context_encoder.forward) as forward:
            views = encoder_views(model, images, indices, patch_seed=41)
        self.assertEqual(2, forward.call_count)
        self.assertEqual({'original', 'patch_shuffle'}, views.keys())
        shuffled = shuffle_patches(images, indices, seed=41)
        for view, batch in (('original', images), ('patch_shuffle', shuffled)):
            self.assertTrue(torch.equal(views[view]['mean'], image_embeddings(model, batch)))
            cells = views[view]['quadrants'].reshape(3, 4, 16)
            self.assertTrue(torch.allclose(cells.mean(1), views[view]['mean'], atol=1e-6))

    def test_regional_rgb_retains_fixed_order_and_global_control(self):
        from experiments.multiblock_spatial_readout import rgb_views

        image = torch.zeros(1, 3, 32, 32, dtype=torch.uint8)
        image[:, 0, :16, :16] = 255
        image[:, 1, :16, 16:] = 255
        image[:, 2, 16:, :16] = 255
        rgb = rgb_views(image, (7,), patch_seed=41)
        original = rgb['original']
        expected = torch.tensor([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [0., 0., 0.]])
        cells = original['quadrants'].reshape(4, 6)
        self.assertTrue(torch.equal(expected, cells[:, :3]))
        self.assertEqual(0, torch.count_nonzero(cells[:, 3:]))
        self.assertTrue(torch.equal(original['mean'], rgb['patch_shuffle']['mean']))
        self.assertFalse(torch.equal(original['quadrants'], rgb['patch_shuffle']['quadrants']))
