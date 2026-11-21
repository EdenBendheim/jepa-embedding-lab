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
