from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch

from experiments.multiblock_audit_sources import AuditSources
from experiments.multiblock_spatial_panel import spatial_panel
from jepa_lab.embeddings import random_encoder
from jepa_lab.model import ModelConfig


class SpatialPanelTests(unittest.TestCase):
    def test_preserves_fitting_order_and_replay_uses_no_image_reader(self):
        torch.set_num_threads(1)
        indices = (7, 1, 4)
        sources = AuditSources(indices, {
            '29/random': random_encoder(ModelConfig(patch_dim=48, embedding_dim=16), 29),
        }, {'fixture': True})
        images = torch.randint(0, 256, (8, 3, 32, 32), dtype=torch.uint8,
                               generator=torch.Generator().manual_seed(7))

        class SyntheticData:
            labels = tuple(range(8))
            reads = []

            def read(self, batch):
                self.reads.extend(batch)
                return images[list(batch)], torch.tensor(batch)

        data = SyntheticData()
        with TemporaryDirectory() as temporary:
            path = Path(temporary)/'panel.pt'
            options = dict(protocol_sha='fixture', batch_size=2, patch_seed=41)
            values, labels = spatial_panel(data, sources, path, **options)
            cached, _ = spatial_panel(data, sources, path, replay=True, **options)
            self.assertEqual(list(indices), data.reads)
            self.assertEqual(list(indices), labels.tolist())
            self.assertEqual((3, 64), values['29/random/original/quadrants'].shape)
            self.assertEqual((3, 24), values['rgb/patch_shuffle/quadrants'].shape)
            self.assertTrue(all(torch.equal(cached[name], value) for name, value in values.items()))
            with self.assertRaisesRegex(ValueError, 'cache identity'):
                spatial_panel(data, sources, path, replay=True, **(options | dict(patch_seed=42)))
            self.assertEqual(list(indices), data.reads)
