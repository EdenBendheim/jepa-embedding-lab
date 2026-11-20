from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch

from experiments.multiblock_audit_panel import feature_panel
from experiments.multiblock_audit_sources import AuditSources
from jepa_lab.embeddings import random_encoder
from jepa_lab.model import ModelConfig


class AuditPanelTests(unittest.TestCase):
    def test_panel_reuses_exact_order_and_rejects_changed_cached_features(self):
        torch.set_num_threads(1)
        model = random_encoder(ModelConfig(patch_dim=48, embedding_dim=16, encoder_depth=1), 29)
        indices = (13, 7, 1)
        sources = AuditSources(indices, {'29/random': model}, {'sources': 'fixture'})
        images = torch.randint(0, 256, (14, 3, 32, 32), generator=torch.Generator().manual_seed(8),
                               dtype=torch.uint8)

        class Dataset:
            labels = tuple(range(len(images)))
            read_rows = []

            def read(self, batch):
                self.read_rows.extend(batch)
                return images[list(batch)], torch.tensor(batch)

        data = Dataset()
        with TemporaryDirectory() as directory:
            path = Path(directory)/'panel.pt'
            options = dict(protocol_sha='declared', batch_size=2, shuffle_seed=29)
            with self.assertRaisesRegex(ValueError, 'original fitting'):
                feature_panel(data, sources, path, replay=True, **options)
            before, labels = feature_panel(data, sources, path, **options)
            self.assertEqual(list(indices), data.read_rows)
            self.assertEqual(list(indices), labels.tolist())
            with patch('experiments.multiblock_audit_panel.image_embeddings',
                       side_effect=AssertionError('unexpected encoding')):
                after, _ = feature_panel(data, sources, path, replay=True, **options)
            self.assertEqual(list(indices), data.read_rows)
            self.assertTrue(all(torch.equal(value, after[name]) for name, value in before.items()))
            saved = torch.load(path, weights_only=True)
            saved['features']['rgb'][0, 0] += 1
            torch.save(saved, path)
            with self.assertRaisesRegex(ValueError, 'content changed'):
                feature_panel(data, sources, path, replay=True, **options)
            self.assertEqual(list(indices), data.read_rows)
