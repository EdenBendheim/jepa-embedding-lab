import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch

from experiments.multiblock_audit_sources import reopen_sources
from experiments.multiblock_milestones import verify_milestones
from experiments.multiblock_trajectory import run_trajectory
from experiments.run_multiblock_budget_20261118 import PROTOCOL_PATH
from jepa_lab.cifar10 import make_manifest
from jepa_lab.compare import write_report
from jepa_lab.model import ModelConfig
from jepa_lab.objective_run import file_sha
from jepa_lab.selection import make_pilot
from jepa_lab.variants.cifar_batches import CifarTrainingData
from test_cifar10_train import Fixture


class AuditSourceTests(unittest.TestCase):
    def test_reopen_has_no_image_reads_and_refuses_unpublished_training_identity(self):
        torch.set_num_threads(1)
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = make_manifest(Fixture(root), Fixture(root, train=False))
            write_report(root/'manifest.json', manifest)
            with patch('jepa_lab.variants.cifar_batches.CIFAR10Binary', Fixture):
                data = CifarTrainingData(root, root/'manifest.json')
            selection = make_pilot(manifest, data.labels, train_per_class=2, validation_per_class=1)
            config = ModelConfig(patch_dim=48, embedding_dim=16, encoder_depth=1)
            endings = dict(separate=[1], joint=[1, 2])
            receipts = {arm: run_trajectory(data, root/'seed-29'/arm, grouping=arm,
                ends=ends, seed=29, batch_size=2, config=config) for arm, ends in endings.items()}
            _, costs = verify_milestones(data, root/'seed-29', receipts)
            costs['arm_order'] = list(endings)
            published = {'29': costs}
            lock = root/'training-lock.json'
            write_report(lock, dict(protocol_sha256=file_sha(PROTOCOL_PATH), milestones=published))
            budget = dict(config=config.__dict__, schedule=dict(checkpoints=endings),
                          seeds=[29], batch_size=2, orders=[list(endings)])
            Fixture.accessed.clear()
            source = reopen_sources(data, selection, root, budget, published)
            self.assertEqual(20, len(source.indices))
            self.assertEqual({'29/separate', '29/joint_matched', '29/joint_extended', '29/random'},
                             set(source.models))
            self.assertEqual([], Fixture.accessed)
            altered = json.loads(lock.read_text())
            altered['milestones']['29']['shared_updates'] = 7
            write_report(lock, altered)
            with self.assertRaisesRegex(ValueError, 'published experiment'):
                reopen_sources(data, selection, root, budget, published)
            self.assertEqual([], Fixture.accessed)
