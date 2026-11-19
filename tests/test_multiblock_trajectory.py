import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch

from experiments.multiblock_milestones import verify_milestones
from experiments.multiblock_trajectory import checkpoint_path, run_trajectory
from jepa_lab.cifar10 import make_manifest
from jepa_lab.model import ModelConfig
from jepa_lab.variants.cifar_batches import CifarTrainingData
from test_cifar10_train import Fixture


class TrajectoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        manifest = self.root/'manifest.json'
        manifest.write_text(json.dumps(make_manifest(Fixture(self.root), Fixture(self.root, train=False))))
        with patch('jepa_lab.variants.cifar_batches.CIFAR10Binary', Fixture):
            self.data = CifarTrainingData(self.root, manifest)
        Fixture.accessed.clear()
        self.options = dict(grouping='joint', ends=[1, 3], seed=29, batch_size=2,
                            config=ModelConfig(patch_dim=48, embedding_dim=16, encoder_depth=1))

    def test_interruption_keeps_finished_chunk_and_replays_without_images(self):
        from experiments import multiblock_trajectory as runner

        real_train = runner.train_multiblock
        with patch.object(runner, 'train_multiblock') as train:
            count = 0

            def interrupt(**kwargs):
                nonlocal count
                count += 1
                if count == 2:
                    raise RuntimeError('interrupted')
                return real_train(**kwargs)

            train.side_effect = interrupt
            with self.assertRaisesRegex(RuntimeError, 'interrupted'):
                run_trajectory(self.data, self.root, **self.options)
        first = checkpoint_path(self.root, 1)
        original = first.read_bytes()
        self.assertFalse(checkpoint_path(self.root, 3).exists())
        receipts = run_trajectory(self.data, self.root, **self.options)
        self.assertEqual(original, first.read_bytes())
        self.assertEqual(6, len(Fixture.accessed))
        with patch.object(runner, 'train_multiblock', side_effect=AssertionError('unexpected training')):
            self.assertEqual(receipts, run_trajectory(self.data, self.root, replay=True, **self.options))
        self.assertEqual(6, len(Fixture.accessed))
        first.write_bytes(b'edited')
        with self.assertRaisesRegex(ValueError, 'differs'):
            run_trajectory(self.data, self.root, replay=True, **self.options)
        self.assertEqual(6, len(Fixture.accessed))

    def test_milestones_check_shared_prefix_and_report_realized_costs(self):
        options = self.options | {'ends': [2]}
        separate = run_trajectory(self.data, self.root/'separate', **(options | {'grouping': 'separate'}))
        joint = run_trajectory(self.data, self.root/'joint', **(options | {'ends': [2, 3]}))
        before = list(Fixture.accessed)
        models, report = verify_milestones(self.data, self.root, dict(separate=separate, joint=joint))
        self.assertEqual({'separate', 'joint_matched', 'joint_extended'}, models.keys())
        self.assertEqual(2, report['shared_updates'])
        costs = report['milestones']
        self.assertEqual(costs['separate']['image_order_sha256'], costs['joint_matched']['image_order_sha256'])
        self.assertEqual(costs['separate']['mask_order_sha256'], costs['joint_matched']['mask_order_sha256'])
        self.assertEqual(3, costs['joint_extended']['updates'])
        self.assertGreater(costs['joint_extended']['loop_seconds'], costs['joint_matched']['loop_seconds'])
        self.assertEqual(before, Fixture.accessed)
        wrong = self.root/'wrong'
        other = run_trajectory(self.data, wrong/'joint', **(options | {'seed': 31}))
        (wrong/'separate').symlink_to(self.root/'separate', target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'share initialization'):
            verify_milestones(self.data, wrong, dict(separate=separate, joint=other))
