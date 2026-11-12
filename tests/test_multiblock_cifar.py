import json
import io
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import torch

from jepa_lab.cifar10 import make_manifest
from jepa_lab.model import ModelConfig
from jepa_lab.variants.cifar_batches import CifarTrainingData
from jepa_lab.variants.multiblock_train import train_multiblock
from test_cifar10_train import Fixture
import test_multiblock_train


class MultiBlockCifarTests(unittest.TestCase):
    assert_state_equal = test_multiblock_train.MultiBlockTrainTests.assert_state_equal
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.path = self.root/'manifest.json'
        self.manifest = make_manifest(Fixture(self.root), Fixture(self.root, train=False))
        self.path.write_text(json.dumps(self.manifest))
        fixture = patch('jepa_lab.variants.cifar_batches.CIFAR10Binary', Fixture)
        fixture.start()
        self.addCleanup(fixture.stop)
        Fixture.accessed.clear()

    def test_manifest_and_split_checked_before_image_reads(self):
        data = CifarTrainingData(self.root, self.path)
        self.assertEqual([], Fixture.accessed)
        train = data.indices[:2]
        validation = self.manifest['partitions']['validation']['indices'][0]
        for invalid in ((train[0], validation), (train[0], train[0]), (True,), ()):
            with self.assertRaisesRegex(ValueError, 'fixed CIFAR-10 training split'):
                data.read(invalid)
        self.assertEqual([], Fixture.accessed)
        images, labels = data.read(train)
        self.assertEqual((2, 3, 32, 32), tuple(images.shape))
        self.assertEqual([index % 10 for index in train], labels.tolist())
        self.assertEqual([(True, index) for index in train], Fixture.accessed)
        self.manifest['partitions']['train']['indices'][0] = validation
        self.path.write_text(json.dumps(self.manifest))
        with self.assertRaisesRegex(ValueError, 'manifest differs'):
            CifarTrainingData(self.root, self.path)
        self.assertEqual(2, len(Fixture.accessed))

    def test_sampling_replays_observed_rows_without_images_or_global_rng_changes(self):
        data = CifarTrainingData(self.root, self.path)
        original = random.getstate()
        historical = random.Random(29)
        history = [dict(train_indices=historical.sample(list(data.indices), 4)) for _ in range(3)]
        replay = data.replay_sampling(history, seed=29, batch_size=4)
        self.assertEqual(historical.getstate(), replay.getstate())
        history[-1]['train_indices'].reverse()
        with self.assertRaisesRegex(ValueError, 'image order'):
            data.replay_sampling(history, seed=29, batch_size=4)
        self.assertEqual(original, random.getstate())
        self.assertEqual([], Fixture.accessed)

    def test_public_training_exact_resume_and_training_split_only(self):
        data = CifarTrainingData(self.root, self.path)
        whole, split = self.root/'whole.pt', self.root/'split.pt'
        config = ModelConfig(patch_dim=48, embedding_dim=16, encoder_depth=1)
        options = dict(training_data=data, config=config, batch_size=2, blocks=4)
        initial_rng = torch.get_rng_state().clone()
        full = train_multiblock(steps=3, checkpoint=whole, **options)
        first = train_multiblock(steps=1, checkpoint=split, **options)
        last = train_multiblock(steps=2, checkpoint=split, resume=split, **options)
        self.assertEqual(full['history'], first['history']+last['history'])
        self.assert_state_equal(torch.load(whole, weights_only=True), torch.load(split, weights_only=True))
        self.assertTrue(torch.equal(initial_rng, torch.get_rng_state()))
        self.assertEqual(data.manifest['manifest_sha256'], full['manifest_sha256'])
        scheduled = [index for row in full['history'] for index in row['train_indices']]
        self.assertEqual([(True, index) for index in scheduled*2], Fixture.accessed)

    def test_public_resume_rejects_changed_manifest_image_order_and_rng_before_sampling(self):
        data = CifarTrainingData(self.root, self.path)
        path = self.root/'cifar.pt'
        options = dict(training_data=data, steps=1, batch_size=2, checkpoint=path)
        train_multiblock(**options)
        original = path.read_bytes()
        Fixture.accessed.clear()
        changed = make_manifest(Fixture(self.root), Fixture(self.root, train=False), seed=31)
        self.path.write_text(json.dumps(changed))
        other = CifarTrainingData(self.root, self.path)
        with self.assertRaisesRegex(ValueError, 'Checkpoint'):
            train_multiblock(**(options | dict(resume=path, training_data=other)))
        for field in ('history', 'data_rng'):
            state = torch.load(io.BytesIO(original), weights_only=True)
            if field == 'history':
                state[field][0]['train_indices'].reverse()
            else:
                state[field] = random.Random(1).getstate()
            torch.save(state, path)
            before = path.read_bytes()
            with self.assertRaisesRegex(ValueError, 'image order'):
                train_multiblock(resume=path, **options)
            self.assertEqual(before, path.read_bytes())
        self.assertEqual([], Fixture.accessed)
