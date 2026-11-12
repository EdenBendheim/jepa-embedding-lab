import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import torch

from jepa_lab.cifar10 import make_manifest
from jepa_lab.variants.cifar_batches import CifarTrainingData
from test_cifar10_train import Fixture


class MultiBlockCifarTests(unittest.TestCase):
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
