import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch

from experiments.multiblock_frozen import prepare_heads, register_validation, score_frozen
from jepa_lab.cifar10 import make_manifest
from jepa_lab.compare import write_report
from jepa_lab.embeddings import random_encoder
from jepa_lab.exposure import make_exposure_bundle
from jepa_lab.model import ModelConfig
from jepa_lab.objective_run import file_sha
from jepa_lab.selection import make_pilot
from jepa_lab.variants.cifar_batches import CifarTrainingData
from test_cifar10_train import Fixture


class FrozenBudgetEvaluationTests(unittest.TestCase):
    def test_fitting_and_fresh_access_are_disjoint_and_heads_replay_without_refitting(self):
        torch.set_num_threads(1)
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = make_manifest(Fixture(root), Fixture(root, train=False))
            write_report(root/'manifest.json', manifest)
            with patch('jepa_lab.variants.cifar_batches.CIFAR10Binary', Fixture):
                data = CifarTrainingData(root, root/'manifest.json')
            pilot = make_pilot(manifest, data.labels, train_per_class=2, validation_per_class=1)
            write_report(root/'pilot.json', pilot)
            exposure = root/'previous.json'
            write_report(exposure, make_exposure_bundle(manifest, data.labels,
                [(root/'pilot.json', file_sha(root/'pilot.json'))]))
            folder = root/'new'
            model = random_encoder(ModelConfig(patch_dim=48, embedding_dim=16), 29)
            models = {'29/random': model}
            options = dict(identity='frozen experiment', alpha=.1)
            Fixture.accessed.clear()
            with self.assertRaisesRegex(ValueError, 'Lock fitted'):
                register_validation(data, folder, pilot, exposure, per_class=1, seed=19)
            self.assertEqual([], Fixture.accessed)
            heads = prepare_heads(data, folder, pilot, models, replay=False, **options)
            fitting = [(True, i) for i in pilot['partitions']['train']['indices']]
            self.assertEqual(fitting, Fixture.accessed)
            fresh, aggregate = register_validation(data, folder, pilot, exposure, per_class=1, seed=19)
            self.assertTrue(set(fresh['partitions']['validation']['indices']).isdisjoint(
                pilot['partitions']['validation']['indices']))
            self.assertEqual(20, aggregate['observed_validation_count'])
            self.assertEqual(fitting, Fixture.accessed)
            scores, _, _ = score_frozen(data, folder, fresh, models, heads)
            self.assertEqual(10, scores['29/random']['count'])
            scoring = [(True, i) for i in fresh['partitions']['validation']['indices']]
            self.assertEqual(fitting+scoring, Fixture.accessed)
            with patch('experiments.multiblock_frozen.fit_ridge', side_effect=AssertionError('refit')):
                replayed = prepare_heads(data, folder, pilot, models, replay=True, **options)
            self.assertEqual(fitting+scoring, Fixture.accessed)
            again, _, _ = score_frozen(data, folder, fresh, models, replayed)
            self.assertEqual(scores, again)
            receipt = folder/'observation-receipt.json'
            write_report(receipt, json.loads(receipt.read_text()) | {'heads_lock_sha256': 'changed'})
            with self.assertRaisesRegex(ValueError, 'matching frozen-head'):
                score_frozen(data, folder, fresh, models, heads)
            self.assertEqual(fitting+scoring*2, Fixture.accessed)
