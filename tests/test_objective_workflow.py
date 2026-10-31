from copy import deepcopy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import torch

from jepa_lab import objective_features as workflow
from jepa_lab.objective_compare import validate_objective_features
from jepa_lab.probes import ProbeConfig
import test_objective_run as fixtures
from test_cifar10_train import Fixture


class ObjectiveWorkflowTests(unittest.TestCase):
    repin = fixtures.ObjectiveRunTests.repin

    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        fixtures.ObjectiveRunTests.setUp(self)
        self.d['evaluation'] = dict(pooling='mean', encoder='context', classifier_seeds=[29],
            probe_steps=3, learning_rates=[.001], weight_decays=[.01], ridge_alphas=[.1], neighbor_counts=[1, 3])
        self.repin()
        self.run.training()
        p = patch('jepa_lab.embeddings.CIFAR10Binary', Fixture)
        p.start()
        self.addCleanup(p.stop)

    def pairs(self, *, reuse_only=False):
        d, m, ref, fresh, labels, bundle = self.run.inputs()
        return {key: tuple(workflow.feature(self.run, d, m, ref, labels, key, partition, reuse_only=reuse_only)
            for partition in ('train', 'validation')) for key in workflow.source_specs(d)}

    def test_unique_shared_controls_and_feature_replay_without_images(self):
        with patch.object(workflow, 'extract_cifar10', wraps=workflow.extract_cifar10) as extraction:
            pairs = self.pairs()
        self.assertEqual(8, extraction.call_count)  # four unique sources, two partitions
        validate_objective_features(workflow.seed_sources(pairs, 29), (ProbeConfig(steps=3),))
        before = {p: p.read_bytes() for p in self.directory.rglob('*') if p.is_file()}
        with patch.object(workflow, 'extract_cifar10', side_effect=AssertionError('extraction')), \
                patch.object(Fixture, '__getitem__', side_effect=AssertionError('images')):
            again = self.pairs(reuse_only=True)
        self.assertEqual(pairs.keys(), again.keys())
        self.assertEqual(before, {p: p.read_bytes() for p in before})

    def test_wrong_objective_and_content_fail_and_fresh_access_is_blocked(self):
        self.pairs()
        path = self.directory / 'features/context_variance-seed29-train.pt'
        original = path.read_bytes()
        for mutation in ('objective', 'content'):
            state = torch.load(path, weights_only=True)
            if mutation == 'objective': state['feature_metadata']['training_settings'].pop('objective')
            else: state['features'][0, 0] += .1
            torch.save(state, path)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.pairs(reuse_only=True)
            path.write_bytes(original)
        d, m, ref, fresh, labels, bundle = self.run.inputs()
        with patch.object(workflow, 'extract_cifar10', side_effect=AssertionError('fresh images')):
            with self.assertRaisesRegex(ValueError, 'observation gate'):
                workflow.feature(self.run, d, m, fresh, labels, 'pixels', 'validation')
        self.assertFalse((self.directory / 'features/pixels-fresh.pt').exists())
