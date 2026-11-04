from copy import deepcopy
import unittest
from unittest.mock import patch

import torch

from jepa_lab.analysis.objective_representations import inspect_training
from test_cifar10_train import Fixture
import test_objective_run as objective_fixture


class ObjectiveRepresentationTests(unittest.TestCase):
    setUp = objective_fixture.ObjectiveRunTests.setUp
    repin = objective_fixture.ObjectiveRunTests.repin

    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_exact_fitting_reads_and_no_changes_to_completed_evidence(self):
        self.run.training()
        _, _, fitting, _, _, _ = self.run.inputs()
        before = {p: p.read_bytes() for p in self.directory.rglob('*') if p.is_file()}
        Fixture.accessed.clear()
        rng = torch.get_rng_state().clone()
        with patch('jepa_lab.analysis.objective_representations.CIFAR10Binary', Fixture), \
                patch('jepa_lab.training_budget.train_cifar10', side_effect=AssertionError('training')):
            result = inspect_training(self.run)
        self.assertEqual([(True, i) for i in fitting['partitions']['train']['indices']], Fixture.accessed)
        self.assertEqual({'prediction-seed29', 'context_variance-seed29', 'random-seed29'}, set(result['arms']))
        for arm in result['arms'].values():
            self.assertEqual(4, len(arm['prediction_controls']))
            self.assertAlmostEqual(arm['crossed_variance']['total_variance'],
                                   sum(arm['crossed_variance']['variance'].values()))
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(before, {p: p.read_bytes() for p in before})

    def test_wrong_split_or_checkpoint_stops_before_image_reads(self):
        self.run.training()
        recipe, manifest, fitting, fresh, labels, bundle = self.run.inputs()
        wrong = deepcopy(fitting)
        wrong['partitions']['train']['indices'][0] = fresh['partitions']['validation']['indices'][0]
        Fixture.accessed.clear()
        with patch.object(self.run, 'training', return_value={}), \
                patch.object(self.run, 'inputs', return_value=(recipe, manifest, wrong, fresh, labels, bundle)), \
                self.assertRaisesRegex(ValueError, 'training split'):
            inspect_training(self.run)
        from jepa_lab.embeddings import load_encoder
        def wrong_checkpoint(*args):
            model, metadata = load_encoder(*args)
            return model, metadata | {'checkpoint_sha256': '0' * 64}
        with patch('jepa_lab.analysis.objective_representations.load_encoder', side_effect=wrong_checkpoint), \
                self.assertRaisesRegex(ValueError, 'checkpoint differs'):
            inspect_training(self.run)
        self.assertEqual([], Fixture.accessed)
