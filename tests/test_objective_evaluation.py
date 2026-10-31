import json
import unittest
from unittest.mock import patch

from jepa_lab import objective_evaluation as evaluation
from jepa_lab import objective_features as features
from jepa_lab.exposure import load_exposure_bundle
import test_objective_workflow as fixtures
from test_cifar10_train import Fixture


class ObjectiveEvaluationTests(unittest.TestCase):
    repin = fixtures.ObjectiveWorkflowTests.repin
    setUpClass = fixtures.ObjectiveWorkflowTests.setUpClass
    setUp = fixtures.ObjectiveWorkflowTests.setUp

    def test_frozen_scoring_lock_order_and_identical_no_fit_no_image_replay(self):
        original = features.feature
        def checked(*args, **kwargs):
            if args[3].get('kind') == 'confirmation':
                for name in ('selection-lock.json', 'observation-receipt.json', 'observed-exposure.json'):
                    self.assertTrue((self.directory/name).exists())
            return original(*args, **kwargs)
        with patch.object(evaluation, 'feature', side_effect=checked):
            result = evaluation.evaluation(self.run)
        self.assertEqual(8, result['verified_candidate_states'])
        self.assertEqual(8, result['verified_linear_states'])
        self.assertEqual(4, result['verified_neighbor_states'])
        self.assertEqual(3, len(result['paired_objectives']))
        self.assertAlmostEqual(result['primary_delta'], result['paired_objectives'][0]['paired_confirmation']['accuracy_delta'])
        self.assertEqual(30, result['exposure']['observed_validation_count'])
        before = {p: p.read_bytes() for p in self.directory.rglob('*') if p.is_file()}
        with (patch.object(features, 'extract_cifar10', side_effect=AssertionError('images')),
                patch('jepa_lab.objective_selection.fit_probe', side_effect=AssertionError('fit')),
                patch('jepa_lab.objective_selection.fit_ridge', side_effect=AssertionError('fit')),
                patch('jepa_lab.neighbor_workflow.select_neighbors', side_effect=AssertionError('retune')),
                patch('jepa_lab.training_budget.train_cifar10', side_effect=AssertionError('train')),
                patch.object(Fixture, '__getitem__', side_effect=AssertionError('images'))):
            self.assertEqual(result, evaluation.evaluation(self.run, reuse_only=True))
        self.assertEqual(before, {p: p.read_bytes() for p in before})

    def test_interrupted_fresh_attempt_retains_exposure_and_immutable_selection(self):
        original = evaluation.feature
        def interrupted(*args, **kwargs):
            if args[3].get('kind') == 'confirmation':
                raise OSError('fresh interrupted')
            return original(*args, **kwargs)
        with patch.object(evaluation, 'feature', side_effect=interrupted):
            with self.assertRaisesRegex(OSError, 'interrupted'):
                evaluation.evaluation(self.run)
        self.assertFalse((self.directory/'summary.json').exists())
        before = {p: p.read_bytes() for p in (self.directory/'selection-lock.json', self.directory/'observation-receipt.json',
                                             self.directory/'observed-exposure.json')}
        result = evaluation.evaluation(self.run)
        self.assertEqual(30, result['exposure']['observed_validation_count'])
        self.assertEqual(before, {p: p.read_bytes() for p in before})

    def test_altered_receipt_or_cached_features_cannot_replace_existing_summary(self):
        import torch
        evaluation.evaluation(self.run)
        summary = (self.directory/'summary.json').read_bytes()
        path = self.directory/'features/context_variance-seed29-fresh.pt'
        saved = torch.load(path, weights_only=True)
        saved['features'][0, 0] += .1
        torch.save(saved, path)
        with self.assertRaisesRegex(ValueError, 'fingerprint'):
            evaluation.evaluation(self.run, reuse_only=True)
        self.assertEqual(summary, (self.directory/'summary.json').read_bytes())
