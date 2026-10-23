import json
from unittest.mock import patch
import unittest

import torch

from jepa_lab import scale_evaluation as workflow
from jepa_lab.exposure import load_exposure_bundle
import test_scale_run as fixtures
from test_budget_driver import Fixture


class ScaleEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        fixtures.ScaleRunTests.setUp(self)
        self.run.training()

    def test_lock_and_observation_precede_fresh_access_and_no_refit_replay_is_identical(self):
        feature = workflow.feature
        def checked(*args, **kwargs):
            if args[3].get('kind') == 'confirmation':
                for name in ('selection-lock.json', 'observed-exposure.json', 'observation-receipt.json'):
                    self.assertTrue((self.directory/name).exists())
            return feature(*args, **kwargs)
        with patch.object(workflow, 'feature', side_effect=checked):
            expected = workflow.evaluation(self.run)
        self.assertEqual(10, expected['verified_linear_states'])
        self.assertEqual(5, expected['verified_neighbor_states'])
        self.assertEqual(2, len(expected['paired_scales']))
        self.assertEqual(.02, expected['paired_scales'][0]['candidate_scale'])
        self.assertEqual(1., expected['paired_scales'][0]['baseline_scale'])
        self.assertAlmostEqual(expected['primary_delta'], expected['paired_scales'][0]['paired_confirmation']['accuracy_delta'])
        before = {p: p.read_bytes() for p in self.directory.rglob('*') if p.is_file()}
        with patch.object(workflow, 'extract_cifar10', side_effect=AssertionError('extraction')), \
                patch.object(workflow, 'compare_scales', side_effect=AssertionError('linear refit')), \
                patch('jepa_lab.neighbor_workflow.select_neighbors', side_effect=AssertionError('neighbor retuning')), \
                patch('jepa_lab.training_budget.train_cifar10', side_effect=AssertionError('training')), \
                patch.object(Fixture, '__getitem__', side_effect=AssertionError('images')):
            self.assertEqual(expected, workflow.evaluation(self.run, reuse_only=True))
        self.assertEqual(before, {p: p.read_bytes() for p in before})
        path = self.directory/'features/scale0.02-seed29-checkpoint-fresh.pt'
        saved = torch.load(path, weights_only=True)
        saved['features'][0, 0] += .1
        torch.save(saved, path)
        with self.assertRaises(ValueError):
            workflow.evaluation(self.run, reuse_only=True)
        self.assertEqual(before[self.directory/'summary.json'], (self.directory/'summary.json').read_bytes())

    def test_interrupted_fresh_attempt_keeps_rows_consumed_and_retries_keep_the_same_lock(self):
        feature = workflow.feature
        def interrupted(*args, **kwargs):
            if args[3].get('kind') == 'confirmation':
                raise OSError('simulated extraction interruption')
            return feature(*args, **kwargs)
        with patch.object(workflow, 'feature', side_effect=interrupted):
            with self.assertRaisesRegex(OSError, 'interruption'):
                workflow.evaluation(self.run)
        self.assertFalse((self.directory/'summary.json').exists())
        d, manifest, ref, fresh, labels = self.run.inputs()
        bundle_path = self.directory/'observed-exposure.json'
        bundle = load_exposure_bundle(bundle_path, manifest, labels)
        expected = set(fresh['exposure_ledger']['observed_validation_indices']) | set(fresh['partitions']['validation']['indices'])
        self.assertEqual(expected, set(bundle['ledger']['observed_validation_indices']))
        before = {p: p.read_bytes() for p in (bundle_path, self.directory/'selection-lock.json', self.directory/'observation-receipt.json')}
        result = workflow.evaluation(self.run)
        self.assertEqual(len(expected), result['exposure']['observed_validation_count'])
        self.assertEqual(before, {p: p.read_bytes() for p in before})

    def test_changed_observation_receipt_refuses_scoring_and_preserves_previous_summary(self):
        workflow.evaluation(self.run)
        summary = (self.directory/'summary.json').read_bytes()
        receipt = self.directory/'observation-receipt.json'
        row = json.loads(receipt.read_text())
        row['selection_lock_sha256'] = '0'*64
        receipt.write_text(json.dumps(row))
        with patch.object(workflow, 'score_confirmation', side_effect=AssertionError('scored before check')):
            with self.assertRaisesRegex(ValueError, 'observation evidence'):
                workflow.evaluation(self.run, reuse_only=True)
        self.assertEqual(summary, (self.directory/'summary.json').read_bytes())
