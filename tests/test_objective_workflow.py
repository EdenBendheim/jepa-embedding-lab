from copy import deepcopy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import torch

from jepa_lab import objective_features as workflow
from jepa_lab.objective_compare import validate_objective_features
from jepa_lab.probes import ProbeConfig
from jepa_lab.objective_selection import prepare_selection
from jepa_lab.objective_observation import observation_gate
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

    def test_observation_gate_consumes_before_images_and_refuses_tampered_receipts(self):
        state = prepare_selection(self.run)
        d, m, ref, fresh, labels, bundle = self.run.inputs()
        with patch.object(Fixture, '__getitem__', side_effect=AssertionError('image before receipt')):
            exposure = observation_gate(self.run, d, m, fresh, labels, register=True)
        self.assertEqual(30, exposure['observed_validation_count'])
        result = workflow.feature(self.run, d, m, fresh, labels, 'pixels', 'validation')
        self.assertEqual(tuple(fresh['partitions']['validation']['indices']), result.indices)
        receipt = self.directory/'observation-receipt.json'
        row = json.loads(receipt.read_text())
        receipt.write_text(json.dumps(row | dict(selection_lock_sha256='0'*64)))
        with patch.object(workflow, 'extract_cifar10', side_effect=AssertionError('fresh read')):
            with self.assertRaisesRegex(ValueError, 'observation evidence'):
                workflow.feature(self.run, d, m, fresh, labels, 'random-seed29', 'validation')

    def test_interrupted_receipt_write_blocks_access_and_retries_preserve_original_bundle(self):
        import jepa_lab.objective_observation as observation
        prepare_selection(self.run)
        d, m, ref, fresh, labels, bundle = self.run.inputs()
        from jepa_lab.compare import write_report
        def interrupted(path, value):
            if path.name == 'observation-receipt.json':
                raise OSError('receipt interrupted')
            return write_report(path, value)
        with patch.object(observation, 'write_report', side_effect=interrupted):
            with self.assertRaisesRegex(OSError, 'interrupted'):
                observation_gate(self.run, d, m, fresh, labels, register=True)
        bundle_path = self.directory/'observed-exposure.json'
        before = bundle_path.read_bytes()
        with patch.object(workflow, 'extract_cifar10', side_effect=AssertionError('fresh read')):
            with self.assertRaisesRegex(ValueError, 'gate receipt'):
                workflow.feature(self.run, d, m, fresh, labels, 'pixels', 'validation')
        observation_gate(self.run, d, m, fresh, labels, register=True)
        self.assertEqual(before, bundle_path.read_bytes())
        report = next((self.directory/'tuning').glob('*.json'))
        state = json.loads(report.read_text())
        artifact = Path(state['grids'][0]['candidates'][0]['path'])
        artifact.write_bytes(artifact.read_bytes()+b'changed')
        with self.assertRaisesRegex(ValueError, 'candidate artifact'):
            observation_gate(self.run, d, m, fresh, labels)

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

    def test_tuning_fits_shared_sources_once_and_locks_without_observing_fresh_rows(self):
        result = prepare_selection(self.run)
        self.assertEqual(8, result['verified_candidate_states'])
        self.assertEqual(8, len(result['lock']['linear']))
        self.assertEqual(4, len(result['lock']['neighbors']))
        self.assertTrue(result['lock_path'].exists())
        self.assertFalse((self.directory/'observed-exposure.json').exists())
        before = {p: p.read_bytes() for p in self.directory.rglob('*') if p.is_file()}
        with patch('jepa_lab.objective_selection.fit_probe', side_effect=AssertionError('refit')), \
                patch('jepa_lab.objective_selection.fit_ridge', side_effect=AssertionError('refit')), \
                patch('jepa_lab.neighbor_workflow.select_neighbors', side_effect=AssertionError('retune')), \
                patch.object(Fixture, '__getitem__', side_effect=AssertionError('images')):
            cached = prepare_selection(self.run, reuse_only=True)
        self.assertEqual(result['lock'], cached['lock'])
        self.assertEqual(before, {p: p.read_bytes() for p in before})
