import hashlib
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import torch

from jepa_lab.exposure import make_exposure_bundle, make_ledger_confirmation
from jepa_lab.scale_run import ScaleRun
import test_budget_driver as fixtures


class ScaleRunTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        fixtures.BudgetDriverTests.setUp(self)
        d = json.loads(self.driver.PROTOCOL.read_text())
        manifest = json.loads(Path(d['manifest_path']).read_text())
        pilot = json.loads(Path(d['selection_path']).read_text())
        labels = fixtures.Fixture(self.root).labels
        observed = Path(d['previous_confirmation_path'])
        sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
        bundle = make_exposure_bundle(manifest, labels, [(observed, sha(observed))])
        bundle_path = self.root/'bundle.json'
        bundle_path.write_text(json.dumps(bundle))
        fresh = make_ledger_confirmation(manifest, labels, pilot, bundle['ledger'], validation_per_class=1, seed=91)
        d |= dict(exposure_bundle_path=str(bundle_path), ledger_sha256=bundle['ledger']['ledger_sha256'],
            confirmation_selection_sha256=fresh['selection_sha256'], confirmation_seed=91,
            position_scales=[.02, 1.], position_encoding='sincos2d', embedding_dim=16, heads=4,
            encoder_depth=1, predictor_depth=1, reused_scale_one_checkpoints={}, neighbor_counts=[1, 3],
            mask=dict(target_rows=2, target_cols=2, context_fraction=.75))
        d['sources'].append(dict(path=str(bundle_path), sha256=sha(bundle_path)))
        self.driver.PROTOCOL.write_text(json.dumps(d))
        self.run = ScaleRun(self.root, self.directory, protocol=self.driver.PROTOCOL,
                            expected_sha=sha(self.driver.PROTOCOL))
        p = patch('jepa_lab.scale_run.CIFAR10Binary', fixtures.Fixture)
        p.start()
        self.addCleanup(p.stop)

    def test_matched_capacity_and_order_replay_without_training_or_image_reads(self):
        result = self.run.training()
        before = {p: p.read_bytes() for p in self.directory.rglob('*') if p.is_file()}
        low, high = result['scale0.02-seed29'], result['scale1.0-seed29']
        self.assertEqual(low['train_image_order_sha256'], high['train_image_order_sha256'])
        self.assertEqual(low['model_size']['trainable_parameters'], high['model_size']['trainable_parameters'])
        with patch('jepa_lab.training_budget.train_cifar10', side_effect=AssertionError('training')), \
                patch.object(fixtures.Fixture, '__getitem__', side_effect=AssertionError('images')):
            self.assertEqual(result, self.run.training(reuse_only=True))
        self.assertEqual(before, {p: p.read_bytes() for p in before})
        checkpoint = self.directory/'checkpoints/scale1.0-seed29.pt'
        state = torch.load(checkpoint, weights_only=True)
        state['training_history'][0]['train_indices'].reverse()
        torch.save(state, checkpoint)
        with patch('jepa_lab.scale_run.run_training_budget', side_effect=AssertionError('training before preflight')):
            with self.assertRaisesRegex(ValueError, 'image order'):
                self.run.training()

    def test_stale_sources_protected_outputs_and_missing_reuse_fail_before_training(self):
        with patch('jepa_lab.scale_run.run_training_budget', side_effect=AssertionError('training')):
            with self.assertRaisesRegex(ValueError, 'missing'):
                self.run.training(reuse_only=True)
            self.assertFalse(self.directory.exists())
            self.run.directory = self.root
            with self.assertRaisesRegex(ValueError, 'protected'):
                self.run.training()
            self.run.directory = self.directory
            d = json.loads(self.run.protocol.read_text())
            path = Path(d['selection_path'])
            path.write_text(path.read_text()+' ')
            with self.assertRaisesRegex(ValueError, 'source bytes'):
                self.run.training()
            self.assertFalse(self.directory.exists())
