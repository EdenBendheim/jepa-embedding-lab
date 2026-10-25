import json
from pathlib import Path
import unittest
import tempfile
from unittest.mock import patch

import torch

import test_scale_run as fixtures
from test_budget_driver import Fixture
from jepa_lab import stability_audit


class StabilityAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): fixtures.ScaleRunTests.setUpClass()

    def setUp(self):
        fixtures.ScaleRunTests.setUp(self)
        self.run.training()
        cache = tempfile.TemporaryDirectory(); self.addCleanup(cache.cleanup)
        self.cache = Path(cache.name)
        p = patch('jepa_lab.stability_audit.CIFAR10Binary', Fixture)
        p.start(); self.addCleanup(p.stop)

    def test_only_original_training_images_and_no_state_mutation(self):
        before = {p: p.read_bytes() for p in self.directory.rglob('*') if p.is_file()}
        rng = torch.get_rng_state().clone(); Fixture.accessed.clear()
        with patch('jepa_lab.training_budget.train_cifar10', side_effect=AssertionError('training')):
            result = stability_audit.inspect_training(self.run, batch_size=4)
        d = json.loads(self.run.protocol.read_text())
        pilot = json.loads(Path(d['selection_path']).read_text())
        self.assertEqual([(True, i) for i in pilot['partitions']['train']['indices']], Fixture.accessed)
        self.assertEqual(10, result['count']); self.assertEqual(2, len(result['arms']))
        self.assertFalse(result['validation_images_read']); self.assertFalse(result['test_evaluated'])
        self.assertNotIn('train_indices', json.dumps(result)); self.assertNotIn('features', result)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(before, {p: p.read_bytes() for p in before})
        for arm in result['arms'].values():
            self.assertEqual(0., arm['random_context']['between_image_population_variance'] -
                             arm['random_target']['between_image_population_variance'])
            self.assertTrue(arm['parameter_lag']['buffers_identical'])

    def test_invalid_bounds_or_changed_weights_fail_before_images(self):
        with patch.object(Fixture, '__getitem__', side_effect=AssertionError('images')):
            for batch in (0, 33, True):
                with self.assertRaises(ValueError): stability_audit.inspect_training(self.run, batch_size=batch)
            path = self.directory/'checkpoints/scale0.02-seed29.pt'
            saved = torch.load(path, weights_only=True)
            saved['model']['context_encoder.projection.weight'][0, 0] += 1
            torch.save(saved, path)
            with self.assertRaises(ValueError): stability_audit.inspect_training(self.run)

    def test_cache_replay_no_images_or_diagnostics_and_corruption_preserves_bytes(self):
        path = self.cache/'audit-cache.json'
        result, reused = stability_audit.cached_audit(self.run, path)
        self.assertFalse(reused); before = path.read_bytes()
        with patch.object(Fixture, '__getitem__', side_effect=AssertionError('image access')), \
                patch.object(stability_audit, 'inspect_training', side_effect=AssertionError('recompute')):
            self.assertEqual((result, True), stability_audit.cached_audit(self.run, path, reuse_only=True))
            with self.assertRaisesRegex(ValueError, 'missing'):
                stability_audit.cached_audit(self.run, self.cache/'missing.json', reuse_only=True)
            with patch.object(stability_audit, '_code_fingerprints', return_value={'changed': 'code'}):
                with self.assertRaisesRegex(ValueError, 'differs'):
                    stability_audit.cached_audit(self.run, path)
        self.assertEqual(before, path.read_bytes())
        saved = json.loads(before); saved['report']['count'] += 1
        path.write_text(json.dumps(saved)); corrupt = path.read_bytes()
        with self.assertRaisesRegex(ValueError, 'differs'):
            stability_audit.cached_audit(self.run, path)
        self.assertEqual(corrupt, path.read_bytes())

    def test_protected_evidence_and_failed_atomic_save(self):
        d = json.loads(self.run.protocol.read_text())
        for path in (self.run.protocol, Path(d['selection_path']), self.directory/'cache.json', self.root/'data.json'):
            with self.assertRaisesRegex(ValueError, 'cannot replace'):
                stability_audit.cached_audit(self.run, path)
        path = self.cache/'failed.json'
        with patch.object(stability_audit, 'inspect_training', return_value={'count': 10}), \
                patch('jepa_lab.compare.os.replace', side_effect=OSError('disk error')):
            with self.assertRaisesRegex(OSError, 'disk error'):
                stability_audit.cached_audit(self.run, path)
        self.assertFalse(path.exists()); self.assertEqual([], list(self.cache.iterdir()))
