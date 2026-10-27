import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

from jepa_lab.model import ModelConfig
from jepa_lab.train import train_smoke


def assert_state_equal(test, left, right):
    if isinstance(left, torch.Tensor):
        test.assertTrue(torch.equal(left, right))
    elif isinstance(left, dict):
        test.assertEqual(left.keys(), right.keys())
        for k in left: assert_state_equal(test, left[k], right[k])
    elif isinstance(left, (tuple, list)):
        test.assertEqual(len(left), len(right))
        for a, b in zip(left, right): assert_state_equal(test, a, b)
    else:
        test.assertEqual(left, right)


class ObjectiveTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def setUp(self):
        d = tempfile.TemporaryDirectory(); self.addCleanup(d.cleanup); self.root = Path(d.name)
        self.options = dict(config=ModelConfig(rows=4, cols=4, embedding_dim=16, encoder_depth=1),
                            batch_size=2, seed=19)

    def test_regularized_resume_matches_continuous_state_and_loss_components(self):
        whole, split = self.root/'whole.pt', self.root/'split.pt'
        options = self.options | dict(context_variance_weight=.1, context_std_floor=.5)
        expected = train_smoke(steps=4, checkpoint=whole, **options)
        first = train_smoke(steps=2, checkpoint=split, **options)
        second = train_smoke(steps=2, resume=split, checkpoint=split, **options)
        self.assertEqual(expected['history'], first['history']+second['history'])
        assert_state_equal(self, torch.load(whole, weights_only=True), torch.load(split, weights_only=True))
        for row in expected['history']:
            self.assertAlmostEqual(row['loss'], row['prediction_loss']+.1*row['context_variance_penalty'], places=6)
        self.assertEqual(.5, expected['objective']['context_std_floor'])

    def test_default_checkpoint_and_journal_keep_historical_identity(self):
        path = self.root/'old.pt'
        expected = train_smoke(steps=2, checkpoint=path, **self.options)
        state = torch.load(path, weights_only=True)
        self.assertNotIn('objective', state)
        self.assertNotIn('objective', expected)
        self.assertEqual({'step', 'loss', 'prediction', 'target'}, set(expected['history'][0]))
        # The unused floor cannot silently turn on a regularizer.
        explicit = train_smoke(steps=2, context_variance_weight=0, context_std_floor=2, **self.options)
        self.assertEqual(expected, explicit)
        train_smoke(steps=1, resume=path, **self.options)

    def test_objective_mismatch_and_single_image_fail_before_model_or_checkpoint_change(self):
        path = self.root/'regularized.pt'
        train_smoke(steps=1, checkpoint=path, context_variance_weight=.1, **self.options)
        before = path.read_bytes()
        with patch('jepa_lab.train.JEPA', side_effect=AssertionError('constructed invalid model')):
            for changes in [dict(), dict(context_variance_weight=.2),
                            dict(context_variance_weight=.1, context_std_floor=2)]:
                with self.assertRaisesRegex(ValueError, 'objective'):
                    train_smoke(steps=1, resume=path, checkpoint=path, **self.options, **changes)
            with self.assertRaisesRegex(ValueError, 'two images'):
                train_smoke(steps=1, context_variance_weight=.1, batch_size=1)
        self.assertEqual(before, path.read_bytes())

    def test_synthetic_cli_reports_the_opt_in_objective(self):
        result = subprocess.run([sys.executable, '-m', 'jepa_lab.train', '--steps', '2',
                                 '--context-variance-weight', '.1', '--context-std-floor', '.5'],
                                text=True, capture_output=True, check=True)
        report = json.loads(result.stdout)
        self.assertEqual(.1, report['objective']['context_variance_weight'])
        self.assertEqual(2, len(report['history']))
        self.assertIn('context_std_mean', report['history'][0])
