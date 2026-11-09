import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from jepa_lab.model import ModelConfig
from jepa_lab.variants.multiblock_train import main, train_multiblock


class MultiBlockTrainTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        self.options = dict(batch_size=2, seed=29,
            config=ModelConfig(rows=4, cols=4, embedding_dim=16, encoder_depth=1))

    def assert_state_equal(self, left, right):
        if isinstance(left, torch.Tensor):
            self.assertTrue(torch.equal(left, right))
        elif isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys())
            for key in left:
                self.assert_state_equal(left[key], right[key])
        elif isinstance(left, (tuple, list)):
            self.assertEqual(len(left), len(right))
            for x, y in zip(left, right):
                self.assert_state_equal(x, y)
        else:
            self.assertEqual(left, right)

    def test_continuous_and_resumed_trajectories_match_all_states_and_preserve_caller_rng(self):
        whole, split = self.root/'whole.pt', self.root/'split.pt'
        rng = torch.get_rng_state().clone()
        full = train_multiblock(steps=4, checkpoint=whole, **self.options)
        first = train_multiblock(steps=2, checkpoint=split, **self.options)
        second = train_multiblock(steps=2, resume=split, checkpoint=split, **self.options)
        self.assertEqual(full['history'], first['history']+second['history'])
        self.assert_state_equal(torch.load(whole, weights_only=True), torch.load(split, weights_only=True))
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(4, second['completed_step'])
        self.assertTrue(all(len(row['block_losses']) == 2 for row in full['history']))

    def test_changed_recipe_runtime_source_or_journal_fails_before_model_allocation(self):
        path = self.root/'model.pt'
        train_multiblock(steps=2, checkpoint=path, **self.options)
        original = path.read_bytes()
        with patch('jepa_lab.variants.multiblock_train.JEPA', side_effect=AssertionError('allocation')):
            for change in (dict(blocks=1), dict(context_fraction=.5), dict(momentum=.5), dict(seed=31)):
                with self.assertRaisesRegex(ValueError, 'Checkpoint'):
                    train_multiblock(steps=1, resume=path, checkpoint=path, **(self.options | change))
            for key in ('runtime', 'code_sha256', 'history'):
                state = torch.load(io.BytesIO(original), weights_only=True)
                if key == 'history':
                    state[key][0]['mask_sha256'] = '0'*64
                else:
                    state[key] = {}
                torch.save(state, path)
                with self.assertRaises(ValueError):
                    train_multiblock(steps=1, resume=path, checkpoint=path, **self.options)
        path.write_bytes(original)
        self.assertEqual(original, path.read_bytes())

    def test_bounds_collision_and_failed_checkpoint_write_preserve_original(self):
        with patch('jepa_lab.variants.multiblock_train.JEPA', side_effect=AssertionError('allocation')):
            for change in (dict(steps=True), dict(steps=101), dict(batch_size=9), dict(blocks=4),
                           dict(learning_rate=True), dict(momentum=-.1), dict(seed=True)):
                with self.assertRaises(ValueError):
                    train_multiblock(**(self.options | dict(steps=1) | change))
        path = self.root/'model.pt'
        train_multiblock(steps=1, checkpoint=path, **self.options)
        before = path.read_bytes()
        with self.assertRaisesRegex(ValueError, 'already exists'):
            train_multiblock(steps=1, checkpoint=path, **self.options)
        with patch('jepa_lab.cifar10_train.os.replace', side_effect=OSError('disk error')), \
                self.assertRaisesRegex(OSError, 'disk error'):
            train_multiblock(steps=1, resume=path, checkpoint=path, **self.options)
        self.assertEqual(before, path.read_bytes())
        self.assertEqual([path], list(self.root.iterdir()))

    def test_changed_optimizer_rate_is_rejected_before_synthetic_sampling(self):
        path = self.root/'model.pt'
        train_multiblock(steps=1, checkpoint=path, **self.options)
        state = torch.load(path, weights_only=True)
        state['optimizer']['param_groups'][0]['lr'] *= 2
        torch.save(state, path)
        before = path.read_bytes()
        with patch('jepa_lab.variants.multiblock_train.synthetic_batch', side_effect=AssertionError('sampling')), \
                self.assertRaisesRegex(ValueError, 'optimizer'):
            train_multiblock(steps=1, resume=path, checkpoint=path, **self.options)
        self.assertEqual(before, path.read_bytes())

    def test_cli_resumes_and_refuses_metrics_overwriting_checkpoint(self):
        path, output = self.root/'cli.pt', self.root/'metrics.json'
        args = ['multi-block', '--steps', '2', '--batch-size', '2', '--checkpoint', str(path), '--output', str(output)]
        with patch('sys.argv', args), contextlib.redirect_stdout(io.StringIO()):
            main()
        with patch('sys.argv', args+['--resume', str(path)]), contextlib.redirect_stdout(io.StringIO()):
            main()
        self.assertEqual(4, json.loads(output.read_text())['completed_step'])
        before = path.read_bytes()
        args[-1] = str(path)
        with patch('sys.argv', args+['--resume', str(path)]), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main()
        self.assertEqual(before, path.read_bytes())
