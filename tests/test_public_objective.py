from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from jepa_lab import cifar10_train, training_budget
from jepa_lab.cifar10 import make_manifest
from jepa_lab.model import ModelConfig
from test_cifar10_train import Fixture
from test_objective_training import assert_state_equal


class PublicObjectiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def setUp(self):
        d=tempfile.TemporaryDirectory(); self.addCleanup(d.cleanup); self.root=Path(d.name)
        self.manifest=make_manifest(Fixture(self.root), Fixture(self.root, train=False))
        self.path=self.root/'manifest.json'; self.path.write_text(json.dumps(self.manifest))
        self.options=dict(batch_size=2, config=ModelConfig(patch_dim=48, embedding_dim=16, encoder_depth=1),
                          context_variance_weight=.1, context_std_floor=.5)
        for module in ('jepa_lab.cifar10_train', 'jepa_lab.training_budget'):
            p=patch(module+'.CIFAR10Binary', Fixture); p.start(); self.addCleanup(p.stop)
        Fixture.accessed.clear()

    def train(self, **changes):
        return cifar10_train.train_cifar10(self.root, self.path, **(self.options|changes))

    def budget(self, checkpoint, **changes):
        return training_budget.run_training_budget(self.root, self.path, checkpoint,
            **(self.options|dict(total_steps=4, chunk_steps=2, checkpoint_every=1)|changes))

    def test_public_resume_reproduces_full_state_and_reads_training_only(self):
        whole, split=self.root/'whole.pt', self.root/'split.pt'
        expected=self.train(steps=4, checkpoint=whole)
        first=self.train(steps=2, checkpoint=split)
        second=self.train(steps=2, resume=split, checkpoint=split)
        self.assertEqual(expected['history'], first['history']+second['history'])
        assert_state_equal(self, torch.load(whole, weights_only=True), torch.load(split, weights_only=True))
        allowed=set(self.manifest['partitions']['train']['indices'])
        self.assertTrue(all(train and index in allowed for train,index in Fixture.accessed))
        self.assertEqual(.1, expected['settings']['objective']['context_variance_weight'])
        self.assertIn('prediction_loss', expected['history'][0])

    def test_changed_objective_or_invalid_batch_preserves_checkpoint_before_sampling(self):
        path=self.root/'state.pt'; self.train(steps=1, checkpoint=path)
        before=path.read_bytes()
        for change in [dict(context_variance_weight=0), dict(context_variance_weight=.2),
                       dict(context_std_floor=1), dict(context_variance_weight=float('nan')),
                       dict(batch_size=1)]:
            Fixture.accessed.clear()
            with self.assertRaises(ValueError): self.train(steps=1, checkpoint=path, resume=path, **change)
            self.assertEqual([], Fixture.accessed); self.assertEqual(before, path.read_bytes())

    def test_chunked_objective_and_cache_match_continuous_and_refuse_new_recipe(self):
        expected=self.train(steps=4)
        path=self.root/'budget.pt'; result=self.budget(path)
        self.assertEqual(expected['history'], result['history'])
        before=path.read_bytes(); Fixture.accessed.clear()
        with patch('jepa_lab.training_budget.train_cifar10', side_effect=AssertionError('trained cache')):
            self.assertEqual(result, self.budget(path, reuse_only=True))
            with self.assertRaises(ValueError): self.budget(path, context_variance_weight=0)
        self.assertEqual([], Fixture.accessed); self.assertEqual(before, path.read_bytes())

    def test_public_and_budget_clis_expose_and_bind_objective_options(self):
        for module in [cifar10_train, training_budget]:
            path=self.root/(module.__name__+'.pt')
            args=['train', str(self.root), '--manifest', str(self.path), '--checkpoint', str(path),
                  '--context-variance-weight', '.1', '--context-std-floor', '.5', '--batch-size', '2']
            args += ['--steps', '2'] if module is cifar10_train else ['--total-steps', '2']
            out=io.StringIO()
            with patch('sys.argv', args), redirect_stdout(out): module.main()
            report=json.loads(out.getvalue())
            self.assertEqual(.5, report['settings']['objective']['context_std_floor'])
            self.assertEqual(2, torch.load(path, weights_only=True)['step'])
