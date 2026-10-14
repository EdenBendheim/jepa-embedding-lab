import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from jepa_lab.cifar10 import make_manifest
from jepa_lab.cifar10_train import train_cifar10
from jepa_lab.model import ModelConfig
from jepa_lab.training_budget import run_training_budget
from test_cifar10_train import Fixture


class TrainingBudgetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def setUp(self):
        d=tempfile.TemporaryDirectory(); self.addCleanup(d.cleanup); self.root=Path(d.name)
        self.manifest=make_manifest(Fixture(self.root), Fixture(self.root, train=False))
        self.path=self.root/'manifest.json'; self.path.write_text(json.dumps(self.manifest))
        self.config=ModelConfig(patch_dim=48, embedding_dim=16, encoder_depth=1)
        for module in ('jepa_lab.cifar10_train', 'jepa_lab.training_budget'):
            p=patch(module+'.CIFAR10Binary', Fixture); p.start(); self.addCleanup(p.stop)
        Fixture.accessed.clear()

    def run_budget(self, path, **changes):
        return run_training_budget(self.root, self.path, path, **(dict(total_steps=5, chunk_steps=2,
            checkpoint_every=1, batch_size=2, config=self.config) | changes))

    def test_chunks_partial_recovery_and_cache_match_one_continuous_run(self):
        expected=train_cifar10(self.root, self.path, steps=5, batch_size=2, config=self.config)
        path=self.root/'budget.pt'
        train_cifar10(self.root, self.path, steps=1, batch_size=2, config=self.config, checkpoint=path)
        result=self.run_budget(path)
        self.assertEqual(expected['history'], result['history']); self.assertEqual(10, result['sample_draws'])
        before=path.read_bytes(); Fixture.accessed.clear()
        with patch('jepa_lab.training_budget.train_cifar10', side_effect=AssertionError('trained cache')):
            cached=self.run_budget(path, reuse_only=True)
        self.assertEqual(result, cached); self.assertEqual(before, path.read_bytes()); self.assertEqual([], Fixture.accessed)

    def test_changed_recipe_budget_or_sample_order_fails_before_training(self):
        path=self.root/'budget.pt'; self.run_budget(path)
        before=path.read_bytes()
        with patch('jepa_lab.training_budget.train_cifar10', side_effect=AssertionError('trained invalid source')):
            for changes in (dict(total_steps=4), dict(seed=29), dict(target_rows=4), dict(batch_size=3),
                            dict(config=ModelConfig(patch_dim=48,embedding_dim=32,encoder_depth=1))):
                with self.assertRaises(ValueError): self.run_budget(path, **changes)
            state=torch.load(path,weights_only=True); state['training_history'][0]['train_indices'].reverse()
            torch.save(state,path)
            with self.assertRaisesRegex(ValueError,'image order'): self.run_budget(path)
        self.assertNotEqual(before,path.read_bytes())

    def test_missing_incomplete_legacy_and_invalid_budget_are_not_silently_reused(self):
        path=self.root/'budget.pt'
        with self.assertRaises(ValueError): self.run_budget(path,reuse_only=True)
        train_cifar10(self.root,self.path,steps=1,batch_size=2,config=self.config,checkpoint=path)
        with self.assertRaises(ValueError): self.run_budget(path,reuse_only=True)
        state=torch.load(path,weights_only=True);state.pop('training_history');state.pop('history_start_step')
        torch.save(state,path)
        with self.assertRaisesRegex(ValueError,'complete journal'):self.run_budget(path)
        for changes in (dict(total_steps=True),dict(total_steps=10001),dict(chunk_steps=1001),dict(checkpoint_every=0)):
            with self.assertRaises(ValueError):self.run_budget(self.root/'new.pt',**changes)
