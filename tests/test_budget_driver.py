import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from jepa_lab.cifar10 import make_manifest
from jepa_lab.model import ModelConfig
from jepa_lab.selection import make_confirmation,make_pilot
from test_cifar10_train import Fixture
from dataclasses import asdict


class BudgetDriverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def setUp(self):
        d=tempfile.TemporaryDirectory();self.addCleanup(d.cleanup);self.root=Path(d.name)
        path=Path(__file__).resolve().parents[1]/'experiments/run_budget_20261014.py'
        spec=importlib.util.spec_from_file_location('budget_driver',path)
        self.driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(self.driver)
        m=make_manifest(Fixture(self.root),Fixture(self.root,train=False));labels=Fixture(self.root).labels
        pilot=make_pilot(m,labels,train_per_class=1,validation_per_class=1)
        old=make_confirmation(m,labels,pilot,validation_per_class=1)
        fresh=make_confirmation(m,labels,pilot,exclusions=(old,),validation_per_class=1,seed=20261014)
        sources=[]
        for name,doc in (('manifest',m),('pilot',pilot),('previous',old)):
            path=self.root/f'{name}.json';path.write_text(json.dumps(doc))
            sources.append(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        declared=dict(sources=sources,manifest_path=sources[0]['path'],selection_path=sources[1]['path'],
            previous_confirmation_path=sources[2]['path'],manifest_sha256=m['manifest_sha256'],selection_sha256=pilot['selection_sha256'],
            confirmation_selection_sha256=fresh['selection_sha256'],confirmation_seed=20261014,confirmation_per_class=1,
            total_steps=3,chunk_steps=2,checkpoint_every=1,batch_size=2,learning_rate=.001,momentum=.99,
            training_seeds=[29],random_seeds=[31],target_sizes=[2,4],context_fraction=.75,
            config=asdict(ModelConfig(patch_dim=48,embedding_dim=16,encoder_depth=1)))
        protocol=self.root/'protocol.json';protocol.write_text(json.dumps(declared))
        self.driver.PROTOCOL=protocol;self.driver.PROTOCOL_SHA=hashlib.sha256(protocol.read_bytes()).hexdigest();self.driver.ROOT=self.root
        for target in (self.driver,'jepa_lab.training_budget','jepa_lab.cifar10_train'):
            p=patch.object(target,'CIFAR10Binary',Fixture) if not isinstance(target,str) else patch(target+'.CIFAR10Binary',Fixture)
            p.start();self.addCleanup(p.stop)
        self.directory=self.root/'run'

    def test_repeated_masks_share_actual_order_and_completed_retry_never_trains(self):
        expected=self.driver.training(self.directory)
        self.assertEqual(expected['mask2-seed29']['history'][0]['train_indices'],expected['mask4-seed29']['history'][0]['train_indices'])
        before={p:p.read_bytes() for p in self.directory.rglob('*') if p.is_file()}
        with patch('jepa_lab.training_budget.train_cifar10',side_effect=AssertionError('trained cache')),patch.object(Fixture,'__getitem__',side_effect=AssertionError('images')):
            self.assertEqual(expected,self.driver.training(self.directory,reuse_only=True))
        self.assertEqual(before,{p:p.read_bytes() for p in before})
        checkpoint=self.driver.checkpoint_path(self.directory,4,29)
        state=torch.load(checkpoint,weights_only=True);state['training_history'][0]['train_indices'].reverse();torch.save(state,checkpoint)
        with patch.object(self.driver,'run_training_budget',side_effect=AssertionError('started before preflight')):
            with self.assertRaisesRegex(ValueError,'image order'):self.driver.training(self.directory)
