from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from jepa_lab.cifar10 import make_manifest
from jepa_lab.confirmation import score_confirmation
from jepa_lab.embeddings import extract_cifar10
from jepa_lab.features import load_features
from jepa_lab.probe_artifacts import save_probe
from jepa_lab.probes import fit_probe, ProbeConfig
from jepa_lab.ridge import fit_ridge, RidgeConfig
from jepa_lab.selection import make_confirmation, make_pilot
from test_embeddings import Fixture


class ConfirmationScoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def setUp(self):
        folder=tempfile.TemporaryDirectory(); self.addCleanup(folder.cleanup)
        self.root=Path(folder.name)
        self.labels=Fixture(self.root).labels
        self.manifest=make_manifest(Fixture(self.root),Fixture(self.root,train=False))
        self.pilot=make_pilot(self.manifest,self.labels,train_per_class=1,validation_per_class=1)
        self.selection=make_confirmation(self.manifest,self.labels,self.pilot,validation_per_class=2)
        self.m,self.p,self.c=[self.root/f'{name}.json' for name in ('manifest','pilot','confirmation')]
        for path,doc in ((self.m,self.manifest),(self.p,self.pilot),(self.c,self.selection)):
            path.write_text(json.dumps(doc))
        self.paths=[self.root/f'{name}.pt' for name in ('train','validation','fresh')]
        self.sets=[]
        with patch('jepa_lab.embeddings.CIFAR10Binary',Fixture):
            for path,selection,selection_path,partition in ((self.paths[0],self.pilot,self.p,'train'),
                                                          (self.paths[1],self.pilot,self.p,'validation'),
                                                          (self.paths[2],self.selection,self.c,'validation')):
                extract_cifar10(self.root,self.m,representation='pixels',selection_path=selection_path,partition=partition,output=path)
                self.sets.append(load_features(path,manifest=self.manifest,selection=selection,training_labels=self.labels,partition=partition))

    def test_adamw_and_ridge_use_unchanged_state_normalization_and_sources(self):
        train,old,new=self.sets
        for family,probe in (('adamw',fit_probe(train.features,train.labels,ProbeConfig(steps=2))),
                             ('ridge',fit_ridge(train.features,train.labels,RidgeConfig(alpha=1)))):
            path=self.root/f'{family}.pt'; save_probe(path,probe,train,old)
            before={p:p.read_bytes() for p in [path,*self.paths]}
            rng=torch.get_rng_state().clone(); Fixture.accessed.clear()
            with patch('jepa_lab.probes.fit_probe',side_effect=AssertionError('refit')):
                report=score_confirmation(path,train,old,new,self.selection)
            self.assertEqual(probe.score(new.features,new.labels),report['confirmation'])
            self.assertFalse(report['normalization_refitted']); self.assertFalse(report['classifier_refitted'])
            self.assertFalse(report['settings_selected']); self.assertFalse(report['test_evaluated'])
            self.assertEqual([],Fixture.accessed)
            self.assertTrue(torch.equal(rng,torch.get_rng_state()))
            self.assertEqual(before,{p:p.read_bytes() for p in before})
            changed=replace(new,features=new.features+0.25)
            with self.assertRaisesRegex(ValueError,'changed after loading'):
                score_confirmation(path,train,old,changed,self.selection)
            with self.assertRaisesRegex(ValueError,'changed after loading'):
                score_confirmation(path,train,old,replace(new,labels=new.labels.roll(1)),self.selection)
            self.assertEqual(before[path],path.read_bytes())

    def test_changed_representation_sources_indices_or_protocol_fail_before_scoring(self):
        train,old,new=self.sets
        invalid=[replace(new,metadata=new.metadata|dict(pooling='changed')),
                 replace(new,manifest_sha256='e'*64),replace(new,selection_sha256='e'*64),
                 replace(new,selection_kind='pilot'),replace(new,partition='train'),
                 replace(new,indices=old.indices),replace(new,features=new.features[:,:4]),
                 replace(new,features=torch.full_like(new.features,float('nan')))]
        with patch('jepa_lab.confirmation.load_probe',side_effect=AssertionError('scored invalid sources')):
            for value in invalid:
                with self.subTest(value=value.selection_kind),self.assertRaises(ValueError):
                    score_confirmation(self.root/'unused.pt',train,old,value,self.selection)
            for doc in (self.selection|dict(seed=0),self.selection|dict(version=2.0),{}):
                with self.assertRaises(ValueError): score_confirmation(self.root/'unused.pt',train,old,new,doc)
