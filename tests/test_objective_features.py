from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from jepa_lab.cifar10 import make_manifest
from jepa_lab.cifar10_train import train_cifar10
from jepa_lab.embeddings import extract_cifar10
from jepa_lab.features import load_features
from jepa_lab.model import ModelConfig
from jepa_lab.objective import ObjectiveConfig, objective_from_settings
from jepa_lab.selection import make_pilot
from test_cifar10_train import Fixture


class ObjectiveFeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def setUp(self):
        d=tempfile.TemporaryDirectory(); self.addCleanup(d.cleanup); self.root=Path(d.name)
        self.labels=Fixture(self.root).labels
        self.manifest=make_manifest(Fixture(self.root), Fixture(self.root, train=False))
        self.selection=make_pilot(self.manifest, self.labels, train_per_class=1, validation_per_class=1)
        self.m,self.s=self.root/'manifest.json',self.root/'selection.json'
        self.m.write_text(json.dumps(self.manifest));self.s.write_text(json.dumps(self.selection))
        self.checkpoint,self.features=self.root/'model.pt',self.root/'features.pt'
        for module in ['jepa_lab.cifar10_train','jepa_lab.embeddings']:
            p=patch(module+'.CIFAR10Binary',Fixture);p.start();self.addCleanup(p.stop)
        train_cifar10(self.root,self.m,steps=2,batch_size=2,
                      config=ModelConfig(patch_dim=48,embedding_dim=16,encoder_depth=1),
                      context_variance_weight=.1,checkpoint=self.checkpoint)
        self.state=torch.load(self.checkpoint,weights_only=True)
        Fixture.accessed.clear()

    def extract(self):
        return extract_cifar10(self.root,self.m,selection_path=self.s,
                              checkpoint=self.checkpoint,output=self.features)

    def load(self):
        return load_features(self.features,manifest=self.manifest,selection=self.selection,
                             training_labels=self.labels,partition='train')

    def test_regularized_frozen_features_keep_objective_sources_rng_and_training_rows(self):
        before=self.checkpoint.read_bytes();rng=torch.get_rng_state().clone()
        self.extract();first=self.load();self.extract();second=self.load()
        self.assertTrue(torch.equal(first.features,second.features))
        self.assertEqual(self.state['settings'],first.metadata['training_settings'])
        self.assertEqual(ObjectiveConfig(.1),objective_from_settings(first.metadata['training_settings']))
        self.assertEqual(before,self.checkpoint.read_bytes())
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        allowed=set(self.selection['partitions']['train']['indices'])
        self.assertTrue(all(train and index in allowed for train,index in Fixture.accessed))

    def test_bad_checkpoint_objective_fails_before_image_reads_or_output_replacement(self):
        self.features.write_bytes(b'previous report')
        for spec in [None,{},dict(context_variance_weight=0,context_std_floor=1),
                     dict(context_variance_weight=True,context_std_floor=1),
                     dict(context_variance_weight=-.1,context_std_floor=1),
                     dict(context_variance_weight=.1,context_std_floor=1,extra='unknown')]:
            state=deepcopy(self.state);state['settings']['objective']=spec
            torch.save(state,self.checkpoint);Fixture.accessed.clear()
            with self.assertRaises(ValueError):self.extract()
            self.assertEqual([],Fixture.accessed)
            self.assertEqual(b'previous report',self.features.read_bytes())

    def test_malformed_feature_recipe_is_rejected_even_with_unchanged_feature_values(self):
        self.extract();good=torch.load(self.features,weights_only=True)
        for spec in [{},dict(context_variance_weight=.1,context_std_floor=0),
                     dict(context_variance_weight=False,context_std_floor=1)]:
            bad=deepcopy(good);bad['feature_metadata']['training_settings']['objective']=spec
            torch.save(bad,self.features)
            with self.assertRaises(ValueError):self.load()
        self.assertEqual(ObjectiveConfig(),objective_from_settings({'seed':29}))
