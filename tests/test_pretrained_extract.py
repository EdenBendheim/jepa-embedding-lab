import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from jepa_lab.cifar10 import make_manifest
from jepa_lab.features import load_features
from jepa_lab.pretrained import pretrained_metadata
from jepa_lab.pretrained_extract import extract_pretrained
from jepa_lab.selection import make_pilot
from test_embeddings import Fixture


def synthetic_encoder(model,images):
    return images[:,0,0,0].float().unsqueeze(1).repeat(1,512)


class PretrainedExtractionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def setUp(self):
        folder=tempfile.TemporaryDirectory();self.addCleanup(folder.cleanup)
        self.root=Path(folder.name)
        self.labels=Fixture(self.root).labels
        self.manifest=make_manifest(Fixture(self.root),Fixture(self.root,train=False))
        self.selection=make_pilot(self.manifest,self.labels,train_per_class=1,validation_per_class=1)
        self.m,self.s=self.root/"manifest.json",self.root/"selection.json"
        self.m.write_text(json.dumps(self.manifest));self.s.write_text(json.dumps(self.selection))
        self.path=self.root/"pretrained.pt"
        self.patcher=patch("jepa_lab.pretrained_extract.CIFAR10Binary",Fixture)
        self.patcher.start();self.addCleanup(self.patcher.stop)
        Fixture.accessed.clear()

    def extract(self,**kwargs):
        with patch("jepa_lab.pretrained_extract.load_pretrained",return_value=(None,pretrained_metadata())), \
                patch("jepa_lab.pretrained_extract.pretrained_embeddings",side_effect=synthetic_encoder):
            return extract_pretrained(self.root,self.m,self.s,"synthetic-only.pth",output=self.path,**kwargs)

    def test_exact_rows_labels_and_only_selected_train_namespace(self):
        rng=torch.get_rng_state().clone()
        report=self.extract(partition="validation",batch_size=3)
        verified=load_features(self.path,manifest=self.manifest,selection=self.selection,
                              training_labels=self.labels,partition="validation")
        indices=self.selection["partitions"]["validation"]["indices"]
        self.assertEqual(Fixture.accessed,[(True,index) for index in indices])
        self.assertEqual(verified.indices,tuple(indices))
        self.assertEqual(verified.labels.tolist(),[self.labels[i] for i in indices])
        self.assertEqual(verified.features[:,0].tolist(),[float(i%256) for i in indices])
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertFalse(report["test_evaluated"])

    def test_invalid_identity_and_bounds_preserve_prior_output_before_sampling(self):
        self.path.write_bytes(b"previous output")
        for settings in (dict(partition="test"),dict(batch_size=17),dict(batch_size=True)):
            with self.assertRaises(ValueError):self.extract(**settings)
        bad=dict(self.manifest);bad["seed"]+=1;self.m.write_text(json.dumps(bad))
        with self.assertRaises(ValueError):self.extract()
        self.assertEqual(self.path.read_bytes(),b"previous output")
        self.assertEqual(Fixture.accessed,[])

    def test_weight_failure_precedes_sampling_and_failed_save_preserves_output(self):
        self.path.write_bytes(b"previous output")
        with patch("jepa_lab.pretrained_extract.load_pretrained",side_effect=ValueError("weight mismatch")):
            with self.assertRaises(ValueError):
                extract_pretrained(self.root,self.m,self.s,"bad.pth",output=self.path)
        self.assertEqual(Fixture.accessed,[])
        with patch("jepa_lab.pretrained_extract.atomic_checkpoint",side_effect=OSError("disk failure")):
            with self.assertRaises(OSError):self.extract()
        self.assertEqual(self.path.read_bytes(),b"previous output")
