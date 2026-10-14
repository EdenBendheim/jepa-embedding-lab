from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from jepa_lab.cifar10 import make_manifest
from jepa_lab.embeddings import extract_cifar10
from jepa_lab.features import load_features
from jepa_lab.pretrained import pretrained_metadata
from jepa_lab.selection import make_pilot
from test_embeddings import Fixture


class PretrainedFeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        folder = tempfile.TemporaryDirectory(); self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        self.labels = Fixture(self.root).labels
        self.manifest = make_manifest(Fixture(self.root), Fixture(self.root, train=False))
        self.selection = make_pilot(self.manifest, self.labels, train_per_class=1, validation_per_class=1)
        m, s = self.root/"manifest.json", self.root/"selection.json"
        m.write_text(json.dumps(self.manifest)); s.write_text(json.dumps(self.selection))
        self.path = self.root/"features.pt"
        with patch("jepa_lab.embeddings.CIFAR10Binary", Fixture):
            extract_cifar10(self.root, m, representation="pixels", selection_path=s, output=self.path)
        self.original = torch.load(self.path, weights_only=True)
        values = torch.arange(10*512,dtype=torch.float32).reshape(10,512)/512
        self.original |= dict(features=values, features_sha256=hashlib.sha256(values.numpy().tobytes()).hexdigest(),
                              feature_metadata=pretrained_metadata())
        torch.save(self.original,self.path)

    def load(self):
        return load_features(self.path,manifest=self.manifest,selection=self.selection,
                             training_labels=self.labels,partition="train")

    def test_verified_pretrained_rows_are_copy_isolated(self):
        with patch("torch.hub.download_url_to_file",side_effect=AssertionError("No network")):
            source=self.load()
        self.assertEqual(tuple(source.features.shape),(10,512))
        self.assertEqual(source.metadata,pretrained_metadata())
        self.assertEqual(source.indices,tuple(self.original["indices"]))
        source.features.zero_(); source.metadata["transform"]["mean"][0]=100
        self.assertFalse(torch.equal(source.features,self.load().features))
        self.assertEqual(self.load().metadata,pretrained_metadata())

    def test_weight_transform_dimension_and_runtime_mislabeling_rejected(self):
        for key,value in (("weights_sha256","0"*64),("architecture","JEPA"),("feature_dim",32),
                          ("normalization","x / 127.5 - 1"),("torchvision","wrong"),
                          ("transform",{"name":"different crop"})):
            with self.subTest(key=key),self.assertRaises(ValueError):
                saved=deepcopy(self.original);saved["feature_metadata"][key]=value
                torch.save(saved,self.path);self.load()
        saved=deepcopy(self.original);saved["features"]=saved["features"][:,:32]
        saved["features_sha256"]=hashlib.sha256(saved["features"].contiguous().numpy().tobytes()).hexdigest()
        torch.save(saved,self.path)
        with self.assertRaises(ValueError):self.load()
        saved=deepcopy(self.original);saved["extraction_settings"]["batch_size"]=17
        torch.save(saved,self.path)
        with self.assertRaises(ValueError):self.load()
