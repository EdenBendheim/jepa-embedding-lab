import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import unittest
from unittest.mock import patch

import torch

from jepa_lab.pretrained import WEIGHTS_SHA256
from test_embeddings import Fixture
import test_transfer_workflow as workflow


class TransferDriverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def setUp(self):
        workflow.TransferWorkflowTests.setUp(self)
        path=Path(__file__).resolve().parents[1]/"experiments/run_pretrained_20261013.py"
        spec=importlib.util.spec_from_file_location("transfer_driver",path)
        self.driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(self.driver)
        self.directory=self.root/"run"
        (self.directory/"features").mkdir(parents=True)
        for p in self.pretrained_dir.glob("*.pt"):shutil.copy2(p,self.directory/"features"/p.name)
        sources=[self.m,self.s,*self.feature_dir.glob("*.pt")]
        declared=dict(manifest_path=str(self.m),selection_path=str(self.s),feature_dir=str(self.feature_dir),
            manifest_sha256=self.manifest["manifest_sha256"],selection_sha256=json.loads(self.s.read_text())["selection_sha256"],
            sources=[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sources],
            weights_path=str(self.root/"missing-offline-weights.pth"),weights_sha256=WEIGHTS_SHA256,
            seeds=[29],steps=2,learning_rates=[.001],weight_decays=[.01],ridge_alphas=[.1])
        protocol=self.root/"protocol.json";protocol.write_text(json.dumps(declared))
        self.driver.PROTOCOL=protocol;self.driver.PROTOCOL_SHA=hashlib.sha256(protocol.read_bytes()).hexdigest()
        self.driver.ROOT=self.root

    def test_cache_retry_avoids_extraction_refitting_and_preserves_output_after_corruption(self):
        with patch.object(self.driver,"CIFAR10Binary",Fixture),patch("jepa_lab.transfer_compare.CIFAR10Binary",Fixture):
            expected=self.driver.run(self.directory)
            previous=(self.directory/"summary.json").read_bytes()
            with patch.object(self.driver,"extract_pretrained",side_effect=AssertionError("extraction")), \
                    patch.object(self.driver,"compare_transfer",side_effect=AssertionError("refit")), \
                    patch.object(Fixture,"__getitem__",side_effect=AssertionError("image sampling")):
                self.assertEqual(expected,self.driver.run(self.directory,reuse_only=True))
            self.assertEqual(previous,(self.directory/"summary.json").read_bytes())
            path=self.directory/"features/pretrained-validation.pt"
            saved=torch.load(path,weights_only=True);saved["features"][0,0]+=1;torch.save(saved,path)
            with self.assertRaises(ValueError):self.driver.run(self.directory,reuse_only=True)
            self.assertEqual(previous,(self.directory/"summary.json").read_bytes())
