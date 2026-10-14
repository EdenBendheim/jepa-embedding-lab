from contextlib import redirect_stdout,redirect_stderr
import hashlib
from io import StringIO
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

from jepa_lab.cifar10 import make_manifest
from jepa_lab.embeddings import extract_cifar10,random_encoder
from jepa_lab.model import ModelConfig
from jepa_lab.pretrained import pretrained_metadata
from jepa_lab.pretrained_extract import extract_pretrained
from jepa_lab.probes import ProbeConfig
from jepa_lab.selection import make_pilot
from jepa_lab.transfer_compare import main,run_transfer,load_transfer_sets
from test_embeddings import Fixture
from test_pretrained_extract import synthetic_encoder


class TransferWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def setUp(self):
        from dataclasses import asdict
        folder=tempfile.TemporaryDirectory();self.addCleanup(folder.cleanup);self.root=Path(folder.name)
        self.manifest=make_manifest(Fixture(self.root),Fixture(self.root,train=False))
        selection=make_pilot(self.manifest,Fixture(self.root).labels,train_per_class=1,validation_per_class=1)
        self.m,self.s=self.root/"manifest.json",self.root/"selection.json"
        self.m.write_text(json.dumps(self.manifest));self.s.write_text(json.dumps(selection))
        self.feature_dir,self.pretrained_dir=self.root/"controls",self.root/"pretrained"
        config=ModelConfig(patch_dim=48,embedding_dim=16,encoder_depth=1)
        checkpoint=self.root/"encoder.pt"
        torch.save(dict(format_version=1,dataset="CIFAR-10",manifest_sha256=self.manifest["manifest_sha256"],
            runtime=dict(torch=str(torch.__version__),cpu_threads=1),step=3,config=asdict(config),
            model=random_encoder(config,29).state_dict(),settings=dict(seed=29)),checkpoint)
        with patch("jepa_lab.embeddings.CIFAR10Binary",Fixture), \
                patch("jepa_lab.pretrained_extract.CIFAR10Binary",Fixture), \
                patch("jepa_lab.pretrained_extract.load_pretrained",return_value=(None,pretrained_metadata())), \
                patch("jepa_lab.pretrained_extract.pretrained_embeddings",side_effect=synthetic_encoder):
            for partition in ("train","validation"):
                for representation in ("checkpoint","random","pixels"):
                    extract_cifar10(self.root,self.m,selection_path=self.s,partition=partition,
                        representation=representation,checkpoint=checkpoint if representation=="checkpoint" else None,
                        config=config if representation=="random" else None,
                        output=self.feature_dir/f"{representation}-{partition}.pt")
                extract_pretrained(self.root,self.m,self.s,"synthetic.pth",partition=partition,
                    output=self.pretrained_dir/f"pretrained-{partition}.pt")
        self.probes=self.root/"probes";self.output=self.root/"report.json"

    def test_saved_feature_cli_never_reads_images_and_preserves_sources_on_failure(self):
        sources=[self.m,self.s,*self.feature_dir.glob("*.pt"),*self.pretrained_dir.glob("*.pt")]
        before={p:p.read_bytes() for p in sources};Fixture.accessed.clear()
        args=["transfer_compare",str(self.root),"--manifest",str(self.m),"--selection",str(self.s),
            "--feature-dir",str(self.feature_dir),"--pretrained-dir",str(self.pretrained_dir),
            "--probe-dir",str(self.probes),"--steps","2","--learning-rates",".001",
            "--weight-decays",".01","--probe-seeds","29","--ridge-alphas",".1",
            "--output",str(self.output)]
        with patch("jepa_lab.transfer_compare.CIFAR10Binary",Fixture),patch.object(sys,"argv",args),redirect_stdout(StringIO()):main()
        report=json.loads(self.output.read_text())
        self.assertFalse(report["test_evaluated"])
        self.assertEqual(report["counts"],dict(train=10,validation=10))
        self.assertEqual(Fixture.accessed,[])
        self.assertEqual(before,{p:p.read_bytes() for p in sources})
        saved=self.output.read_bytes()
        path=self.pretrained_dir/"pretrained-validation.pt"
        artifact=torch.load(path,weights_only=True);artifact["labels"]=artifact["labels"].roll(1);torch.save(artifact,path)
        with patch("jepa_lab.transfer_compare.CIFAR10Binary",Fixture), \
                patch("jepa_lab.transfer_compare.fit_representation",side_effect=AssertionError("fit ran")), \
                patch.object(sys,"argv",args),redirect_stderr(StringIO()):
            with self.assertRaises(SystemExit):main()
        self.assertEqual(self.output.read_bytes(),saved)
