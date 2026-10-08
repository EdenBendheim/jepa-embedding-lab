from copy import deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from jepa_lab.cifar10 import make_manifest
from jepa_lab.compare import compare_features, run_comparison, write_report
from jepa_lab.embeddings import random_encoder
from jepa_lab.features import FeatureSet
from jepa_lab.model import ModelConfig
from jepa_lab.probes import ProbeConfig
from jepa_lab.selection import make_pilot
from test_embeddings import Fixture


def sets():
    labels = torch.arange(10)
    config = asdict(ModelConfig(patch_dim=48,embedding_dim=16,encoder_depth=1))
    result = {}
    for name in ("checkpoint","random","pixels"):
        values = torch.zeros(10,3072 if name == "pixels" else 16)
        values[:,:10] = torch.eye(10)
        metadata = dict(representation=name,normalization="x / 127.5 - 1",config=config,encoder="context",
                        pooling="flatten NCHW RGB values" if name == "pixels" else "mean of all 64 patch tokens")
        result[name] = tuple(FeatureSet(values.clone(),labels.clone(),tuple(range(start,start+10)),partition,
            "a"*64,"b"*64,deepcopy(metadata),"c"*64) for partition,start in (("train",0),("validation",10)))
    return result


class ComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def test_matched_comparison_records_counts_provenance_metrics_and_no_test(self):
        data = sets()
        state = torch.get_rng_state().clone()
        report = compare_features(data,(ProbeConfig(steps=30),))
        self.assertEqual(dict(train=10,validation=10),report["counts"])
        self.assertFalse(report["test_evaluated"])
        self.assertTrue(torch.equal(state,torch.get_rng_state()))
        for result in report["representations"].values():
            self.assertEqual(1.0,result["validation"]["accuracy"])
            self.assertEqual("c"*64,result["train_artifact"]["artifact_sha256"])

    def test_mismatched_examples_splits_metadata_or_architecture_fail_before_fitting(self):
        for change in ("order","overlap","selection","architecture","source","nonfinite"):
            data = sets(); train,validation = data["random"]
            if change == "order": validation = replace(validation,indices=validation.indices[::-1])
            if change == "overlap": validation = replace(validation,indices=train.indices)
            if change == "selection": validation = replace(validation,selection_sha256="wrong")
            if change == "source": validation = replace(validation,metadata=validation.metadata | dict(seed=7))
            if change == "architecture":
                train = replace(train,metadata=train.metadata | dict(config=train.metadata["config"] | dict(encoder_depth=2)))
                validation = replace(validation,metadata=train.metadata)
            if change == "nonfinite": validation = replace(validation,features=torch.full_like(validation.features,float("nan")))
            data["random"] = (train,validation)
            with self.subTest(change=change),patch("jepa_lab.compare.select_probe",side_effect=AssertionError("fit ran")),self.assertRaises(ValueError):
                compare_features(data,(ProbeConfig(steps=2),))

    def test_complete_fixture_extraction_verification_comparison_and_atomic_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = make_manifest(Fixture(root),Fixture(root,train=False))
            selection = make_pilot(manifest,Fixture(root).labels,train_per_class=1,validation_per_class=1)
            m,s = root/"manifest.json",root/"selection.json"
            m.write_text(json.dumps(manifest)); s.write_text(json.dumps(selection))
            config = ModelConfig(patch_dim=48,embedding_dim=16,encoder_depth=1)
            checkpoint = root/"encoder.pt"
            torch.save(dict(format_version=1,dataset="CIFAR-10",manifest_sha256=manifest["manifest_sha256"],
                runtime=dict(torch=str(torch.__version__),cpu_threads=1),step=3,config=asdict(config),
                model=random_encoder(config,29).state_dict(),settings=dict(seed=29)),checkpoint)
            Fixture.accessed.clear()
            with patch("jepa_lab.compare.CIFAR10Binary",Fixture),patch("jepa_lab.embeddings.CIFAR10Binary",Fixture):
                report = run_comparison(root,m,s,root/"features",checkpoint=checkpoint,candidates=(ProbeConfig(steps=3),))
                self.assertTrue(all(official for official,index in Fixture.accessed))
                Fixture.accessed.clear()
                again = run_comparison(root,m,s,root/"features",candidates=(ProbeConfig(steps=3),))
                self.assertEqual([],Fixture.accessed)
            for name in report["representations"]:
                self.assertEqual(report["representations"][name]["validation"],again["representations"][name]["validation"])
            output = root/"report.json"; write_report(output,report)
            previous = output.read_bytes()
            with patch("jepa_lab.compare.os.replace",side_effect=OSError("simulated disk error")),self.assertRaises(OSError):
                write_report(output,again)
            self.assertEqual(previous,output.read_bytes())
            self.assertEqual([],list(root.glob(".report.json.*")))
