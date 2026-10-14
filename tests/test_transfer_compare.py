from copy import deepcopy
from dataclasses import replace
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from jepa_lab.compare import fit_representation
from jepa_lab.features import FeatureSet
from jepa_lab.pretrained import pretrained_metadata
from jepa_lab.probe_artifacts import load_probe
from jepa_lab.probes import ProbeConfig
from jepa_lab.transfer_compare import ARMS, compare_transfer
from test_compare import sets


def transfer_sets():
    data=sets()
    data["pretrained"]=tuple(replace(item,features=torch.cat((item.features[:,:10],torch.zeros(10,502)),1),
        metadata=deepcopy(pretrained_metadata())) for item in data["checkpoint"])
    for name,pair in data.items():
        data[name]=tuple(replace(item,
            features_sha256=hashlib.sha256(item.features.numpy().tobytes()).hexdigest(),
            labels_sha256=hashlib.sha256(item.labels.numpy().tobytes()).hexdigest()) for item in pair)
    return data


class TransferComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_shared_budgets_saved_states_and_rng(self):
        data=transfer_sets();rng=torch.get_rng_state().clone()
        with tempfile.TemporaryDirectory() as directory:
            with patch("jepa_lab.transfer_compare.fit_representation",wraps=fit_representation) as fitted:
                report=compare_transfer(data,(ProbeConfig(steps=30),),[29,31],
                    probe_dir=directory,ridge_alphas=[.1,1])
            self.assertEqual(fitted.call_count,8)
            self.assertEqual(set(report["summary"]),set(ARMS))
            self.assertEqual(report["summary"]["pretrained"]["feature_dim"],512)
            self.assertIn("ImageNet supervision",report["limitation"])
            for index,run in enumerate(report["runs"]):
                for name,result in run["representations"].items():
                    probe,artifact=load_probe(result["selected_probe_artifact"]["path"],*data[name])
                    self.assertEqual(probe.score(data[name][1].features,data[name][1].labels),result["validation"])
                    self.assertEqual(result["selected_config"]["seed"],[29,31][index])
                    self.assertEqual("ridge" in result,index==0)
                    self.assertEqual(artifact["state_sha256"],result["selected_probe_artifact"]["state_sha256"])
            self.assertTrue(torch.equal(rng,torch.get_rng_state()))
            self.assertFalse(report["test_evaluated"])

    def test_misalignment_confirmation_and_changed_content_fail_before_fitting(self):
        for change in ("indices","labels","weights","dimension","confirmation","content","baseline_content"):
            data=transfer_sets();train,validation=data["pretrained"]
            if change=="indices":validation=replace(validation,indices=validation.indices[::-1])
            if change=="labels":validation=replace(validation,labels=validation.labels.roll(1))
            if change=="weights":validation=replace(validation,metadata=validation.metadata|dict(weights_sha256="0"*64))
            if change=="dimension":validation=replace(validation,features=validation.features[:,:32])
            if change=="confirmation":validation=replace(validation,selection_kind="confirmation")
            if change=="content":validation.features[0,0]+=1
            if change=="baseline_content":data["pixels"][0].features[0,0]+=1
            data["pretrained"]=(train,validation)
            with self.subTest(change=change), \
                    patch("jepa_lab.transfer_compare.fit_representation",side_effect=AssertionError("fit ran")), \
                    self.assertRaises(ValueError):
                compare_transfer(data,(ProbeConfig(steps=2),),[29])
