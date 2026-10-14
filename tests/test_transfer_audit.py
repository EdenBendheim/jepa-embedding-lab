from copy import deepcopy
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from jepa_lab.probes import ProbeConfig
from jepa_lab.transfer_compare import compare_transfer
from jepa_lab.transfer_audit import audit_transfer_report
from test_transfer_compare import transfer_sets


class TransferAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def setUp(self):
        folder=tempfile.TemporaryDirectory();self.addCleanup(folder.cleanup)
        self.root=Path(folder.name);self.sets=transfer_sets()
        self.report=compare_transfer(self.sets,(ProbeConfig(steps=3),),[29,31],
            probe_dir=self.root,ridge_alphas=[.1,1])

    def test_selected_states_pair_errors_and_aggregates_reproduce_without_fit(self):
        before={p:p.read_bytes() for p in self.root.rglob("*.pt")}
        rng=torch.get_rng_state().clone()
        with patch("jepa_lab.transfer_compare.fit_representation",side_effect=AssertionError("refit")), \
                patch("jepa_lab.probes.fit_probe",side_effect=AssertionError("refit")):
            audit=audit_transfer_report(self.report,self.sets)
        self.assertEqual(audit["verified_classifiers"],12)
        self.assertTrue(audit["no_refitting"])
        self.assertEqual(audit["summary"],self.report["summary"])
        self.assertEqual(before,{p:p.read_bytes() for p in before})
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))

    def test_edited_scores_configs_pairs_and_summaries_rejected(self):
        for change in ("score","config","pair","summary","source","seed","boundary"):
            report=deepcopy(self.report)
            row=report["runs"][0]["representations"]["pretrained"]
            if change=="score":row["validation"]["accuracy"]+=.1
            if change=="config":row["selected_config"]["steps"]+=1
            if change=="pair":report["runs"][0]["paired_validation"]["adamw"]["checkpoint_vs_pretrained"]["both_correct"]+=1
            if change=="summary":report["summary"]["pretrained"]["mean_accuracy"]+=.1
            if change=="source":row["train_artifact"]["artifact_sha256"]="0"*64
            if change=="seed":report["runs"][0]["seed"]=1
            if change=="boundary":report["test_evaluated"]=True
            with self.subTest(change=change),self.assertRaises(ValueError):audit_transfer_report(report,self.sets)
