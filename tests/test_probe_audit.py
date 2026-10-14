from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from jepa_lab.compare import compare_repeated_features
from jepa_lab.probes import ProbeConfig
from jepa_lab.probe_audit import audit_repeated_report
from test_compare import sets


class ProbeAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_selected_states_pairs_and_seed_aggregates_reproduce_and_edits_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            data=sets();candidates=(ProbeConfig(steps=3),);seeds=[29,31];alphas=[.1,1.]
            report=compare_repeated_features(data,candidates,seeds,probe_dir=Path(directory),ridge_alphas=alphas)
            before={p:p.read_bytes() for p in Path(directory).rglob('*.pt')};rng=torch.get_rng_state().clone()
            with patch('jepa_lab.probes.fit_probe',side_effect=AssertionError('refit')):
                result=audit_repeated_report(report,data,candidates,seeds,ridge_alphas=alphas)
            self.assertEqual(9,result['verified_classifiers']);self.assertTrue(result['no_refitting'])
            self.assertTrue(torch.equal(rng,torch.get_rng_state()));self.assertEqual(before,{p:p.read_bytes() for p in before})
            for change in ('grid','source','pair','summary','boundary'):
                bad=deepcopy(report)
                if change=='grid':bad['runs'][0]['candidate_grid'][0]['steps']+=1
                if change=='source':bad['runs'][0]['representations']['checkpoint']['train_artifact']['artifact_sha256']='e'*64
                if change=='pair':bad['runs'][0]['paired_validation']['adamw']['checkpoint_vs_random']['both_correct']+=1
                if change=='summary':bad['summary']['checkpoint']['mean_accuracy']+=.1
                if change=='boundary':bad['test_evaluated']=True
                with self.subTest(change=change),self.assertRaises(ValueError):
                    audit_repeated_report(bad,data,candidates,seeds,ridge_alphas=alphas)
