from dataclasses import replace
import unittest
from unittest.mock import patch
import torch

from jepa_lab.confirmation_compare import compare_confirmed_probes,run_paired_confirmation
from jepa_lab.probe_artifacts import save_probe
from jepa_lab.probes import fit_probe,ProbeConfig
import test_confirmation as fixtures
from test_embeddings import Fixture


class PairedConfirmationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)
    setUp=fixtures.ConfirmationScoreTests.setUp

    def test_known_recoveries_and_regressions_preserve_states_rng_and_no_refit(self):
        train,old,fresh=self.sets
        probe=fit_probe(train.features,train.labels,ProbeConfig(steps=2))
        paths=[]
        for label in (0,1):
            bias=torch.zeros_like(probe.bias);bias[label]=1
            p=replace(probe,weight=torch.zeros_like(probe.weight),bias=bias)
            path=self.root/f'fixed{label}.pt';save_probe(path,p,train,old);paths.append(path)
        before={p:p.read_bytes() for p in [*paths,*self.paths]};rng=torch.get_rng_state().clone()
        with patch('jepa_lab.probes.fit_probe',side_effect=AssertionError('refit')),patch.object(Fixture,'__getitem__',side_effect=AssertionError('image read')):
            report=compare_confirmed_probes(self.sets,self.sets,*paths,self.selection)
            with patch('jepa_lab.confirmation_compare.CIFAR10Binary',Fixture):
                cli=run_paired_confirmation(self.root,self.m,self.p,self.c,[*self.paths,paths[0]],[*self.paths,paths[1]])
        self.assertEqual(report,cli);paired=report['paired_confirmation']
        self.assertEqual(2,paired['recovered']);self.assertEqual(2,paired['regressed']);self.assertEqual(0.,paired['accuracy_delta'])
        self.assertFalse(report['refitted']);self.assertFalse(report['test_evaluated'])
        self.assertTrue(torch.equal(rng,torch.get_rng_state()));self.assertEqual(before,{p:p.read_bytes() for p in before})

    def test_corrupted_or_reordered_fresh_sources_fail_before_loading_classifiers(self):
        train,old,fresh=self.sets
        for changed in (replace(fresh,indices=fresh.indices[::-1]),replace(fresh,features=fresh.features+.2)):
            with patch('jepa_lab.confirmation_compare.load_probe',side_effect=AssertionError('loaded invalid state')),self.assertRaises(ValueError):
                compare_confirmed_probes(self.sets,(train,old,changed),'unused','unused',self.selection)
