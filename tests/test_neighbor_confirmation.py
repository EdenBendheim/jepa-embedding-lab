from dataclasses import replace
from pathlib import Path
import unittest
from unittest.mock import patch
import torch

from jepa_lab.neighbors import NeighborConfig
from jepa_lab.neighbor_artifacts import select_neighbors,save_neighbors,load_neighbors
from jepa_lab.neighbor_confirmation import score_neighbor_confirmation
import test_confirmation as fixtures


class NeighborConfirmationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def setUp(self):fixtures.ConfirmationScoreTests.setUp(self)

    def test_fixed_index_scores_fresh_rows_without_refit_or_statistics_changes(self):
        train,tuning,fresh=self.sets;path=self.root/'index.pt'
        index,_=select_neighbors(train,tuning,[NeighborConfig(1),NeighborConfig(3)]);save_neighbors(path,index,train,tuning)
        before={p:p.read_bytes() for p in [path,*self.paths]};rng=torch.get_rng_state().clone()
        with patch('jepa_lab.neighbor_artifacts.fit_neighbors',side_effect=AssertionError('refit')):
            result=score_neighbor_confirmation(path,train,tuning,fresh,self.selection)
        self.assertEqual(index.score(fresh.features,fresh.labels),result['confirmation'])
        self.assertFalse(result['normalization_refitted']);self.assertFalse(result['settings_selected']);self.assertFalse(result['index_refitted'])
        self.assertEqual(before,{p:p.read_bytes() for p in before});self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        for bad in (replace(fresh,features=fresh.features+.1),replace(fresh,indices=tuning.indices),replace(fresh,metadata=fresh.metadata|dict(encoder='changed'))):
            with patch('jepa_lab.neighbor_confirmation.load_neighbors',side_effect=AssertionError('loaded invalid source')),self.assertRaises(ValueError):
                score_neighbor_confirmation(path,train,tuning,bad,self.selection)
