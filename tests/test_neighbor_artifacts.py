from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch

from jepa_lab.neighbors import NeighborConfig
from jepa_lab.neighbor_artifacts import select_neighbors,save_neighbors,load_neighbors
from test_compare import sets


class NeighborArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_selection_uses_tuning_and_saved_state_reproduces_without_refitting(self):
        train,validation=sets()['checkpoint'];candidates=[NeighborConfig(1),NeighborConfig(3)]
        index,report=select_neighbors(train,validation,candidates)
        self.assertEqual(0,report['selected_candidate'])
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'neighbors.pt';saved=save_neighbors(path,index,train,validation);before=path.read_bytes()
            rng=torch.get_rng_state().clone()
            with patch('jepa_lab.neighbor_artifacts.fit_neighbors',side_effect=AssertionError('refit')):
                restored,metadata=load_neighbors(path,train,validation)
            self.assertEqual(saved,metadata);self.assertEqual(index.score(validation.features,validation.labels),restored.score(validation.features,validation.labels))
            self.assertTrue(torch.equal(rng,torch.get_rng_state()));self.assertEqual(before,path.read_bytes())
            state=torch.load(path,weights_only=True);state['state']['vectors'][0,0]+=.1;torch.save(state,path)
            with self.assertRaises(ValueError):load_neighbors(path,train,validation)

    def test_mutated_sources_confirmation_and_invalid_grids_are_rejected(self):
        from dataclasses import replace
        train,validation=sets()['random']
        for candidates in ([],[NeighborConfig(1)]*2,[1]):
            with self.assertRaises(ValueError):select_neighbors(train,validation,candidates)
        with self.assertRaises(ValueError):select_neighbors(train,replace(validation,selection_kind='confirmation'),[NeighborConfig(1)])
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'state.pt';index,_=select_neighbors(train,validation,[NeighborConfig(1)]);save_neighbors(path,index,train,validation)
            changed=replace(train,features=train.features+.2)
            with self.assertRaises(ValueError):load_neighbors(path,changed,validation)
