from copy import deepcopy
import unittest
import torch

from jepa_lab.neighbors import NeighborConfig,fit_neighbors


class NeighborTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_exact_neighbors_training_only_stats_and_unchanged_sources_rng(self):
        x=torch.eye(10);y=torch.arange(10);before=x.clone();rng=torch.get_rng_state().clone()
        index=fit_neighbors(x,y,NeighborConfig(1));stats=(index.mean.clone(),index.scale.clone())
        self.assertTrue(torch.equal(y,index.predict(x)))
        self.assertEqual(1.,index.score(x,y)['accuracy'])
        index.predict(x+100)
        self.assertTrue(torch.equal(stats[0],index.mean));self.assertTrue(torch.equal(stats[1],index.scale))
        self.assertTrue(torch.equal(before,x));self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        x.fill_(100);self.assertTrue(torch.equal(y,index.predict(before)))

    def test_zero_vectors_neighbor_and_vote_ties_are_deterministic(self):
        x=torch.zeros(10,3);y=torch.tensor([9,8,7,6,5,4,3,2,1,0])
        self.assertEqual([9,9],fit_neighbors(x,y,NeighborConfig(1)).predict(x[:2]).tolist())
        self.assertEqual([8,8],fit_neighbors(x,y,NeighborConfig(2)).predict(x[:2]).tolist())
        self.assertTrue(torch.equal(fit_neighbors(x,y).scale,torch.ones(3)))

    def test_bad_counts_inputs_and_shapes_are_rejected(self):
        x=torch.eye(10);y=torch.arange(10)
        for k in (True,0,101):
            with self.assertRaises(ValueError):NeighborConfig(k)
        for a,b,k in ((x,y,11),(x.double(),y,1),(x,torch.zeros(10,dtype=torch.long),1),
                       (x*float('nan'),y,1)):
            with self.assertRaises(ValueError):fit_neighbors(a,b,NeighborConfig(k))
        index=fit_neighbors(x,y)
        for query in (torch.ones(2,9),x*float('inf')):
            with self.assertRaises(ValueError):index.predict(query)
