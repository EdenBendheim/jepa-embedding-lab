from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import torch

from jepa_lab.explorer import explore_neighbors
from jepa_lab.explorer_compare import compare_explorations
from jepa_lab.explorer_html import render_html
from jepa_lab.neighbors import fit_neighbors,NeighborConfig
from jepa_lab.neighbor_artifacts import save_neighbors
from test_compare import sets


class ExplorerComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def reports(self,folder):
        data=sets();train,val=data['random'];data['random']=train,replace(val,features=val.features.roll(1,0))
        result={}
        for name,(train,val) in data.items():
            path=Path(folder)/f'{name}.pt'
            save_neighbors(path,fit_neighbors(train.features,train.labels,NeighborConfig(3)),train,val)
            result[name]=explore_neighbors(path,train,val)
        return result

    def test_exact_overlap_differences_common_depth_and_renderer_metric_validation(self):
        with tempfile.TemporaryDirectory() as folder:
            reports=self.reports(folder);before=deepcopy(reports);rng=torch.get_rng_state().clone()
            result=compare_explorations(reports,depth=1)
            pairs={(p['left'],p['right']):p for p in result['pairs']}
            self.assertEqual(0.,pairs['checkpoint','random']['mean_overlap_fraction'])
            self.assertEqual(1.,pairs['checkpoint','pixels']['mean_overlap_fraction'])
            row=pairs['checkpoint','random']['queries'][0]
            self.assertEqual([0],row['left_only']);self.assertEqual([9],row['right_only'])
            self.assertEqual(reports,before);self.assertTrue(torch.equal(rng,torch.get_rng_state()))
            self.assertIn('Representation',render_html(result))
            changed=deepcopy(result);changed['pairs'][0]['mean_overlap_fraction']=.99
            with self.assertRaises(ValueError):render_html(changed)
            self.assertEqual(3,compare_explorations(reports)['depth'])

    def test_mismatched_identity_labels_queries_or_depth_cannot_be_compared(self):
        with tempfile.TemporaryDirectory() as folder:
            original=self.reports(folder)
            for field in ('manifest_sha256','selection_sha256','count','partition'):
                r=deepcopy(original);r['random']['validation'][field]='wrong'
                with self.assertRaises(ValueError):compare_explorations(r)
            r=deepcopy(original);r['random']['queries'].reverse()
            with self.assertRaises(ValueError):compare_explorations(r)
            r=deepcopy(original);n=r['pixels']['queries'][0]['neighbors'][0];n['label']=1;n['class_name']='automobile'
            with self.assertRaises(ValueError):compare_explorations(r)
            for depth in (True,0,4):
                with self.assertRaises(ValueError):compare_explorations(original,depth=depth)
            for r in ({},dict(single=original['random']),original|{'<script>':original['random']}):
                with self.assertRaises(ValueError):compare_explorations(r)
