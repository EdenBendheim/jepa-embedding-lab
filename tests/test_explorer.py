from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from jepa_lab.explorer import explore_neighbors, load_exploration
from jepa_lab.neighbors import fit_neighbors, NeighborConfig
from jepa_lab.neighbor_artifacts import save_neighbors
from test_compare import sets


class ExplorerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def test_report_has_exact_official_rows_scores_provenance_without_refitting(self):
        train,validation=sets()['checkpoint']
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'index.pt'
            save_neighbors(path,fit_neighbors(train.features,train.labels,NeighborConfig(3)),train,validation)
            before=path.read_bytes();rng=torch.get_rng_state().clone()
            with patch('jepa_lab.neighbors.fit_neighbors',side_effect=AssertionError('refit')):
                report=explore_neighbors(path,train,validation,query_indices=[19,10])
            self.assertEqual([19,10],[q['index'] for q in report['queries']])
            self.assertEqual([9,0],[q['neighbors'][0]['index'] for q in report['queries']])
            self.assertAlmostEqual(1.,report['queries'][0]['neighbors'][0]['cosine'],places=6)
            self.assertEqual('c'*64,report['training']['artifact_sha256'])
            self.assertNotIn('features',report);json.dumps(report,allow_nan=False)
            self.assertFalse(report['test_evaluated']);self.assertTrue(report['restored_without_fitting'])
            self.assertEqual(before,path.read_bytes());self.assertTrue(torch.equal(rng,torch.get_rng_state()))
            report['training']['feature_metadata']['pooling']='edited'
            self.assertNotEqual('edited',train.metadata['pooling'])
            for ids in ([],[10,10],[True],[0],list(range(101))):
                with self.assertRaises(ValueError):explore_neighbors(path,train,validation,query_indices=ids)
            with self.assertRaises(ValueError):explore_neighbors(path,train,replace(validation,features=validation.features+.1))
            with self.assertRaises(ValueError):explore_neighbors(path,train,replace(validation,selection_kind='confirmation'))

    def test_disk_workflow_rejects_corruption_without_reading_image_records(self):
        from test_embeddings import Fixture
        from jepa_lab.cifar10 import make_manifest
        from jepa_lab.embeddings import extract_cifar10
        from jepa_lab.selection import make_pilot
        from jepa_lab.neighbor_cli import run_neighbors
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);manifest=make_manifest(Fixture(root),Fixture(root,train=False))
            selection=make_pilot(manifest,Fixture(root).labels,train_per_class=1,validation_per_class=1)
            m,s=root/'m.json',root/'s.json';m.write_text(json.dumps(manifest));s.write_text(json.dumps(selection))
            paths=[root/f'{part}.pt' for part in ('train','validation')];index=root/'index.pt'
            with patch('jepa_lab.embeddings.CIFAR10Binary',Fixture):
                for path,part in zip(paths,('train','validation')):
                    extract_cifar10(root,m,representation='pixels',selection_path=s,partition=part,output=path)
            with patch('jepa_lab.neighbor_cli.CIFAR10Binary',Fixture):
                run_neighbors(root,m,s,*paths,ks=[3],save_index=index)
            with patch('jepa_lab.explorer.CIFAR10Binary',Fixture),patch.object(Fixture,'__getitem__',side_effect=AssertionError('image read')):
                result=load_exploration(root,m,s,*paths,index)
                self.assertEqual(10,len(result['queries']))
                saved=torch.load(paths[1],weights_only=True);saved['features'][0,0]+=.1;torch.save(saved,paths[1])
                with self.assertRaises(ValueError):load_exploration(root,m,s,*paths,index)
