import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch

from jepa_lab.cifar10 import make_manifest
from jepa_lab.compare import write_report
from jepa_lab.embeddings import extract_cifar10
from jepa_lab.neighbor_cli import run_neighbors
from jepa_lab.selection import make_pilot
from test_embeddings import Fixture


class NeighborWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_saved_feature_selection_restore_and_corruption_never_reads_raw_images(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);manifest=make_manifest(Fixture(root),Fixture(root,train=False))
            selection=make_pilot(manifest,Fixture(root).labels,train_per_class=1,validation_per_class=1)
            m,s=root/'manifest.json',root/'selection.json';m.write_text(json.dumps(manifest));s.write_text(json.dumps(selection))
            paths=[root/f'{part}.pt' for part in ('train','validation')]
            with patch('jepa_lab.embeddings.CIFAR10Binary',Fixture):
                for path,part in zip(paths,('train','validation')):
                    extract_cifar10(root,m,representation='pixels',selection_path=s,partition=part,output=path)
            state,report=root/'index.pt',root/'report.json'
            with patch('jepa_lab.neighbor_cli.CIFAR10Binary',Fixture),patch.object(Fixture,'__getitem__',side_effect=AssertionError('raw images')):
                first=run_neighbors(root,m,s,*paths,ks=[1,3],save_index=state);write_report(report,first)
                old_report=report.read_bytes();old_state=state.read_bytes()
                with patch('jepa_lab.neighbor_cli.select_neighbors',side_effect=AssertionError('retuning')):
                    restored=run_neighbors(root,m,s,*paths,restore_index=state)
                self.assertEqual(first['validation'],restored['validation']);self.assertTrue(restored['restored_without_fitting'])
                self.assertEqual(old_state,state.read_bytes())
                saved=torch.load(paths[1],weights_only=True);saved['features'][0,0]+=.5;torch.save(saved,paths[1])
                with self.assertRaises(ValueError):run_neighbors(root,m,s,*paths,restore_index=state)
                self.assertEqual(old_report,report.read_bytes());self.assertEqual(old_state,state.read_bytes())
