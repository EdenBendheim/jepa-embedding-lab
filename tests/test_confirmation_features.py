from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

from jepa_lab.cifar10 import make_manifest
from jepa_lab.compare import fit_representation
from jepa_lab.embeddings import extract_cifar10
from jepa_lab.features import load_features
from jepa_lab.probe_artifacts import probe_context
from jepa_lab.probes import ProbeConfig
from jepa_lab.selection import main, make_confirmation, make_pilot
from test_embeddings import Fixture


class ConfirmationFeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def setUp(self):
        folder = tempfile.TemporaryDirectory(); self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        self.labels = Fixture(self.root).labels
        self.manifest = make_manifest(Fixture(self.root), Fixture(self.root, train=False))
        self.pilot = make_pilot(self.manifest, self.labels, train_per_class=1, validation_per_class=1)
        self.confirm = make_confirmation(self.manifest, self.labels, self.pilot, validation_per_class=2)
        self.m, self.p, self.c = [self.root/f'{name}.json' for name in ('manifest','pilot','confirmation')]
        for path,value in ((self.m,self.manifest),(self.p,self.pilot),(self.c,self.confirm)):
            path.write_text(json.dumps(value))

    def load(self, path, selection, partition):
        return load_features(path, manifest=self.manifest, selection=selection, training_labels=self.labels, partition=partition)

    def test_validation_only_extraction_identity_and_fitting_boundary(self):
        trainpath, newpath = self.root/'train.pt', self.root/'new.pt'
        with patch('jepa_lab.embeddings.CIFAR10Binary', Fixture):
            extract_cifar10(self.root,self.m,representation='pixels',selection_path=self.p,output=trainpath)
            Fixture.accessed.clear()
            result = extract_cifar10(self.root,self.m,representation='pixels',partition='validation',selection_path=self.c,output=newpath)
            self.assertEqual([(True,i) for i in self.confirm['partitions']['validation']['indices']], Fixture.accessed)
            self.assertEqual('confirmation',result['selection_kind'])
            Fixture.accessed.clear()
            for args in (dict(partition='train'),dict(partition='test'),dict(partition='validation',limit=1)):
                with self.assertRaises(ValueError):
                    extract_cifar10(self.root,self.m,representation='pixels',selection_path=self.c,**args)
            self.assertEqual([],Fixture.accessed)
        old,new=self.load(trainpath,self.pilot,'train'),self.load(newpath,self.confirm,'validation')
        self.assertEqual('confirmation',new.summary()['selection_kind'])
        with patch('jepa_lab.compare.select_probe',side_effect=AssertionError('retuned confirmation')):
            with self.assertRaisesRegex(ValueError,'Confirmation'):
                fit_representation((old,new),(ProbeConfig(steps=1),),'pixels')
        with self.assertRaisesRegex(ValueError,'Confirmation'): probe_context(old,new)
        with self.assertRaises(ValueError): self.load(newpath,self.confirm,'train')
        edited=torch.load(newpath,weights_only=True); edited['selection_kind']='pilot'; torch.save(edited,newpath)
        with self.assertRaises(ValueError): self.load(newpath,self.confirm,'validation')

    def test_selector_cli_preserves_training_and_rejects_invalid_parent_before_output(self):
        out=self.root/'chosen.json'
        args=['selection',str(self.root),'--manifest',str(self.m),'--output',str(out),'--confirmation-of',str(self.p),
              '--validation-per-class','2','--seed','20261012']
        with patch('jepa_lab.selection.CIFAR10Binary',Fixture),patch.object(sys,'argv',args),redirect_stdout(StringIO()): main()
        self.assertEqual(self.confirm,json.loads(out.read_text()))
        before=out.read_bytes()
        with patch('jepa_lab.selection.CIFAR10Binary',Fixture),patch.object(sys,'argv',args+['--train-per-class','2']),redirect_stdout(StringIO()),redirect_stderr(StringIO()):
            with self.assertRaises(SystemExit): main()
        self.assertEqual(before,out.read_bytes())
        invalid=['selection',str(self.root),'--manifest',str(self.m),'--output',str(out),'--train-per-class','0']
        with patch('jepa_lab.selection.CIFAR10Binary',Fixture),patch.object(sys,'argv',invalid),redirect_stderr(StringIO()):
            with self.assertRaises(SystemExit): main()
        self.assertEqual(before,out.read_bytes())
