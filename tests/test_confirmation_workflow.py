from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
import json
import sys
import unittest
from unittest.mock import patch

import torch

from jepa_lab.confirmation import main, score_saved_confirmation
from jepa_lab.probe_artifacts import save_probe
from jepa_lab.probes import fit_probe, ProbeConfig
import test_confirmation as fixtures
from test_embeddings import Fixture


class ConfirmationWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def setUp(self):
        fixtures.ConfirmationScoreTests.setUp(self)
        self.probe_path=self.root/'probe.pt'
        probe=fit_probe(self.sets[0].features,self.sets[0].labels,ProbeConfig(steps=2))
        save_probe(self.probe_path,probe,*self.sets[:2])

    def test_cli_reproduces_without_images_fits_or_source_changes(self):
        output=self.root/'report.json'
        before={p:p.read_bytes() for p in [self.m,self.p,self.c,self.probe_path,*self.paths]}
        args=['confirmation',str(self.root),str(self.probe_path),'--manifest',str(self.m),
              '--reference-selection',str(self.p),'--confirmation-selection',str(self.c),
              '--train-features',str(self.paths[0]),'--tuning-features',str(self.paths[1]),
              '--confirmation-features',str(self.paths[2]),'--output',str(output)]
        Fixture.accessed.clear()
        with patch('jepa_lab.confirmation.CIFAR10Binary',Fixture),patch('jepa_lab.probes.fit_probe',side_effect=AssertionError('refit')):
            expected=score_saved_confirmation(self.root,self.m,self.p,self.c,*self.paths,self.probe_path)
            with patch.object(sys,'argv',args),redirect_stdout(StringIO()): main()
        self.assertEqual(expected,json.loads(output.read_text()))
        self.assertEqual([],Fixture.accessed)
        self.assertEqual(before,{p:p.read_bytes() for p in before})
        saved=output.read_bytes()
        self.c.write_text(json.dumps(self.selection|dict(seed=0)))
        with patch('jepa_lab.confirmation.CIFAR10Binary',Fixture),patch.object(sys,'argv',args),redirect_stderr(StringIO()):
            with self.assertRaises(SystemExit): main()
        self.assertEqual(saved,output.read_bytes())
