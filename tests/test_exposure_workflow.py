from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
from pathlib import Path
import hashlib
import json
import sys
import unittest
from unittest.mock import patch

import torch

from jepa_lab.compare import fit_representation
from jepa_lab.confirmation import score_confirmation, score_saved_confirmation
from jepa_lab.embeddings import extract_cifar10
from jepa_lab.exposure import make_exposure_bundle, make_ledger_confirmation
from jepa_lab.features import load_features
from jepa_lab.probe_artifacts import save_probe
from jepa_lab.probes import fit_probe, ProbeConfig
from jepa_lab.ridge import fit_ridge, RidgeConfig
from jepa_lab.selection import main, validate_pilot
from test_embeddings import Fixture
import test_confirmation


class ExposureWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def setUp(self):
        fixture = test_confirmation.ConfirmationScoreTests()
        fixture.setUp(); self.addCleanup(fixture.doCleanups)
        self.f = fixture
        self.source_paths = [fixture.p, fixture.c]
        self.bundle = make_exposure_bundle(fixture.manifest, fixture.labels,
            [(p, hashlib.sha256(p.read_bytes()).hexdigest()) for p in self.source_paths])
        self.bundle_path = fixture.root / 'exposure.json'
        self.bundle_path.write_text(json.dumps(self.bundle))
        self.selected = make_ledger_confirmation(fixture.manifest, fixture.labels, fixture.pilot,
                                                 self.bundle['ledger'], validation_per_class=2, seed=19)

    def test_cli_reservation_rechecks_pinned_sources_and_preserves_old_output_on_failure(self):
        f = self.f; out = f.root/'later.json'
        args = ['selection', str(f.root), '--manifest', str(f.m), '--output', str(out),
                '--confirmation-of', str(f.p), '--exposure-bundle', str(self.bundle_path),
                '--expected-ledger-sha256', self.bundle['ledger']['ledger_sha256'],
                '--validation-per-class', '2', '--seed', '19']
        with patch('jepa_lab.selection.CIFAR10Binary', Fixture), patch.object(sys, 'argv', args), redirect_stdout(StringIO()):
            Fixture.accessed.clear(); main()
        self.assertEqual([], Fixture.accessed)
        self.assertEqual(self.selected, json.loads(out.read_text()))
        self.assertEqual(self.selected, validate_pilot(self.selected, f.manifest, f.labels))
        before = out.read_bytes()
        for bad in (args + ['--expected-ledger-sha256', '0'*64],
                    args + ['--output', str(self.bundle_path)],
                    args + ['--exclude-pilot', str(f.c)]):
            with patch('jepa_lab.selection.CIFAR10Binary', Fixture), patch.object(sys, 'argv', bad), redirect_stderr(StringIO()):
                with self.assertRaises(SystemExit): main()
        with patch('jepa_lab.selection.CIFAR10Binary', Fixture), patch.object(sys, 'argv', args), patch('jepa_lab.compare.os.replace', side_effect=OSError('disk failure')), redirect_stderr(StringIO()):
            with self.assertRaises(SystemExit): main()
        self.assertEqual([], list(f.root.glob('.later.json.*')))
        f.c.write_text(f.c.read_text() + '\n')
        with patch('jepa_lab.selection.CIFAR10Binary', Fixture), patch.object(sys, 'argv', args), redirect_stderr(StringIO()):
            with self.assertRaises(SystemExit): main()
        self.assertEqual(before, out.read_bytes())

    def test_ledger_features_only_allow_frozen_scoring_with_unchanged_classifiers(self):
        f = self.f; path = f.root/'later.json'; path.write_text(json.dumps(self.selected))
        features = f.root/'later.pt'
        with patch('jepa_lab.embeddings.CIFAR10Binary', Fixture):
            extract_cifar10(f.root, f.m, representation='pixels', partition='validation', selection_path=path, output=features)
        fresh = load_features(features, manifest=f.manifest, selection=self.selected,
                              training_labels=f.labels, partition='validation')
        train, tuning = f.sets[:2]
        with patch('jepa_lab.compare.select_probe', side_effect=AssertionError('retuned fresh rows')):
            with self.assertRaisesRegex(ValueError, 'Confirmation'):
                fit_representation((train, fresh), (ProbeConfig(steps=1),), 'pixels')
        for family, head in (('adamw', fit_probe(train.features, train.labels, ProbeConfig(steps=2))),
                             ('ridge', fit_ridge(train.features, train.labels, RidgeConfig(alpha=1)))):
            classifier = f.root/f'{family}-later.pt'; save_probe(classifier, head, train, tuning)
            before = {p: p.read_bytes() for p in [classifier, features, *f.paths]}
            Fixture.accessed.clear(); rng = torch.get_rng_state().clone()
            with patch('jepa_lab.confirmation.CIFAR10Binary', Fixture), patch('jepa_lab.probes.fit_probe', side_effect=AssertionError('refitted')):
                result = score_saved_confirmation(f.root, f.m, f.p, path, f.paths[0], f.paths[1], features, classifier)
            self.assertEqual(head.score(fresh.features, fresh.labels), result['confirmation'])
            self.assertEqual(len(self.bundle['ledger']['observed_validation_indices']), result['excluded_validation_count'])
            self.assertFalse(result['classifier_refitted']); self.assertFalse(result['test_evaluated'])
            self.assertEqual([], Fixture.accessed); self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            self.assertEqual(before, {p: p.read_bytes() for p in before})
