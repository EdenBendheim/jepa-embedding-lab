import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from jepa_lab import objective_features
from jepa_lab.analysis.spatial_run import declare_protocol, main, run_analysis
from jepa_lab.exposure import make_exposure_bundle, make_ledger_confirmation
from jepa_lab.objective_run import file_sha
from jepa_lab.selection import make_confirmation, make_pilot
from test_cifar10_train import Fixture
import test_objective_run as fixtures


class SpatialRunTests(unittest.TestCase):
    repin = fixtures.ObjectiveRunTests.repin

    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        fixtures.ObjectiveRunTests.setUp(self)
        labels = Fixture(self.root).labels
        pilot = make_pilot(self.manifest, labels, train_per_class=5, validation_per_class=1)
        path = Path(self.d['selection_path'])
        path.write_text(json.dumps(pilot))
        observed_path = self.root/'observed.json'
        observed_path.write_text(json.dumps(make_confirmation(self.manifest, labels, pilot, validation_per_class=1)))
        bundle = make_exposure_bundle(self.manifest, labels, [(observed_path, file_sha(observed_path))])
        Path(self.d['exposure_bundle_path']).write_text(json.dumps(bundle))
        fresh = make_ledger_confirmation(self.manifest, labels, pilot, bundle['ledger'], validation_per_class=1, seed=91)
        self.d.update(selection_sha256=pilot['selection_sha256'], ledger_sha256=bundle['ledger']['ledger_sha256'],
            confirmation_selection_sha256=fresh['selection_sha256'],
            evaluation=dict(pooling='mean', encoder='context', classifier_seeds=[29], probe_steps=3,
                            learning_rates=[.001], weight_decays=[.01], ridge_alphas=[.1], neighbor_counts=[1]))
        for row in self.d['sources']:
            row['sha256'] = file_sha(row['path'])
        self.repin()
        self.run.training()
        p = patch('jepa_lab.embeddings.CIFAR10Binary', Fixture)
        p.start()
        self.addCleanup(p.stop)
        d, m, ref, _, labels, _ = self.run.inputs()
        for key in objective_features.source_specs(d):
            objective_features.feature(self.run, d, m, ref, labels, key, 'train')
        Fixture.accessed.clear()
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.features = Path(folder.name)/'features.pt'
        self.output = Path(folder.name)/'report.json'

    def test_declared_cli_end_to_end_and_replay_without_images_encoding_or_fitting(self):
        with patch.object(Fixture, '__getitem__', side_effect=AssertionError('declaration images')):
            declared = declare_protocol(self.run)
        protocol = self.output.with_name('protocol.json')
        protocol.write_text(json.dumps(declared))
        args = ['spatial-run', '--root', str(self.root), '--directory', str(self.directory),
            '--objective-protocol', str(self.protocol), '--objective-sha256', self.run.expected_sha,
            '--diagnostic-protocol', str(protocol), '--diagnostic-sha256', file_sha(protocol),
            '--features', str(self.features), '--output', str(self.output)]
        with patch('sys.argv', args), patch('jepa_lab.analysis.spatial_sources.CIFAR10Binary', Fixture), \
                contextlib.redirect_stdout(io.StringIO()) as stream:
            main()
        self.assertFalse(json.loads(stream.getvalue())['report_cache_reused'])
        saved = json.loads(self.output.read_text())
        self.assertEqual(10, len(saved['report']['sources']))
        self.assertEqual(5, len(saved['report']['sources']['pixels']['score']['folds']))
        self.assertEqual(10, len(saved['report']['paired']))
        before = self.output.read_bytes(), self.features.read_bytes()
        with patch('sys.argv', args+['--reuse-only']), \
                patch.object(Fixture, '__getitem__', side_effect=AssertionError('images')), \
                patch('jepa_lab.analysis.spatial_sources.paired_embeddings', side_effect=AssertionError('encoding')), \
                patch('jepa_lab.analysis.spatial_evaluation.fit_ridge', side_effect=AssertionError('fit')), \
                contextlib.redirect_stdout(io.StringIO()) as stream:
            main()
        self.assertTrue(json.loads(stream.getvalue())['report_cache_reused'])
        self.assertEqual(before, (self.output.read_bytes(), self.features.read_bytes()))

    def test_failed_report_save_recovers_from_features_without_new_images_and_refuses_recipe_changes(self):
        declared = declare_protocol(self.run)
        from jepa_lab.analysis.spatial_cache import cached_features
        with patch('jepa_lab.analysis.spatial_sources.CIFAR10Binary', Fixture):
            cached_features(self.run, self.features)
        with patch.object(Fixture, '__getitem__', side_effect=AssertionError('new images')), \
                patch('jepa_lab.analysis.spatial_run.os.link', side_effect=OSError('report disk error')), \
                self.assertRaisesRegex(OSError, 'disk error'):
            run_analysis(self.run, self.features, self.output, declared)
        self.assertFalse(self.output.exists())
        self.assertTrue(self.features.exists())
        with patch.object(Fixture, '__getitem__', side_effect=AssertionError('new images')):
            run_analysis(self.run, self.features, self.output, declared)
        before = self.output.read_bytes()
        declared['evaluation']['alpha'] = 1.
        with self.assertRaisesRegex(ValueError, 'declared recipe'):
            run_analysis(self.run, self.features, self.output, declared)
        self.assertEqual(before, self.output.read_bytes())

    def test_wrong_evaluation_is_rejected_before_images_or_cache_creation(self):
        declared = declare_protocol(self.run)
        declared['evaluation']['alpha'] = 1.
        with patch.object(Fixture, '__getitem__', side_effect=AssertionError('images')), \
                patch('jepa_lab.analysis.spatial_run.cached_features', side_effect=AssertionError('cache')), \
                self.assertRaisesRegex(ValueError, 'declared recipe'):
            run_analysis(self.run, self.features, self.output, declared)
        self.assertFalse(self.features.exists())
        self.assertFalse(self.output.exists())
