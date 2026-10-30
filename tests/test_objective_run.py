from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from jepa_lab.cifar10 import make_manifest
from jepa_lab.exposure import make_exposure_bundle, make_ledger_confirmation
from jepa_lab.model import ModelConfig, config_dict
from jepa_lab.objective_run import ObjectiveRun, code_paths, file_sha
from jepa_lab.selection import make_confirmation, make_pilot
from test_cifar10_train import Fixture


class ObjectiveRunTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        self.directory = self.root / 'run'
        self.manifest = make_manifest(Fixture(self.root), Fixture(self.root, train=False))
        labels = Fixture(self.root).labels
        pilot = make_pilot(self.manifest, labels, train_per_class=1, validation_per_class=1)
        observed = make_confirmation(self.manifest, labels, pilot, validation_per_class=1)
        paths = [self.root / name for name in ('manifest.json', 'pilot.json', 'observed.json', 'bundle.json')]
        for p, value in zip(paths, [self.manifest, pilot, observed]):
            p.write_text(json.dumps(value))
        bundle = make_exposure_bundle(self.manifest, labels, [(paths[2], file_sha(paths[2]))])
        paths[3].write_text(json.dumps(bundle))
        fresh = make_ledger_confirmation(self.manifest, labels, pilot, bundle['ledger'],
                                         validation_per_class=1, seed=91)
        self.d = dict(version=1, kind='context_variance_comparison',
            config=config_dict(ModelConfig(patch_dim=48, embedding_dim=16, encoder_depth=1,
                                           position_encoding='sincos2d')),
            arms=[dict(name='prediction', context_variance_weight=0., context_std_floor=1.),
                  dict(name='context_variance', context_variance_weight=.1, context_std_floor=1.)],
            total_steps=3, chunk_steps=2, checkpoint_every=1, batch_size=2, training_seeds=[29],
            learning_rate=.001, momentum=.99, mask=dict(target_rows=2, target_cols=2, context_fraction=.75),
            runtime=dict(torch=str(torch.__version__), cpu_threads=1),
            manifest_path=str(paths[0]), selection_path=str(paths[1]), exposure_bundle_path=str(paths[3]),
            sources=[dict(path=str(p), sha256=file_sha(p)) for p in [*paths[:2], paths[3], *code_paths()]],
            manifest_sha256=self.manifest['manifest_sha256'], selection_sha256=pilot['selection_sha256'],
            ledger_sha256=bundle['ledger']['ledger_sha256'], confirmation_seed=91, confirmation_per_class=1,
            confirmation_selection_sha256=fresh['selection_sha256'])
        self.protocol = self.root / 'protocol.json'
        self.repin()
        p = patch('jepa_lab.objective_run.CIFAR10Binary', Fixture)
        p.start()
        self.addCleanup(p.stop)
        Fixture.accessed.clear()

    def repin(self):
        self.protocol.write_text(json.dumps(self.d))
        self.run = ObjectiveRun(self.root, self.directory, protocol=self.protocol, expected_sha=file_sha(self.protocol))

    def test_prepare_is_unscored_idempotent_and_does_not_read_images_or_change_rng(self):
        rng = torch.get_rng_state().clone()
        with patch.object(Fixture, '__getitem__', side_effect=AssertionError('image access')):
            result = self.run.prepare()
            before = (self.directory / 'confirmation-selection.json').read_bytes()
            self.assertEqual(result, self.run.prepare(reuse_only=True))
        self.assertEqual(before, (self.directory / 'confirmation-selection.json').read_bytes())
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(10, result['reserved_count'])
        self.assertEqual(20, result['exposure']['observed_validation_count'])
        self.assertFalse(result['models_trained'])
        self.assertFalse(result['confirmation_scored'])
        self.assertNotIn('indices', result)

    def test_missing_or_changed_sources_and_protected_outputs_do_not_replace_evidence(self):
        with self.assertRaisesRegex(ValueError, 'missing'):
            self.run.prepare(reuse_only=True)
        self.assertFalse(self.directory.exists())
        self.run.directory = self.root
        before = self.protocol.read_bytes()
        with self.assertRaisesRegex(ValueError, 'protected'):
            self.run.prepare()
        self.assertEqual(before, self.protocol.read_bytes())
        self.run.directory = self.directory
        original = deepcopy(self.d)
        for mutate in (lambda: self.d['sources'].pop(),
                       lambda: self.d['sources'].append(self.d['sources'][0]),
                       lambda: self.d['sources'][0].update(sha256='0'*64)):
            self.d = deepcopy(original)
            mutate()
            self.repin()
            with self.assertRaises(ValueError):
                self.run.prepare()
            self.assertFalse(self.directory.exists())
        self.d = original
        self.repin()
        self.run.prepare()
        selected = self.directory / 'confirmation-selection.json'
        selected.write_text('{}')
        with self.assertRaisesRegex(ValueError, 'Existing'):
            self.run.prepare()
        self.assertEqual('{}', selected.read_text())

    def test_invalid_contrasts_runtime_and_reservation_fail_before_writes(self):
        original = deepcopy(self.d)
        variants = [dict(training_seeds=[29, 29]), dict(total_steps=True), dict(batch_size=1),
                    dict(runtime=dict(torch='other', cpu_threads=1)), dict(confirmation_seed=True),
                    dict(confirmation_selection_sha256='0'*64), dict(selection_sha256='0'*64),
                    dict(arms=list(reversed(self.d['arms']))),
                    dict(arms=[self.d['arms'][0], self.d['arms'][1] | dict(context_variance_weight=0.)])]
        for changes in variants:
            self.d = original | changes
            self.repin()
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.run.prepare()
            self.assertFalse(self.directory.exists())
        self.d = original
        self.repin()
        self.protocol.write_text(self.protocol.read_text() + ' ')
        with self.assertRaisesRegex(ValueError, 'protocol bytes'):
            self.run.prepare()
