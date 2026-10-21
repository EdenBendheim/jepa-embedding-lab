import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from jepa_lab.exposure import make_exposure_bundle, make_ledger_confirmation
import test_exposure as fixtures


class ScalePreparationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_pinned_unscored_reservation_replays_and_failures_preserve_evidence(self):
        spec = importlib.util.spec_from_file_location('scale_preparation',
            Path(__file__).resolve().parents[1]/'experiments/prepare_scale_20261020.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        fixture = fixtures.ExposureTests()
        fixture.setUp()
        sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            manifest, pilot, observed, bundle_path, protocol = [root/name for name in
                ('manifest.json', 'pilot.json', 'observed.json', 'bundle.json', 'protocol.json')]
            for path, document in ((manifest, fixture.manifest), (pilot, fixture.pilot), (observed, fixture.fresh)):
                path.write_text(json.dumps(document))
            bundle = make_exposure_bundle(fixture.manifest, fixture.labels, [(observed, sha(observed))])
            bundle_path.write_text(json.dumps(bundle))
            fresh = make_ledger_confirmation(fixture.manifest, fixture.labels, fixture.pilot,
                                              bundle['ledger'], validation_per_class=1, seed=91)
            d = dict(manifest_path=str(manifest), selection_path=str(pilot), exposure_bundle_path=str(bundle_path),
                     sources=[dict(path=str(p), sha256=sha(p)) for p in (manifest, pilot, bundle_path)],
                     reused_scale_one_checkpoints={}, manifest_sha256=fixture.manifest['manifest_sha256'],
                     ledger_sha256=bundle['ledger']['ledger_sha256'], confirmation_selection_sha256=fresh['selection_sha256'],
                     confirmation_per_class=1, confirmation_seed=91, embedding_dim=16, heads=4,
                     encoder_depth=1, predictor_depth=1, position_encoding='sincos2d', position_scales=[.02, 1.])
            protocol.write_text(json.dumps(d))
            output = root/'reserved.json'
            with patch.object(module, 'CIFAR10Binary', fixtures.Fixture), \
                    patch.object(fixtures.Fixture, '__getitem__', side_effect=AssertionError('raw image read')):
                result = module.prepare(root, protocol, sha(protocol), output)
                before = output.read_bytes()
                self.assertEqual(10, result['reserved_count'])
                self.assertEqual(result['trainable_parameters'][0], result['trainable_parameters'][1])
                self.assertFalse(result['confirmation_scored'])
                self.assertEqual(result, module.prepare(root, protocol, sha(protocol), output))
                self.assertEqual(before, output.read_bytes())
                with self.assertRaisesRegex(ValueError, 'overwrite'):
                    module.prepare(root, protocol, sha(protocol), observed)
                output.write_text('{}')
                with self.assertRaisesRegex(ValueError, 'Existing'):
                    module.prepare(root, protocol, sha(protocol), output)
                self.assertEqual('{}', output.read_text())
                output.write_bytes(before)
                observed.write_text(observed.read_text()+' ')
                with self.assertRaisesRegex(ValueError, 'bytes differ'):
                    module.prepare(root, protocol, sha(protocol), output)
                self.assertEqual(before, output.read_bytes())
                with self.assertRaisesRegex(ValueError, 'protocol bytes'):
                    module.prepare(root, protocol, '0'*64, output)
