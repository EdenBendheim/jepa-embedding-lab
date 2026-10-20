from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
from pathlib import Path
import hashlib
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

from jepa_lab.cifar10 import make_manifest
from jepa_lab.exposure import (make_exposure_bundle, load_exposure_bundle, main,
                               extend_exposure_bundle, make_ledger_confirmation)
from jepa_lab.selection import make_pilot
from test_embeddings import Fixture


class ExposureFileTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(); self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.labels = Fixture('unused').labels
        self.manifest = make_manifest(Fixture('unused'), Fixture('unused', train=False))
        self.source = self.root / 'pilot.json'
        self.source.write_text(json.dumps(make_pilot(self.manifest, self.labels)))
        self.pin = hashlib.sha256(self.source.read_bytes()).hexdigest()

    def test_pins_detect_changed_bytes_even_when_selection_content_is_unchanged(self):
        bundle = make_exposure_bundle(self.manifest, self.labels, [(self.source, self.pin)])
        path = self.root / 'ledger.json'; path.write_text(json.dumps(bundle))
        self.assertEqual(bundle, load_exposure_bundle(path, self.manifest, self.labels))
        self.source.write_text(self.source.read_text() + '\n')
        with self.assertRaisesRegex(ValueError, 'bytes differ'):
            load_exposure_bundle(path, self.manifest, self.labels)
        self.source.unlink()
        with self.assertRaises(OSError): load_exposure_bundle(path, self.manifest, self.labels)

    def test_cli_build_inspect_and_failure_preserve_sources_and_previous_output(self):
        manifest_path = self.root / 'manifest.json'
        manifest_path.write_text(json.dumps(self.manifest))
        out = self.root / 'bundle.json'
        base = ['exposure', str(self.root), '--manifest', str(manifest_path)]
        args = base + ['--source', str(self.source), self.pin, '--output', str(out)]
        with patch('jepa_lab.exposure.CIFAR10Binary', Fixture), patch.object(sys, 'argv', args), redirect_stdout(StringIO()) as stdout:
            main()
        report = json.loads(stdout.getvalue())
        self.assertNotIn('path', stdout.getvalue()); self.assertNotIn('indices', stdout.getvalue())
        self.assertEqual(1, report['source_file_count'])
        before = out.read_bytes()
        with patch('jepa_lab.exposure.CIFAR10Binary', Fixture), patch.object(sys, 'argv', base + ['--bundle', str(out)]), redirect_stdout(StringIO()):
            main()
        self.assertEqual(before, out.read_bytes())
        for invalid in (base + ['--source', str(self.source), '0'*64, '--output', str(out)],
                        base + ['--source', str(self.source), self.pin, '--output', str(self.source)],
                        base + ['--bundle', str(out), '--output', str(out)]):
            with patch('jepa_lab.exposure.CIFAR10Binary', Fixture), patch.object(sys, 'argv', invalid), redirect_stderr(StringIO()):
                with self.assertRaises(SystemExit): main()
        self.assertEqual(before, out.read_bytes())
        self.assertEqual(self.pin, hashlib.sha256(self.source.read_bytes()).hexdigest())

    def test_extension_retains_original_byte_pins_and_cannot_drop_old_evidence(self):
        bundle = make_exposure_bundle(self.manifest, self.labels, [(self.source, self.pin)])
        old = self.root/'old.json'; old.write_text(json.dumps(bundle))
        pilot = json.loads(self.source.read_text())
        selection = make_ledger_confirmation(self.manifest, self.labels, pilot, bundle['ledger'], validation_per_class=1)
        fresh = self.root/'fresh.json'; fresh.write_text(json.dumps(selection))
        pin = hashlib.sha256(fresh.read_bytes()).hexdigest()
        result = extend_exposure_bundle(old, self.manifest, self.labels, [(fresh, pin)],
                                       expected_ledger_sha256=bundle['ledger']['ledger_sha256'])
        self.assertEqual({str(self.source.resolve()), str(fresh.resolve())}, {r['path'] for r in result['source_files']})
        self.assertTrue(set(bundle['ledger']['observed_validation_indices']) < set(result['ledger']['observed_validation_indices']))
        repeated = extend_exposure_bundle(old, self.manifest, self.labels, [(self.source, self.pin)],
                                         expected_ledger_sha256=bundle['ledger']['ledger_sha256'])
        self.assertEqual(bundle, repeated)
        with self.assertRaisesRegex(ValueError, 'predeclared'):
            extend_exposure_bundle(old, self.manifest, self.labels, [(fresh, pin)], expected_ledger_sha256='0'*64)
        self.source.write_text(self.source.read_text() + '\n')
        with self.assertRaisesRegex(ValueError, 'bytes differ'):
            extend_exposure_bundle(old, self.manifest, self.labels, [(fresh, pin)],
                                   expected_ledger_sha256=bundle['ledger']['ledger_sha256'])

    def test_cli_extension_writes_new_bundle_and_preserves_previous_snapshot(self):
        manifest = self.root/'manifest.json'; manifest.write_text(json.dumps(self.manifest))
        bundle = make_exposure_bundle(self.manifest, self.labels, [(self.source, self.pin)])
        old = self.root/'old.json'; old.write_text(json.dumps(bundle)); before = old.read_bytes()
        fresh = self.root/'fresh.json'
        fresh.write_text(json.dumps(make_pilot(self.manifest, self.labels, seed=31)))
        out = self.root/'extended.json'
        args = ['exposure', str(self.root), '--manifest', str(manifest), '--source', str(fresh),
                hashlib.sha256(fresh.read_bytes()).hexdigest(), '--extend-bundle', str(old),
                '--expected-ledger-sha256', bundle['ledger']['ledger_sha256'], '--output', str(out)]
        with patch('jepa_lab.exposure.CIFAR10Binary', Fixture), patch.object(sys, 'argv', args), redirect_stdout(StringIO()): main()
        self.assertEqual(2, len(load_exposure_bundle(out, self.manifest, self.labels)['source_files']))
        with patch('jepa_lab.exposure.CIFAR10Binary', Fixture), patch.object(sys, 'argv', args+['--output',str(old)]), redirect_stderr(StringIO()):
            with self.assertRaises(SystemExit): main()
        self.assertEqual(before, old.read_bytes())
