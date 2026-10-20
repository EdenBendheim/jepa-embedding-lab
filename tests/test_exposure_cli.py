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
from jepa_lab.exposure import make_exposure_bundle, load_exposure_bundle, main
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
