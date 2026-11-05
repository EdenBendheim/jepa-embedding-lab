import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from jepa_lab.analysis.color_cache import cached_audit, code_fingerprints
from test_cifar10_train import Fixture
import test_color_audit as fixtures


class ColorCacheTests(unittest.TestCase):
    setUp = fixtures.ColorAuditTests.setUp
    repin = fixtures.ColorAuditTests.repin
    setUpClass = fixtures.ColorAuditTests.setUpClass

    def output(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        return Path(folder.name) / 'report.json'

    def test_round_trip_rechecks_sources_without_images_or_fitting_and_refuses_changes(self):
        output = self.output()
        with patch('jepa_lab.analysis.color_audit.CIFAR10Binary', Fixture):
            report, reused = cached_audit(self.run, output)
        self.assertFalse(reused)
        before = output.read_bytes()
        with patch.object(Fixture, '__getitem__', side_effect=AssertionError('images')), \
                patch('jepa_lab.analysis.crossfit.fit_reconstruction', side_effect=AssertionError('fit')):
            again, reused = cached_audit(self.run, output, reuse_only=True)
        self.assertTrue(reused)
        self.assertEqual(report, again)
        self.assertEqual(before, output.read_bytes())
        changed = code_fingerprints() | {'analysis/color_audit.py': '0' * 64}
        with patch('jepa_lab.analysis.color_cache.code_fingerprints', return_value=changed), \
                self.assertRaisesRegex(ValueError, 'differs'):
            cached_audit(self.run, output)
        self.assertEqual(before, output.read_bytes())
        saved = json.loads(before)
        saved['report']['count'] = 999
        output.write_text(json.dumps(saved))
        corrupt = output.read_bytes()
        with self.assertRaisesRegex(ValueError, 'differs'):
            cached_audit(self.run, output)
        self.assertEqual(corrupt, output.read_bytes())

    def test_failed_write_leaves_no_partial_report_and_protected_paths_are_refused(self):
        output = self.output()
        with self.assertRaisesRegex(ValueError, 'missing'):
            cached_audit(self.run, output, reuse_only=True)
        with patch('jepa_lab.analysis.color_audit.CIFAR10Binary', Fixture), \
                patch('jepa_lab.analysis.color_cache.os.link', side_effect=OSError('disk error')), \
                self.assertRaisesRegex(OSError, 'disk error'):
            cached_audit(self.run, output)
        self.assertEqual([], list(output.parent.iterdir()))
        for path in (self.directory/'report.json', self.protocol, self.root/'data.json'):
            with self.assertRaisesRegex(ValueError, 'cannot replace'):
                cached_audit(self.run, path)
