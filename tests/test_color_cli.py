import contextlib
import io
import json
import unittest
from unittest.mock import patch

from jepa_lab.analysis.color_cli import declare_protocol, load_protocol, main
from jepa_lab.objective_run import file_sha
from test_cifar10_train import Fixture
import test_color_cache as fixtures


class ColorCLITests(unittest.TestCase):
    setUp = fixtures.ColorCacheTests.setUp
    repin = fixtures.ColorCacheTests.repin
    setUpClass = fixtures.ColorCacheTests.setUpClass
    output = fixtures.ColorCacheTests.output

    def test_cli_requires_matching_declared_sources_then_replays_without_images_or_fitting(self):
        output = self.output()
        protocol = output.with_name('protocol.json')
        declared = declare_protocol(self.run)
        protocol.write_text(json.dumps(declared))
        args = ['color-audit', '--root', str(self.root), '--directory', str(self.directory),
                '--objective-protocol', str(self.protocol), '--objective-sha256', self.run.expected_sha,
                '--diagnostic-protocol', str(protocol), '--diagnostic-sha256', file_sha(protocol),
                '--output', str(output)]
        with patch('sys.argv', args), patch('jepa_lab.analysis.color_audit.CIFAR10Binary', Fixture), \
                contextlib.redirect_stdout(io.StringIO()) as stream:
            main()
        self.assertFalse(json.loads(stream.getvalue())['cache_reused'])
        before = output.read_bytes()
        with patch('sys.argv', args + ['--reuse-only']), \
                patch.object(Fixture, '__getitem__', side_effect=AssertionError('images')), \
                patch('jepa_lab.analysis.crossfit.fit_reconstruction', side_effect=AssertionError('fit')), \
                contextlib.redirect_stdout(io.StringIO()) as stream:
            main()
        self.assertTrue(json.loads(stream.getvalue())['cache_reused'])
        self.assertEqual(before, output.read_bytes())
        declared['identity']['measurement']['alpha'] = .1
        protocol.write_text(json.dumps(declared))
        args[args.index('--diagnostic-sha256') + 1] = file_sha(protocol)
        with patch('sys.argv', args), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main()
        self.assertEqual(before, output.read_bytes())
        with self.assertRaisesRegex(ValueError, 'checksum'):
            load_protocol(protocol, '0' * 64)
