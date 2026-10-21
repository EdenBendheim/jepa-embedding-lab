from contextlib import redirect_stdout, redirect_stderr
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from jepa_lab.cifar10 import make_manifest
from jepa_lab.cifar10_train import main, train_cifar10
from jepa_lab.embeddings import main as extract, load_encoder
from jepa_lab.model import ModelConfig
from test_cifar10_train import Fixture
import test_cifar10_train as training_tests


class PositionScaleCLITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_cli_resume_matches_continuous_state_and_rejects_changed_scale_before_sampling(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            manifest = make_manifest(Fixture(root), Fixture(root, train=False))
            path = root/'manifest.json'
            path.write_text(json.dumps(manifest))
            checkpoint, whole = root/'split.pt', root/'whole.pt'
            config = ModelConfig(patch_dim=48, position_encoding='sincos2d', position_scale=.02)
            args = ['train', str(root), '--manifest', str(path), '--steps', '2',
                    '--position-encoding', 'sincos2d', '--position-scale', '.02',
                    '--checkpoint', str(checkpoint)]
            with patch('jepa_lab.cifar10_train.CIFAR10Binary', Fixture), redirect_stdout(io.StringIO()):
                with patch('sys.argv', args):
                    main()
                with patch('sys.argv', args+['--resume', str(checkpoint)]):
                    main()
                train_cifar10(root, path, steps=4, config=config, checkpoint=whole)
                before = checkpoint.read_bytes()
                Fixture.accessed.clear()
                with patch('sys.argv', args+['--resume', str(checkpoint), '--position-scale', '1']), \
                        redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    main()
                self.assertEqual([], Fixture.accessed)
                self.assertEqual(before, checkpoint.read_bytes())
            training_tests.PublicTrainingTests().assert_state_equal(torch.load(whole, weights_only=True),
                                                     torch.load(checkpoint, weights_only=True))
            model, metadata = load_encoder(checkpoint, manifest['manifest_sha256'])
            self.assertEqual(.02, model.config.position_scale)
            self.assertEqual(.02, metadata['config']['position_scale'])

    def test_random_extraction_records_scale_and_refuses_overrides_for_other_representations(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path, output = root/'manifest.json', root/'random.pt'
            path.write_text(json.dumps(make_manifest(Fixture(root), Fixture(root, train=False))))
            args = ['extract', str(root), '--manifest', str(path), '--limit', '3',
                    '--representation', 'random', '--position-encoding', 'sincos2d',
                    '--position-scale', '.02', '--output', str(output)]
            with patch('jepa_lab.embeddings.CIFAR10Binary', Fixture), redirect_stdout(io.StringIO()):
                with patch('sys.argv', args):
                    extract()
                self.assertEqual(.02, torch.load(output, weights_only=True)['feature_metadata']['config']['position_scale'])
                before = output.read_bytes()
                for representation in ('checkpoint', 'pixels'):
                    Fixture.accessed.clear()
                    with patch('sys.argv', args+['--representation', representation]), \
                            redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                        extract()
                    self.assertEqual([], Fixture.accessed)
                    self.assertEqual(before, output.read_bytes())
