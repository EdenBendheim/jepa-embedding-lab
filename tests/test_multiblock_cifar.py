from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import torch

from jepa_lab.cifar10 import make_manifest
from jepa_lab.model import ModelConfig
from jepa_lab.variants.cifar_batches import CifarTrainingData
from jepa_lab.variants.multiblock_train import main, train_multiblock
from test_cifar10_train import Fixture
import test_multiblock_train


class MultiBlockCifarTests(unittest.TestCase):
    assert_state_equal = test_multiblock_train.MultiBlockTrainTests.assert_state_equal
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.path = self.root/'manifest.json'
        self.manifest = make_manifest(Fixture(self.root), Fixture(self.root, train=False))
        self.path.write_text(json.dumps(self.manifest))
        fixture = patch('jepa_lab.variants.cifar_batches.CIFAR10Binary', Fixture)
        fixture.start()
        self.addCleanup(fixture.stop)
        Fixture.accessed.clear()

    def test_manifest_and_split_checked_before_image_reads(self):
        data = CifarTrainingData(self.root, self.path)
        self.assertEqual([], Fixture.accessed)
        train = data.indices[:2]
        validation = self.manifest['partitions']['validation']['indices'][0]
        for invalid in ((train[0], validation), (train[0], train[0]), (True,), ()):
            with self.assertRaisesRegex(ValueError, 'fixed CIFAR-10 training split'):
                data.read(invalid)
        self.assertEqual([], Fixture.accessed)
        images, labels = data.read(train)
        self.assertEqual((2, 3, 32, 32), tuple(images.shape))
        self.assertEqual([index % 10 for index in train], labels.tolist())
        self.assertEqual([(True, index) for index in train], Fixture.accessed)
        self.manifest['partitions']['train']['indices'][0] = validation
        self.path.write_text(json.dumps(self.manifest))
        with self.assertRaisesRegex(ValueError, 'manifest differs'):
            CifarTrainingData(self.root, self.path)
        self.assertEqual(2, len(Fixture.accessed))

    def test_sampling_replays_observed_rows_without_images_or_global_rng_changes(self):
        data = CifarTrainingData(self.root, self.path)
        original = random.getstate()
        historical = random.Random(29)
        history = [dict(train_indices=historical.sample(list(data.indices), 4)) for _ in range(3)]
        replay = data.replay_sampling(history, seed=29, batch_size=4)
        self.assertEqual(historical.getstate(), replay.getstate())
        history[-1]['train_indices'].reverse()
        with self.assertRaisesRegex(ValueError, 'image order'):
            data.replay_sampling(history, seed=29, batch_size=4)
        self.assertEqual(original, random.getstate())
        self.assertEqual([], Fixture.accessed)

    def test_public_training_exact_resume_and_training_split_only(self):
        data = CifarTrainingData(self.root, self.path)
        whole, split = self.root/'whole.pt', self.root/'split.pt'
        config = ModelConfig(patch_dim=48, embedding_dim=16, encoder_depth=1)
        options = dict(training_data=data, config=config, batch_size=2, blocks=4)
        initial_rng = torch.get_rng_state().clone()
        full = train_multiblock(steps=3, checkpoint=whole, **options)
        first = train_multiblock(steps=1, checkpoint=split, **options)
        last = train_multiblock(steps=2, checkpoint=split, resume=split, **options)
        self.assertEqual(full['history'], first['history']+last['history'])
        self.assert_state_equal(torch.load(whole, weights_only=True), torch.load(split, weights_only=True))
        self.assertTrue(torch.equal(initial_rng, torch.get_rng_state()))
        self.assertEqual(data.manifest['manifest_sha256'], full['manifest_sha256'])
        scheduled = [index for row in full['history'] for index in row['train_indices']]
        self.assertEqual([(True, index) for index in scheduled*2], Fixture.accessed)

    def test_public_resume_rejects_changed_manifest_image_order_and_rng_before_sampling(self):
        data = CifarTrainingData(self.root, self.path)
        path = self.root/'cifar.pt'
        options = dict(training_data=data, steps=1, batch_size=2, checkpoint=path)
        train_multiblock(**options)
        original = path.read_bytes()
        Fixture.accessed.clear()
        changed = make_manifest(Fixture(self.root), Fixture(self.root, train=False), seed=31)
        self.path.write_text(json.dumps(changed))
        other = CifarTrainingData(self.root, self.path)
        with self.assertRaisesRegex(ValueError, 'Checkpoint'):
            train_multiblock(**(options | dict(resume=path, training_data=other)))
        for field in ('history', 'data_rng'):
            state = torch.load(io.BytesIO(original), weights_only=True)
            if field == 'history':
                state[field][0]['train_indices'].reverse()
            else:
                state[field] = random.Random(1).getstate()
            torch.save(state, path)
            before = path.read_bytes()
            with self.assertRaisesRegex(ValueError, 'image order'):
                train_multiblock(resume=path, **options)
            self.assertEqual(before, path.read_bytes())
        self.assertEqual([], Fixture.accessed)

    def test_cli_explicit_public_inputs_resume_and_input_path_protection(self):
        dataset = self.root/'dataset'
        checkpoint, metrics = self.root/'run.pt', self.root/'metrics.json'
        args = ['multi-block', '--cifar-root', str(dataset), '--manifest', str(self.path),
                '--steps', '1', '--checkpoint', str(checkpoint)]
        original = self.path.read_bytes()
        for invalid in (args[:3], args+['--output', str(self.path)],
                        args+['--output', str(dataset/'data_batch_1.bin')]):
            with patch('sys.argv', invalid), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main()
        self.assertEqual([], Fixture.accessed)
        self.assertEqual(original, self.path.read_bytes())
        with patch('sys.argv', args+['--output', str(metrics)]), redirect_stdout(io.StringIO()):
            main()
        with patch('sys.argv', args+['--resume', str(checkpoint), '--output', str(metrics)]), redirect_stdout(io.StringIO()):
            main()
        result = json.loads(metrics.read_text())
        self.assertEqual('CIFAR-10', result['dataset'])
        self.assertEqual(2, result['completed_step'])
        self.assertEqual(8, len(Fixture.accessed))

    def test_frozen_encoder_validates_exact_checkpoint_and_replays_without_image_access(self):
        from jepa_lab.embeddings import image_embeddings
        from jepa_lab.variants.multiblock_features import load_cifar_encoder

        data = CifarTrainingData(self.root, self.path)
        checkpoint = self.root/'frozen.pt'
        result = train_multiblock(training_data=data, steps=1, checkpoint=checkpoint)
        original = checkpoint.read_bytes()
        Fixture.accessed.clear()
        rng = torch.get_rng_state().clone()
        model, metadata, history = load_cifar_encoder(checkpoint, data)
        self.assertEqual(result['history'], history)
        self.assertEqual(1, metadata['completed_steps'])
        self.assertFalse(model.training)
        self.assertFalse(any(parameter.requires_grad for parameter in model.parameters()))
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        embeddings = image_embeddings(model, torch.zeros(2, 3, 32, 32, dtype=torch.uint8))
        self.assertEqual((2, 32), tuple(embeddings.shape))
        self.assertTrue(torch.isfinite(embeddings).all())
        with self.assertRaisesRegex(ValueError, 'checksum'):
            load_cifar_encoder(checkpoint, data, expected_sha256='0'*64)

        for field in ('kind', 'manifest_sha256', 'history', 'model'):
            state = torch.load(io.BytesIO(original), weights_only=True)
            if field == 'history':
                state[field][0]['train_indices'].reverse()
            elif field == 'model':
                next(iter(state[field].values())).fill_(float('nan'))
            else:
                state[field] = 'changed'
            torch.save(state, checkpoint)
            with self.assertRaises(ValueError):
                load_cifar_encoder(checkpoint, data)
        self.assertEqual([], Fixture.accessed)

    def test_matched_pair_reuses_completed_arm_after_interruption(self):
        from jepa_lab.variants.multiblock_pair import pair_spec, train_pair, verify_pair

        data = CifarTrainingData(self.root, self.path)
        folder = self.root/'pair'
        spec = pair_spec(data, steps=2, batch_size=2)
        def interrupt(**options):
            if options['blocks'] == 4:
                raise RuntimeError('interrupted second arm')
            return train_multiblock(**options)
        with patch('jepa_lab.variants.multiblock_pair.train_multiblock', side_effect=interrupt), \
                self.assertRaisesRegex(RuntimeError, 'second arm'):
            train_pair(data, folder, spec)
        first = (folder/'one_4x4.pt').read_bytes()
        self.assertEqual(4, len(Fixture.accessed))
        receipt = train_pair(data, folder, spec)
        self.assertEqual(first, (folder/'one_4x4.pt').read_bytes())
        self.assertEqual(Fixture.accessed[:4], Fixture.accessed[4:])
        arms, verified = verify_pair(data, folder, spec)
        self.assertEqual(receipt, verified)
        self.assertEqual(8, len(Fixture.accessed))
        self.assertEqual([1, 4], [arm[2][0]['predictor_passes'] for arm in arms.values()])
        with self.assertRaisesRegex(ValueError, 'declared training recipe'):
            train_pair(data, folder, pair_spec(data, steps=3, batch_size=2))
        self.assertEqual(8, len(Fixture.accessed))
