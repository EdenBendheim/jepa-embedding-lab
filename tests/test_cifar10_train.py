import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from jepa_lab.cifar10 import make_manifest
from jepa_lab.cifar10_train import atomic_checkpoint, train_cifar10
from jepa_lab.model import ModelConfig


class Fixture:
    accessed = []

    def __init__(self, root, *, train=True):
        self.train = train
        self.labels = [index%10 for index in range(50000 if train else 10000)]
        self.batch_sha256 = {"train" if train else "test": "synthetic-checksum"}

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        self.accessed.append((self.train, index))
        pixels = ((torch.arange(3*32*32)+index)%256).to(torch.uint8).reshape(3,32,32)
        return pixels, self.labels[index]


class PublicTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.manifest = make_manifest(Fixture(self.root), Fixture(self.root, train=False))
        self.path = self.root/"manifest.json"
        self.path.write_text(json.dumps(self.manifest))
        self.patcher = patch("jepa_lab.cifar10_train.CIFAR10Binary", Fixture)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)
        Fixture.accessed.clear()
        self.config = ModelConfig(patch_dim=48, embedding_dim=16, encoder_depth=1)

    def run_training(self, **settings):
        return train_cifar10(self.root, self.path, **(dict(steps=2, batch_size=2, config=self.config) | settings))

    def assert_state_equal(self, left, right):
        if isinstance(left, torch.Tensor):
            self.assertTrue(torch.equal(left, right))
        elif isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys())
            for key in left:
                self.assert_state_equal(left[key], right[key])
        elif isinstance(left, (tuple, list)):
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                self.assert_state_equal(a, b)
        else:
            self.assertEqual(left, right)

    def test_resume_matches_uninterrupted_history_parameters_optimizer_and_rng(self):
        whole_path, split_path = self.root/"whole.pt", self.root/"split.pt"
        whole = self.run_training(steps=4, checkpoint=whole_path)
        first = self.run_training(checkpoint=split_path)
        resumed = self.run_training(resume=split_path, checkpoint=split_path)
        self.assertEqual(whole["history"], first["history"]+resumed["history"])
        self.assertEqual(2, resumed["start_step"])
        self.assertEqual(4, resumed["completed_steps"])
        self.assert_state_equal(torch.load(whole_path, weights_only=True), torch.load(split_path, weights_only=True))
        train = set(self.manifest["partitions"]["train"]["indices"])
        validation = set(self.manifest["partitions"]["validation"]["indices"])
        self.assertTrue(all(is_train and index in train and index not in validation
                            for is_train, index in Fixture.accessed))

    def test_resume_rejects_changed_settings_model_or_manifest_before_sampling(self):
        checkpoint = self.root/"pilot.pt"
        self.run_training(checkpoint=checkpoint)
        original = checkpoint.read_bytes()
        changed_config = ModelConfig(patch_dim=48, embedding_dim=32, encoder_depth=1)
        for changes in (dict(seed=17), dict(batch_size=3), dict(learning_rate=0.002),
                        dict(momentum=0.9), dict(config=changed_config)):
            Fixture.accessed.clear()
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, "Checkpoint"):
                self.run_training(resume=checkpoint, checkpoint=checkpoint, **changes)
            self.assertEqual([], Fixture.accessed)
            self.assertEqual(original, checkpoint.read_bytes())
        # A different valid split protocol is still incompatible with the saved training state.
        changed = make_manifest(Fixture(self.root), Fixture(self.root, train=False), seed=20261004)
        self.path.write_text(json.dumps(changed))
        Fixture.accessed.clear()
        with self.assertRaisesRegex(ValueError, "Checkpoint"):
            self.run_training(resume=checkpoint, checkpoint=checkpoint, seed=20261003)
        self.assertEqual([], Fixture.accessed)
        self.assertEqual(original, checkpoint.read_bytes())

    def test_resume_rejects_runtime_mismatch_and_missing_state(self):
        checkpoint = self.root/"pilot.pt"
        self.run_training(checkpoint=checkpoint)
        state = torch.load(checkpoint, weights_only=True)
        for bad in (state | {"runtime": dict(torch="different-version", cpu_threads=1)},
                    state | {"step": True}, {key:value for key,value in state.items() if key != "sample_rng"}):
            torch.save(bad, checkpoint)
            Fixture.accessed.clear()
            with self.assertRaises(ValueError):
                self.run_training(resume=checkpoint)
            self.assertEqual([], Fixture.accessed)

    def test_failed_checkpoint_save_preserves_previous_file_and_cleans_temporary(self):
        path = self.root/"pilot.pt"
        path.write_bytes(b"previous checkpoint")
        def fail(state, temporary):
            Path(temporary).write_bytes(b"partial new checkpoint")
            raise OSError("simulated disk error")
        with patch("jepa_lab.cifar10_train.torch.save", fail), self.assertRaises(OSError):
            atomic_checkpoint(path, {})
        self.assertEqual(b"previous checkpoint", path.read_bytes())
        self.assertEqual([], list(self.root.glob(".pilot.pt.*")))

    def test_invalid_bounds_or_stale_manifest_never_sample_images(self):
        for settings in (dict(steps=0), dict(steps=1001), dict(steps=True), dict(batch_size=33),
                         dict(momentum=float("nan")), dict(learning_rate=0), dict(seed=True),
                         dict(config=ModelConfig())):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                self.run_training(**settings)
        self.manifest["partitions"]["train"]["indices"][0] = self.manifest["partitions"]["validation"]["indices"][0]
        self.path.write_text(json.dumps(self.manifest))
        with self.assertRaisesRegex(ValueError, "Manifest"):
            self.run_training()
        self.assertEqual([], Fixture.accessed)


if __name__ == "__main__":
    unittest.main()
