import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from jepa_lab.cifar10 import make_manifest
from jepa_lab.cifar10_train import train_cifar10
from jepa_lab.embeddings import load_encoder
from jepa_lab.model import ModelConfig
from test_cifar10_train import Fixture


class TrainingMaskTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def setUp(self):
        folder = tempfile.TemporaryDirectory(); self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        self.manifest = make_manifest(Fixture(self.root), Fixture(self.root, train=False))
        self.path = self.root/"manifest.json"; self.path.write_text(json.dumps(self.manifest))
        self.config = ModelConfig(patch_dim=48, embedding_dim=16, encoder_depth=1)
        mocked = patch("jepa_lab.cifar10_train.CIFAR10Binary", Fixture)
        mocked.start(); self.addCleanup(mocked.stop)

    def train(self, **changes):
        return train_cifar10(self.root, self.path, **(dict(steps=2, batch_size=2, config=self.config,
                                                         target_rows=3, target_cols=4, context_fraction=0.5) | changes))

    def test_custom_mask_resume_reproduces_history_weights_and_rng(self):
        a, b = self.root/"whole.pt", self.root/"split.pt"
        whole = self.train(steps=4, checkpoint=a)
        first = self.train(checkpoint=b)
        second = self.train(resume=b, checkpoint=b)
        self.assertEqual(whole["history"], first["history"] + second["history"])
        self.assertEqual(dict(target_rows=3, target_cols=4, context_fraction=0.5,
                              context_patches=26, target_patches=12, unused_patches=26), whole["mask_geometry"])
        left, right = torch.load(a, weights_only=True), torch.load(b, weights_only=True)
        self.assertTrue(all(torch.equal(left["model"][key], right["model"][key]) for key in left["model"]))
        self.assertEqual(left["sample_rng"], right["sample_rng"])
        self.assertTrue(torch.equal(left["torch_rng"], right["torch_rng"]))
        self.assertEqual(whole["settings"], load_encoder(a, self.manifest["manifest_sha256"])[1]["training_settings"])

    def test_changed_geometry_and_malformed_masks_fail_before_sampling(self):
        path = self.root/"model.pt"; self.train(checkpoint=path)
        original = path.read_bytes()
        for changes in (dict(target_rows=4), dict(context_fraction=0.75), dict(target_rows=0),
                        dict(target_rows=True), dict(target_cols=9), dict(context_fraction=True),
                        dict(target_rows=8, target_cols=8), dict(context_fraction=float("nan"))):
            Fixture.accessed.clear()
            with self.subTest(changes=changes), self.assertRaises(ValueError): self.train(resume=path, checkpoint=path, **changes)
            self.assertEqual([], Fixture.accessed)
            self.assertEqual(original, path.read_bytes())
        state = torch.load(path, weights_only=True)
        state["settings"]["mask"]["target_rows"] = 9
        torch.save(state, path)
        with self.assertRaises(ValueError): load_encoder(path, self.manifest["manifest_sha256"])

    def test_default_recipe_keeps_legacy_checkpoint_identity(self):
        path = self.root/"default.pt"
        first = self.train(checkpoint=path, target_rows=2, target_cols=2, context_fraction=0.75)
        self.assertNotIn("mask", first["settings"])
        resumed = self.train(resume=path, target_rows=2, target_cols=2, context_fraction=0.75)
        self.assertEqual(2, resumed["start_step"])
        self.assertEqual(45, resumed["mask_geometry"]["context_patches"])
        self.assertEqual(4, resumed["mask_geometry"]["target_patches"])
