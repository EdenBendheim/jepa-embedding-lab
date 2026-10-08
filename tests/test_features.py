import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from jepa_lab.cifar10 import make_manifest
from jepa_lab.embeddings import extract_cifar10
from jepa_lab.features import load_features
from jepa_lab.selection import make_pilot
from test_embeddings import Fixture


class FeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        folder = tempfile.TemporaryDirectory(); self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        self.labels = Fixture(self.root).labels
        self.manifest = make_manifest(Fixture(self.root), Fixture(self.root, train=False))
        self.selection = make_pilot(self.manifest, self.labels, train_per_class=1, validation_per_class=1)
        m, s = self.root/"manifest.json", self.root/"selection.json"
        m.write_text(json.dumps(self.manifest)); s.write_text(json.dumps(self.selection))
        self.path = self.root/"pixels.pt"
        with patch("jepa_lab.embeddings.CIFAR10Binary", Fixture):
            extract_cifar10(self.root, m, representation="pixels", selection_path=s, output=self.path)
        self.original = torch.load(self.path, weights_only=True)

    def load(self, **changes):
        return load_features(self.path, **(dict(manifest=self.manifest, selection=self.selection,
            training_labels=self.labels, partition="train") | changes))

    def test_verified_rows_labels_and_file_identity_are_preserved(self):
        features = self.load()
        self.assertEqual((10,3072), tuple(features.features.shape))
        self.assertEqual(tuple(self.selection["partitions"]["train"]["indices"]), features.indices)
        self.assertEqual(self.original["labels"].tolist(), features.labels.tolist())
        self.assertEqual(self.selection["selection_sha256"], features.selection_sha256)
        self.assertEqual(features.artifact_sha256, self.load().artifact_sha256)
        self.assertNotIn("features", features.summary())
        features.features.zero_()
        self.assertTrue(torch.equal(self.original["features"], self.load().features))

    def test_wrong_partition_namespace_indices_labels_and_content_are_rejected(self):
        changes = [dict(partition="test"), dict(official_split="test"), dict(selection_sha256="wrong"),
                   dict(indices=self.original["indices"][::-1]), dict(labels=self.original["labels"].roll(1)),
                   dict(features_sha256="wrong"), dict(features=self.original["features"].double()),
                   dict(features=torch.full_like(self.original["features"], float("nan")))]
        for changed in changes:
            with self.subTest(fields=list(changed)), self.assertRaises(ValueError):
                torch.save(self.original | changed, self.path); self.load()
        torch.save(self.original, self.path)
        with self.assertRaises(ValueError): self.load(partition="test")
        with self.assertRaises(ValueError): self.load(partition="validation")

    def test_invalid_normalization_architecture_and_runtime_are_rejected(self):
        for changed in (dict(feature_metadata=self.original["feature_metadata"] | dict(normalization="all-data z-score")),
                        dict(feature_metadata=dict(representation="random", config={})),
                        dict(runtime=dict(torch="other-version", cpu_threads=1)),
                        dict(extraction_settings=dict(limit=999, batch_size=1))):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                torch.save(self.original | changed, self.path); self.load()
        self.path.write_bytes(b"not a feature artifact")
        with self.assertRaises(ValueError): self.load()
