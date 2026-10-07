from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

from jepa_lab.cifar10 import make_manifest
from jepa_lab.embeddings import extract_cifar10
from jepa_lab.selection import make_pilot, validate_pilot
from test_embeddings import Fixture


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.train, self.test = Fixture("unused"), Fixture("unused", train=False)
        self.manifest = make_manifest(self.train, self.test)
        self.selection = make_pilot(self.manifest, self.train.labels, train_per_class=2, validation_per_class=1)

    def test_balanced_disjoint_reproducible_selection_preserves_global_rng(self):
        state = random.getstate()
        self.assertEqual(self.selection, make_pilot(self.manifest, self.train.labels, train_per_class=2, validation_per_class=1))
        self.assertEqual(state, random.getstate())
        selected = self.selection["partitions"]
        self.assertEqual({"train", "validation"}, set(selected))
        for name,count in (("train",2),("validation",1)):
            indices = selected[name]["indices"]
            self.assertEqual(Counter({i:count for i in range(10)}), Counter(self.train.labels[i] for i in indices))
            self.assertTrue(set(indices) <= set(self.manifest["partitions"][name]["indices"]))
        self.assertFalse(set(selected["train"]["indices"]) & set(selected["validation"]["indices"]))
        self.assertNotEqual(self.selection, make_pilot(self.manifest, self.train.labels, train_per_class=2, validation_per_class=1, seed=42))

    def test_edited_selection_counts_and_identity_are_rejected(self):
        for changes in (dict(seed=True), dict(train_per_class=0), dict(validation_per_class=101)):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                make_pilot(self.manifest, self.train.labels, **changes)
        edited = deepcopy(self.selection); edited["partitions"]["train"]["indices"].reverse()
        for value in (edited, self.selection | dict(manifest_sha256="wrong"), {}):
            with self.assertRaises(ValueError):
                validate_pilot(value, self.manifest, self.train.labels)

    def test_extraction_uses_the_entire_selection_and_rejects_test_or_edits_before_sampling(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); m, s = root/"manifest.json", root/"selection.json"
            m.write_text(json.dumps(self.manifest)); s.write_text(json.dumps(self.selection))
            with patch("jepa_lab.embeddings.CIFAR10Binary", Fixture):
                Fixture.accessed.clear()
                report = extract_cifar10(root, m, selection_path=s, representation="pixels")
                self.assertEqual(20, report["count"])
                self.assertEqual(self.selection["selection_sha256"], report["selection_sha256"])
                self.assertEqual([(True,i) for i in self.selection["partitions"]["train"]["indices"]], Fixture.accessed)
                Fixture.accessed.clear()
                for settings in (dict(partition="test"), dict(limit=4)):
                    with self.assertRaises(ValueError):
                        extract_cifar10(root, m, selection_path=s, representation="pixels", **settings)
                s.write_text(json.dumps(self.selection | dict(seed=1)))
                with self.assertRaises(ValueError):
                    extract_cifar10(root, m, selection_path=s, representation="pixels")
                self.assertEqual([], Fixture.accessed)
