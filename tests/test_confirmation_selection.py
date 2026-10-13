from collections import Counter
from copy import deepcopy
import random
import unittest

from jepa_lab.cifar10 import make_manifest
from jepa_lab.selection import document_hash, make_confirmation, make_pilot, validate_pilot
from test_embeddings import Fixture


class ConfirmationSelectionTests(unittest.TestCase):
    def setUp(self):
        self.labels = Fixture("unused").labels
        self.manifest = make_manifest(Fixture("unused"), Fixture("unused", train=False))
        self.reference = make_pilot(self.manifest, self.labels, train_per_class=2, validation_per_class=2)
        self.older = make_pilot(self.manifest, self.labels, validation_per_class=1, seed=29)

    def build(self, **changes):
        return make_confirmation(self.manifest, self.labels, self.reference,
                                 **(dict(exclusions=(self.older,), validation_per_class=3) | changes))

    def test_balanced_unseen_rows_preserve_training_and_rng(self):
        state = random.getstate()
        selected = self.build()
        self.assertEqual(selected, self.build())
        self.assertEqual(state, random.getstate())
        self.assertEqual(selected["partitions"]["train"], self.reference["partitions"]["train"])
        rows = selected["partitions"]["validation"]["indices"]
        self.assertEqual(Counter({i:3 for i in range(10)}), Counter(self.labels[i] for i in rows))
        excluded = {i for s in (self.reference, self.older) for i in s["partitions"]["validation"]["indices"]}
        self.assertFalse(set(rows) & excluded)
        self.assertTrue(set(rows) <= set(self.manifest["partitions"]["validation"]["indices"]))
        self.assertEqual(selected, validate_pilot(selected, self.manifest, self.labels))
        selected["reference_selection"]["seed"] = 0
        self.assertNotEqual(0, self.reference["seed"])

    def test_rehashed_edits_nested_sources_and_invalid_settings_fail(self):
        selected = self.build()
        for key in ("validation", "train"):
            edited = deepcopy(selected)
            edited["partitions"][key]["indices"].reverse()
            edited["selection_sha256"] = document_hash(edited, "selection_sha256")
            with self.assertRaises(ValueError): validate_pilot(edited, self.manifest, self.labels)
        for changes in (dict(seed=True), dict(validation_per_class=0), dict(validation_per_class=101),
                        dict(exclusions=(self.reference,)), dict(exclusions=(selected,)),
                        dict(exclusions=(self.older,) * 9)):
            with self.subTest(changes=list(changes)), self.assertRaises(ValueError): self.build(**changes)
        bad = deepcopy(self.older); bad["seed"] += 1
        with self.assertRaises(ValueError): self.build(exclusions=(bad,))

    def test_exhausted_fixed_validation_split_is_rejected(self):
        tiny = deepcopy(self.manifest)
        tiny["partitions"]["validation"]["indices"] = self.reference["partitions"]["validation"]["indices"]
        tiny["manifest_sha256"] = document_hash(tiny, "manifest_sha256")
        used = make_pilot(tiny, self.labels, train_per_class=1, validation_per_class=2)
        with self.assertRaisesRegex(ValueError, "Not enough unseen"):
            make_confirmation(tiny, self.labels, used, validation_per_class=1)
