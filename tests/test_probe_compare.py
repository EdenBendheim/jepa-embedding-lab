from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from jepa_lab.compare import compare_features
from jepa_lab.probe_compare import compare_saved_probes
from jepa_lab.probes import ProbeConfig
from test_compare import sets


class SavedPairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def test_saved_errors_reproduce_without_fitting_and_reject_unpaired_sources(self):
        data = sets()
        train, validation = data["random"]
        data["random"] = train, replace(validation, features=validation.features.roll(1, 0))
        with tempfile.TemporaryDirectory() as directory:
            original = compare_features(data, (ProbeConfig(steps=30),), probe_dir=Path(directory), ridge_alphas=[0.1])
            for head in ("adamw", "ridge"):
                reports = original["representations"]
                paths = [reports[name]["ridge" if head == "ridge" else "selected_probe_artifact"] for name in ("random", "checkpoint")]
                if head == "ridge": paths = [item["selected_probe_artifact"] for item in paths]
                with patch("jepa_lab.probes.fit_probe", side_effect=AssertionError("refit")), patch("jepa_lab.ridge.fit_ridge", side_effect=AssertionError("refit")):
                    result = compare_saved_probes(data["random"], data["checkpoint"], *(item["path"] for item in paths))
                self.assertEqual(original["paired_validation"][head]["checkpoint_vs_random"], result["paired_validation"])
                self.assertFalse(result["refitted"])
                self.assertFalse(result["test_evaluated"])
            for field, value in (("indices", tuple(reversed(validation.indices))), ("selection_sha256", "d"*64)):
                pair = data["checkpoint"]
                changed = (replace(pair[0], selection_sha256=value), replace(pair[1], selection_sha256=value)) if field == "selection_sha256" else (pair[0], replace(pair[1], indices=value))
                with patch("jepa_lab.probe_compare.load_probe", side_effect=AssertionError("scored mismatched sources")), self.assertRaises(ValueError):
                    compare_saved_probes(data["random"], changed, "unused", "unused")
