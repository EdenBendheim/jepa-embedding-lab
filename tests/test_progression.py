from copy import deepcopy
from dataclasses import replace
import unittest
from pathlib import Path
import tempfile
from unittest.mock import patch

import torch

from jepa_lab.progression import compare_progression, validate_progression
from jepa_lab.probes import ProbeConfig, select_probe
from jepa_lab.ridge import select_ridge
from test_compare import sets


def arms():
    first = sets()
    result = {}
    for name, step, fingerprint in (("early", 3, "e"*64), ("later", 30, "f"*64)):
        data = dict(first)
        metadata = first["checkpoint"][0].metadata | dict(completed_steps=step, checkpoint_sha256=fingerprint,
                                                       training_settings=dict(seed=29, batch_size=4))
        data["checkpoint"] = tuple(replace(item, metadata=deepcopy(metadata)) for item in first["checkpoint"])
        result[name] = data
    return result


class ProgressionContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def test_ages_and_exact_shared_baselines(self):
        data = arms()
        self.assertEqual([3, 30], [item["completed_steps"] for item in validate_progression(data, (ProbeConfig(steps=2),))])
        for change in ("age", "duplicate", "seed", "pooling", "indices", "baseline", "baseline_values"):
            data = arms()
            train, validation = data["later"]["checkpoint"]
            metadata = deepcopy(train.metadata)
            if change == "age": metadata["completed_steps"] = 3
            if change == "duplicate": metadata["checkpoint_sha256"] = "e"*64
            if change == "seed": metadata["training_settings"]["seed"] = 31
            if change == "pooling": metadata["pooling"] = "different"
            data["later"]["checkpoint"] = replace(train, metadata=metadata), replace(validation, metadata=metadata)
            if change == "indices":
                data["later"] = {name: tuple(replace(item, indices=tuple(reversed(item.indices))) for item in pair)
                                 for name, pair in data["later"].items()}
            if change.startswith("baseline"):
                pair = data["later"]["random"]
                changed = replace(pair[0], artifact_sha256="d"*64) if change == "baseline" else replace(pair[0], features=pair[0].features + 0.1)
                data["later"]["random"] = changed, pair[1]
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_progression(data, (ProbeConfig(steps=2),))
        for invalid in ({}, {"early": arms()["early"]}, {"random": arms()["early"], "later": arms()["later"]},
                        {"../../escape": arms()["early"], "later": arms()["later"]}):
            with self.assertRaises(ValueError): validate_progression(invalid, (ProbeConfig(steps=2),))

    def test_shared_baselines_fit_once_and_age_errors_are_paired(self):
        data = arms()
        train, validation = data["later"]["checkpoint"]
        data["later"]["checkpoint"] = train, replace(validation, features=validation.features.roll(1, 0))
        rng = torch.get_rng_state().clone()
        with tempfile.TemporaryDirectory() as directory, patch("jepa_lab.compare.select_probe", wraps=select_probe) as adamw, patch("jepa_lab.compare.select_ridge", wraps=select_ridge) as ridge:
            result = compare_progression(data, (ProbeConfig(steps=30),), [29, 31],
                                         ridge_alphas=[0.1], probe_dir=directory)
            self.assertEqual(8, adamw.call_count)  # two ages + two shared baselines, for two seeds
            self.assertEqual(4, ridge.call_count)  # deterministic ridge is fitted once per representation
            self.assertEqual(12, len(list(Path(directory).glob("**/*.pt"))))
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertFalse(result["test_evaluated"])
        self.assertEqual(1.0, result["summary"]["early"]["mean_accuracy"])
        self.assertEqual(0.0, result["summary"]["later"]["mean_accuracy"])
        for run in result["runs"]:
            changes = run["paired_validation"]["adamw"]["later"]["vs_early"]
            self.assertEqual(10, changes["regressed"])
            self.assertEqual(0, changes["recovered"])
            self.assertEqual({run["seed"]}, {item["seed"] for item in run["candidate_grid"]})
        self.assertIn("ridge", result["runs"][0]["paired_validation"])
        self.assertNotIn("ridge", result["runs"][1]["paired_validation"])

    def test_all_arms_and_grids_fail_before_any_fitting(self):
        data = arms()
        pair = data["later"]["pixels"]
        data["later"]["pixels"] = replace(pair[0], artifact_sha256="d"*64), pair[1]
        with patch("jepa_lab.progression.fit_representation", side_effect=AssertionError("fit ran")):
            for value, seeds, alphas in ((data, [29], None), (arms(), [29, 29], None), (arms(), [29], [0])):
                with self.assertRaises(ValueError):
                    compare_progression(value, (ProbeConfig(steps=2),), seeds, ridge_alphas=alphas)
