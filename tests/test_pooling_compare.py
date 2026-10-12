from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from jepa_lab.compare import fit_representation
from jepa_lab.pooling import POOLING
from jepa_lab.pooling_compare import compare_pooling, validate_pooling_arms
from jepa_lab.probe_artifacts import load_probe
from jepa_lab.probes import ProbeConfig
from test_compare import sets


def pooling_arms():
    source = sets()
    result = {}
    for mode in ("mean", "max", "mean_std"):
        data = dict(source)
        for name in ("checkpoint", "random"):
            metadata = source[name][0].metadata | dict(pooling=POOLING[mode])
            if name == "checkpoint":
                metadata |= dict(checkpoint_sha256="e"*64, completed_steps=3, training_settings=dict(seed=29))
            else: metadata |= dict(seed=29)
            data[name] = tuple(replace(item, metadata=deepcopy(metadata),
                features=torch.cat((item.features, item.features), 1) if mode == "mean_std" else item.features.clone())
                for item in source[name])
        result[mode] = data
    return result


class PoolingComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def test_variation_is_limited_to_pooling_and_capacity_is_explicit(self):
        data = pooling_arms()
        validate_pooling_arms(data, (ProbeConfig(steps=2),))
        for bad in ("checkpoint", "step", "seed", "pooling", "pixels", "pixel_values", "examples"):
            data = pooling_arms()
            if bad in ("checkpoint", "step", "pooling"):
                key, value = {"checkpoint": ("checkpoint_sha256", "f"*64), "step": ("completed_steps", 4),
                              "pooling": ("pooling", POOLING["mean"])}[bad]
                data["max"]["checkpoint"] = tuple(replace(item, metadata=item.metadata | {key: value}) for item in data["max"]["checkpoint"])
            if bad == "seed":
                data["max"]["random"] = tuple(replace(item, metadata=item.metadata | dict(seed=31)) for item in data["max"]["random"])
            if bad.startswith("pixel"):
                pair = data["max"]["pixels"]
                value = replace(pair[0], artifact_sha256="d"*64) if bad == "pixels" else replace(pair[0], features=pair[0].features + 0.1)
                data["max"]["pixels"] = value, pair[1]
            if bad == "examples":
                data["max"] = {name: tuple(replace(item, indices=item.indices[::-1]) for item in pair) for name, pair in data["max"].items()}
            with self.subTest(bad=bad), self.assertRaises(ValueError): validate_pooling_arms(data, (ProbeConfig(steps=2),))

    def test_pixel_fits_are_shared_and_saved_scores_and_paired_errors_reproduce(self):
        data = pooling_arms()
        pair = data["max"]["checkpoint"]
        data["max"]["checkpoint"] = pair[0], replace(pair[1], features=pair[1].features.roll(1, 0))
        rng = torch.get_rng_state().clone()
        with tempfile.TemporaryDirectory() as folder, patch("jepa_lab.pooling_compare.fit_representation", wraps=fit_representation) as fitted:
            result = compare_pooling(data, (ProbeConfig(steps=30),), [29, 31], ridge_alphas=[1], probe_dir=folder)
            self.assertEqual(14, fitted.call_count)  # six encoder pairs + one pixel, per seed
            self.assertEqual(21, len(list(Path(folder).glob("**/*.pt"))))  # 14 AdamW + seven ridge
            self.assertEqual({"mean": 16, "max": 16, "mean_std": 32}, result["feature_dimensions"])
            self.assertEqual(10, result["runs"][0]["paired_validation"]["adamw"]["max"]["vs_mean"]["regressed"])
            for run in result["runs"]:
                for mode in data:
                    for name in ("checkpoint", "random"):
                        report = run["arms"][mode][name]
                        for scored in (report, report.get("ridge")):
                            if scored:
                                probe, _ = load_probe(scored["selected_probe_artifact"]["path"], *data[mode][name])
                                self.assertEqual(scored["validation"], probe.score(data[mode][name][1].features, data[mode][name][1].labels))
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertFalse(result["test_evaluated"])

    def test_invalid_arms_and_settings_fail_before_fitting(self):
        with patch("jepa_lab.pooling_compare.fit_representation", side_effect=AssertionError("fit ran")):
            for arms, seeds, alphas in (({}, [29], None), ({"mean": pooling_arms()["mean"]}, [29], None),
                                       (pooling_arms(), [29, 29], None), (pooling_arms(), [29], [0])):
                with self.assertRaises(ValueError): compare_pooling(arms, (ProbeConfig(steps=2),), seeds, ridge_alphas=alphas)
