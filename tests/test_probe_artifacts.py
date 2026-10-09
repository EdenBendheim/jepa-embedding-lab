from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from jepa_lab.features import FeatureSet
from jepa_lab.probe_artifacts import load_probe, save_probe
from jepa_lab.probes import ProbeConfig, fit_probe
from jepa_lab.ridge import RidgeConfig
from test_probes import synthetic


def sources():
    values, labels = synthetic(2)
    metadata = dict(representation="random", seed=29)
    return tuple(FeatureSet(values.clone(), labels.clone(), tuple(range(start, start + 20)), partition,
                            "a" * 64, "b" * 64, metadata.copy(), digest * 64)
                 for partition, start, digest in (("train", 0, "c"), ("validation", 20, "d")))


class ProbeArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def test_round_trip_keeps_scores_weights_rng_and_source_identity(self):
        train, validation = sources()
        probe = fit_probe(train.features, train.labels, ProbeConfig(steps=20))
        rng = torch.get_rng_state().clone()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "probe.pt"
            summary = save_probe(path, probe, train, validation)
            restored, again = load_probe(path, train, validation)
            self.assertEqual(summary, again)
            self.assertTrue(torch.equal(probe.weight, restored.weight))
            self.assertEqual(probe.score(validation.features, validation.labels), again["scores"]["validation"])
            self.assertFalse(restored.weight.requires_grad)
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            with self.assertRaises(ValueError):
                load_probe(path, train, replace(validation, artifact_sha256="e" * 64))
            restored.weight.zero_()
            self.assertTrue(torch.equal(load_probe(path, train, validation)[0].weight, probe.weight))

    def test_corrupt_state_schema_normalization_scores_and_runtime_are_rejected(self):
        train, validation = sources()
        probe = fit_probe(train.features, train.labels, ProbeConfig(steps=2))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "probe.pt"
            save_probe(path, probe, train, validation)
            original = path.read_bytes()
            for change in ("weight", "scale", "shape", "version", "config", "scores", "runtime"):
                path.write_bytes(original)
                saved = torch.load(path, weights_only=True)
                if change == "weight": saved["state"]["weight"][0, 0] += 1
                if change == "scale": saved["state"]["scale"].zero_()
                if change == "shape": saved["state"]["bias"] = torch.zeros(9)
                if change == "version": saved["format_version"] = True
                if change == "config": saved["config"]["steps"] = True
                if change == "scores": saved["scores"]["validation"]["correct"] += 1
                if change == "runtime": saved["runtime"]["torch"] = "other"
                torch.save(saved, path)
                with self.subTest(change=change), self.assertRaises(ValueError): load_probe(path, train, validation)
            path.write_bytes(b"corrupt")
            with self.assertRaises(ValueError): load_probe(path, train, validation)

    def test_atomic_save_failure_preserves_previous_classifier(self):
        train, validation = sources()
        probe = fit_probe(train.features, train.labels, ProbeConfig(steps=2))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "probe.pt"
            save_probe(path, probe, train, validation)
            previous = path.read_bytes()
            with patch("jepa_lab.cifar10_train.os.replace", side_effect=OSError("disk failure")), self.assertRaises(OSError):
                save_probe(path, probe, train, validation)
            self.assertEqual(previous, path.read_bytes())
            self.assertEqual([], list(Path(directory).glob(".probe.pt.*")))

    def test_wrong_probe_family_config_fails_before_replacing_file(self):
        train, validation = sources()
        probe = fit_probe(train.features, train.labels, ProbeConfig(steps=2))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "probe.pt"
            save_probe(path, probe, train, validation)
            before = path.read_bytes()
            with self.assertRaises(ValueError): save_probe(path, replace(probe, config=RidgeConfig()), train, validation)
            self.assertEqual(before, path.read_bytes())
