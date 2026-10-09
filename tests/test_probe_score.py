from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from jepa_lab.cifar10 import make_manifest
from jepa_lab.compare import run_comparison
from jepa_lab.embeddings import random_encoder
from jepa_lab.model import ModelConfig
from jepa_lab.probes import ProbeConfig
from jepa_lab.probe_score import score_saved_probe
from jepa_lab.selection import make_pilot
from test_embeddings import Fixture


class SavedScoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def test_fixture_score_reproduces_without_refitting_or_reading_images(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = make_manifest(Fixture(root), Fixture(root, train=False))
            selection = make_pilot(manifest, Fixture(root).labels, train_per_class=1, validation_per_class=1)
            m, s, checkpoint = root / "manifest.json", root / "selection.json", root / "encoder.pt"
            m.write_text(json.dumps(manifest)); s.write_text(json.dumps(selection))
            config = ModelConfig(patch_dim=48, embedding_dim=16, encoder_depth=1)
            torch.save(dict(format_version=1, dataset="CIFAR-10", manifest_sha256=manifest["manifest_sha256"],
                            runtime=dict(torch=str(torch.__version__), cpu_threads=1), step=3, config=asdict(config),
                            model=random_encoder(config, 29).state_dict(), settings=dict(seed=29)), checkpoint)
            with patch("jepa_lab.compare.CIFAR10Binary", Fixture), patch("jepa_lab.embeddings.CIFAR10Binary", Fixture):
                original = run_comparison(root, m, s, root / "features", checkpoint=checkpoint,
                                          candidates=(ProbeConfig(steps=2),), probe_dir=root / "probes")
            Fixture.accessed.clear()
            artifact = original["representations"]["checkpoint"]["selected_probe_artifact"]
            inputs = (root, m, s, root / "features/checkpoint-train.pt", root / "features/checkpoint-validation.pt", Path(artifact["path"]))
            with patch("jepa_lab.probe_score.CIFAR10Binary", Fixture), patch("jepa_lab.probes.fit_probe", side_effect=AssertionError("refit")):
                result = score_saved_probe(*inputs)
                self.assertEqual([], Fixture.accessed)
                self.assertEqual(original["representations"]["checkpoint"]["validation"], result["probe"]["scores"]["validation"])
                self.assertFalse(result["refitted"])
                self.assertFalse(result["test_evaluated"])
                bad = torch.load(inputs[4], weights_only=True)
                bad["labels"][0] = (bad["labels"][0] + 1) % 10
                torch.save(bad, inputs[4])
                with patch("jepa_lab.probe_score.load_probe", side_effect=AssertionError("scored bad features")), self.assertRaises(ValueError):
                    score_saved_probe(*inputs)
