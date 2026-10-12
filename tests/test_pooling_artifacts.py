from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from jepa_lab.cifar10 import make_manifest
from jepa_lab.compare import run_comparison
from jepa_lab.embeddings import extract_cifar10, random_encoder
from jepa_lab.features import load_features
from jepa_lab.model import ModelConfig
from jepa_lab.pooling import POOLING, pooling_dimension
from jepa_lab.probes import ProbeConfig
from jepa_lab.selection import make_pilot
from test_embeddings import Fixture


class PoolingArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def test_each_pooling_round_trips_and_cannot_mislabel_dimensions(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            labels = Fixture(root).labels
            manifest = make_manifest(Fixture(root), Fixture(root, train=False))
            selection = make_pilot(manifest, labels, train_per_class=1, validation_per_class=1)
            m, s = root/"manifest.json", root/"selection.json"
            m.write_text(json.dumps(manifest)); s.write_text(json.dumps(selection))
            config = ModelConfig(patch_dim=48, embedding_dim=16, encoder_depth=1)
            for mode in POOLING:
                path = root/f"{mode}.pt"
                with patch("jepa_lab.embeddings.CIFAR10Binary", Fixture):
                    report = extract_cifar10(root, m, representation="random", config=config, pooling=mode,
                                            selection_path=s, output=path)
                options = dict(manifest=manifest, selection=selection, training_labels=labels, partition="train")
                values = load_features(path, **options)
                self.assertEqual(pooling_dimension(16, mode), values.features.shape[1])
                self.assertEqual(POOLING[mode], values.metadata["pooling"])
                self.assertEqual(report["features_sha256"], torch.load(path, weights_only=True)["features_sha256"])
                saved = torch.load(path, weights_only=True)
                for bad in ("unknown", POOLING["mean_std"] if mode != "mean_std" else POOLING["mean"]):
                    torch.save(saved | dict(feature_metadata=saved["feature_metadata"] | dict(pooling=bad)), path)
                    with self.assertRaises(ValueError): load_features(path, **options)

    def test_complete_max_and_mean_std_comparisons_match_random_and_reject_wrong_reuse(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            manifest = make_manifest(Fixture(root), Fixture(root, train=False))
            selection = make_pilot(manifest, Fixture(root).labels, train_per_class=1, validation_per_class=1)
            m, s, checkpoint = root/"manifest.json", root/"selection.json", root/"model.pt"
            m.write_text(json.dumps(manifest)); s.write_text(json.dumps(selection))
            config = ModelConfig(patch_dim=48, embedding_dim=16, encoder_depth=1)
            torch.save(dict(format_version=1, dataset="CIFAR-10", manifest_sha256=manifest["manifest_sha256"],
                runtime=dict(torch=str(torch.__version__), cpu_threads=1), step=3, config=asdict(config),
                model=random_encoder(config, 29).state_dict(), settings=dict(seed=29)), checkpoint)
            with patch("jepa_lab.embeddings.CIFAR10Binary", Fixture), patch("jepa_lab.compare.CIFAR10Binary", Fixture):
                for mode in ("max", "mean_std"):
                    result = run_comparison(root, m, s, root/mode, checkpoint=checkpoint, pooling=mode,
                                            candidates=(ProbeConfig(steps=2),), ridge_alphas=[1])
                    self.assertFalse(result["test_evaluated"])
                    for name in ("checkpoint", "random"):
                        self.assertEqual(pooling_dimension(16, mode), result["representations"][name]["train_artifact"]["feature_dim"])
                    with patch("jepa_lab.compare.select_probe", side_effect=AssertionError("fit ran")):
                        with self.assertRaisesRegex(ValueError, "requested"):
                            run_comparison(root, m, s, root/mode, pooling="mean", candidates=(ProbeConfig(steps=2),))
            Fixture.accessed.clear()
            with self.assertRaises(ValueError): extract_cifar10(root, m, representation="pixels", pooling="max")
            self.assertEqual([], Fixture.accessed)
