from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from jepa_lab.cifar10 import make_manifest
from jepa_lab.embeddings import extract_cifar10, random_encoder
from jepa_lab.model import ModelConfig
from jepa_lab.pooling_compare import run_pooling
from jepa_lab.probes import ProbeConfig
from jepa_lab.selection import make_pilot
from test_embeddings import Fixture


class PoolingWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def test_saved_artifacts_are_verified_without_image_reads_or_source_mutation(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            manifest = make_manifest(Fixture(root), Fixture(root, train=False))
            selection = make_pilot(manifest, Fixture(root).labels, train_per_class=1, validation_per_class=1)
            m, s, checkpoint = root/"manifest.json", root/"selection.json", root/"encoder.pt"
            m.write_text(json.dumps(manifest)); s.write_text(json.dumps(selection))
            config = ModelConfig(patch_dim=48, embedding_dim=16, encoder_depth=1)
            torch.save(dict(format_version=1, dataset="CIFAR-10", manifest_sha256=manifest["manifest_sha256"],
                runtime=dict(torch=str(torch.__version__), cpu_threads=1), step=3, config=asdict(config),
                model=random_encoder(config, 29).state_dict(), settings=dict(seed=29)), checkpoint)
            directories = {mode: root/mode for mode in ("mean", "max", "mean_std")}
            with patch("jepa_lab.embeddings.CIFAR10Binary", Fixture):
                for mode, directory in directories.items():
                    for name in ("checkpoint", "random"):
                        for partition in ("train", "validation"):
                            extract_cifar10(root, m, selection_path=s, representation=name, partition=partition,
                                pooling=mode, config=config if name == "random" else None,
                                checkpoint=checkpoint if name == "checkpoint" else None,
                                output=directory/f"{name}-{partition}.pt")
                for partition in ("train", "validation"):
                    extract_cifar10(root, m, selection_path=s, representation="pixels", partition=partition,
                                   output=root/"shared"/f"pixels-{partition}.pt")
            before = {path: path.read_bytes() for path in root.glob("**/*.pt")}
            Fixture.accessed.clear()
            with patch("jepa_lab.pooling_compare.CIFAR10Binary", Fixture):
                result = run_pooling(root, m, s, root/"shared", directories,
                                     candidates=(ProbeConfig(steps=2),), seeds=[29], ridge_alphas=[1])
                self.assertEqual([], Fixture.accessed)
                self.assertEqual(before, {path: path.read_bytes() for path in before})
                self.assertEqual(dict(train=10, validation=10), result["counts"])
                self.assertFalse(result["test_evaluated"])
                path = directories["max"]/"random-validation.pt"
                saved = torch.load(path, weights_only=True)
                torch.save(saved | dict(labels=saved["labels"].roll(1)), path)
                with patch("jepa_lab.pooling_compare.fit_representation", side_effect=AssertionError("fit ran")):
                    with self.assertRaises(ValueError):
                        run_pooling(root, m, s, root/"shared", directories, candidates=(ProbeConfig(steps=2),), seeds=[29])
