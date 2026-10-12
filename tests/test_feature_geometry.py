import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from jepa_lab.cifar10 import make_manifest
from jepa_lab.embeddings import extract_cifar10
from jepa_lab.feature_geometry import feature_geometry, inspect_geometry
from jepa_lab.selection import make_pilot
from test_embeddings import Fixture


class FeatureGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def test_known_spectrum_pair_angles_and_constant_rows(self):
        values = torch.tensor([[1., 0.], [-1., 0.], [0., 1.], [0., -1.]], requires_grad=True)
        before = values.detach().clone()
        rng = torch.get_rng_state().clone()
        result = feature_geometry(values)
        self.assertEqual(2, result["centered_numerical_rank"])
        self.assertAlmostEqual(2., result["covariance_entropy_effective_rank"])
        self.assertAlmostEqual(-1/3, result["mean_nonzero_row_pair_cosine"])
        changed = feature_geometry(values.flip(1) * 3)
        self.assertAlmostEqual(result["covariance_entropy_effective_rank"], changed["covariance_entropy_effective_rank"])
        self.assertAlmostEqual(3 * result["mean_row_l2"], changed["mean_row_l2"])
        self.assertTrue(torch.equal(values.detach(), before))
        self.assertIsNone(values.grad)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        constant = feature_geometry(torch.ones(3, 2))
        self.assertEqual(0, constant["centered_numerical_rank"])
        self.assertEqual(0., constant["covariance_entropy_effective_rank"])
        self.assertEqual(2, constant["zero_variance_columns"])
        self.assertAlmostEqual(1., constant["mean_nonzero_row_pair_cosine"])

    def test_zero_rows_singletons_and_invalid_inputs(self):
        for values in (torch.zeros(1, 4), torch.zeros(3, 4)):
            report = feature_geometry(values)
            self.assertIsNone(report["mean_nonzero_row_pair_cosine"])
            self.assertEqual(len(values), report["zero_norm_rows"])
            json.dumps(report, allow_nan=False)
        for values in (torch.ones(2, 3).double(), torch.ones(1001, 2), torch.ones(2, 4097),
                       torch.ones(0, 2), torch.ones(2, 0), torch.full((2, 3), float("inf"))):
            with self.assertRaises(ValueError): feature_geometry(values)

    def test_verified_geometry_never_reads_images_and_rejects_test(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            manifest = make_manifest(Fixture(root), Fixture(root, train=False))
            selection = make_pilot(manifest, Fixture(root).labels, train_per_class=1, validation_per_class=1)
            m, s, artifact = root/"manifest.json", root/"selection.json", root/"pixels.pt"
            m.write_text(json.dumps(manifest)); s.write_text(json.dumps(selection))
            with patch("jepa_lab.embeddings.CIFAR10Binary", Fixture):
                extract_cifar10(root, m, selection_path=s, representation="pixels", output=artifact)
            before = artifact.read_bytes(); Fixture.accessed.clear()
            with patch("jepa_lab.feature_geometry.CIFAR10Binary", Fixture):
                report = inspect_geometry(root, m, s, artifact, partition="train")
                self.assertEqual(10, report["geometry"]["count"])
                self.assertNotIn("features", report["source"])
                self.assertFalse(report["test_evaluated"])
                with self.assertRaises(ValueError): inspect_geometry(root, m, s, artifact, partition="test")
            self.assertEqual([], Fixture.accessed)
            self.assertEqual(before, artifact.read_bytes())
