from pathlib import Path
import tempfile
import unittest

import torch

from jepa_lab.probe_artifacts import save_probe, load_probe
from jepa_lab.ridge import RidgeConfig, RidgeProbe, fit_ridge, select_ridge
from test_probe_artifacts import sources
from test_probes import synthetic


class RidgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def test_train_only_normalization_frozen_inputs_rng_and_regularization(self):
        x, y = synthetic(); x.requires_grad_()
        rng = torch.get_rng_state().clone()
        probe = fit_ridge(x, y, RidgeConfig(0.01))
        stronger = fit_ridge(x, y, RidgeConfig(100))
        self.assertEqual(1.0, probe.score(x, y)["accuracy"])
        self.assertLess(stronger.weight.norm(), probe.weight.norm())
        self.assertTrue(torch.equal(probe.mean, x.detach().mean(0)))
        self.assertEqual(1.0, float(probe.scale[-1]))
        self.assertIsNone(x.grad)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        a, _ = select_ridge(x, y, x.detach(), y, [0.1])
        b, _ = select_ridge(x, y, x.detach() + 3, y, [0.1])
        self.assertTrue(torch.equal(a.weight, b.weight))

    def test_dual_solve_matches_primal_formula_on_high_dimensional_features(self):
        labels = torch.arange(10).repeat(2)
        generator = torch.Generator().manual_seed(23)
        values = torch.randn(20, 64, generator=generator)
        probe = fit_ridge(values, labels, RidgeConfig(0.3))
        x = ((values - probe.mean) / probe.scale).double()
        center = x.mean(0); x = x - center
        target = torch.nn.functional.one_hot(labels, 10).double()
        prior = target.mean(0)
        expected = torch.linalg.solve(x.T @ x + 6 * torch.eye(64, dtype=torch.float64), x.T @ (target - prior))
        torch.testing.assert_close(probe.weight, expected.T.float())
        torch.testing.assert_close(probe.bias, (prior - center @ expected).float())

    def test_selection_validation_metrics_invalid_candidates_and_artifact_round_trip(self):
        train, validation = sources()
        probe, report = select_ridge(train.features, train.labels, validation.features, validation.labels, [0.1, 1, 10])
        results = report["candidates"]
        expected = min(range(3), key=lambda i: (-results[i]["validation"]["accuracy"], results[i]["validation"]["mean_squared_error"], i))
        self.assertEqual(expected, report["selected_candidate"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ridge.pt"
            save_probe(path, probe, train, validation)
            loaded, summary = load_probe(path, train, validation)
            self.assertIsInstance(loaded, RidgeProbe)
            self.assertEqual(probe.config, loaded.config)
            self.assertEqual(report["candidates"][expected]["validation"], summary["scores"]["validation"])
        for alphas in ([], [True], [-1], [float("nan")], [0.1, 0.1], [1e5]):
            with self.subTest(alphas=alphas), self.assertRaises(ValueError):
                select_ridge(train.features, train.labels, validation.features, validation.labels, alphas)
