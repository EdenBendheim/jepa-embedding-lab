from pathlib import Path
import tempfile
import unittest

import torch

from jepa_lab.model import JEPA, ModelConfig
from jepa_lab.train import train_smoke


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(7)
        self.config = ModelConfig(rows=4, cols=4, embedding_dim=16, encoder_depth=1)
        self.model = JEPA(self.config)
        self.patches = torch.randn(2, 16, 4)
        self.context = torch.tensor([[0, 1, 2, 3], [8, 9, 10, 11]])
        self.target = torch.tensor([[4, 5], [12, 13]])

    def test_shapes_stop_gradient_and_context_training(self):
        predictions, targets = self.model(self.patches, self.context, self.target)
        self.assertEqual(predictions.shape, (2, 2, 16))
        self.assertEqual(targets.shape, predictions.shape)
        self.assertFalse(targets.requires_grad)
        before = self.model.context_encoder.projection.weight.detach().clone()
        optimizer = torch.optim.AdamW((p for p in self.model.parameters() if p.requires_grad), lr=0.01)
        torch.nn.functional.smooth_l1_loss(predictions, targets).backward()
        self.assertTrue(all(p.grad is None for p in self.model.target_encoder.parameters()))
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in self.model.context_encoder.parameters()))
        optimizer.step()
        self.assertFalse(torch.equal(before, self.model.context_encoder.projection.weight))
        self.model.train()
        self.assertFalse(self.model.target_encoder.training)

    def test_hidden_pixels_do_not_affect_predictions_or_receive_gradients(self):
        altered = self.patches.clone()
        for batch in range(2):
            hidden = [i for i in range(16) if i not in self.context[batch].tolist()]
            altered[batch, hidden] += 100
        predictions, _ = self.model(self.patches, self.context, self.target)
        changed, _ = self.model(altered, self.context, self.target)
        self.assertTrue(torch.equal(predictions, changed))
        patches = self.patches.clone().requires_grad_(True)
        self.model.loss(patches, self.context, self.target).backward()
        for batch in range(2):
            hidden = [i for i in range(16) if i not in self.context[batch].tolist()]
            self.assertEqual(float(patches.grad[batch, hidden].abs().sum()), 0)
            self.assertGreater(float(patches.grad[batch, self.context[batch]].abs().sum()), 0)

    def test_ema_interpolation_and_endpoints(self):
        with torch.no_grad():
            for parameter in self.model.context_encoder.parameters():
                parameter.add_(2)
        old = [p.clone() for p in self.model.target_encoder.parameters()]
        self.model.update_target(0.75)
        for before, target, context in zip(old, self.model.target_encoder.parameters(),
                                            self.model.context_encoder.parameters()):
            self.assertTrue(torch.allclose(target, 0.75*before+0.25*context))
        old = [p.clone() for p in self.model.target_encoder.parameters()]
        self.model.update_target(1)
        self.assertTrue(all(torch.equal(a, b) for a,b in zip(old, self.model.target_encoder.parameters())))
        self.model.update_target(0)
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(self.model.context_encoder.parameters(),
                                                           self.model.target_encoder.parameters())))
        for value in (-0.1, 1.1, float("nan")):
            with self.assertRaises(ValueError):
                self.model.update_target(value)

    def test_invalid_or_overlapping_masks_rejected(self):
        for context, target in (([0,1],[1,2]), ([0,0],[2]), ([16],[2]),
                                ([-1],[2]), ([0.5],[2]), ([],[2]), ([[0],[1],[2]], [3])):
            with self.assertRaises(ValueError):
                self.model(self.patches, context, target)
        with self.assertRaises(ValueError):
            self.model(self.patches[:, :3], self.context, self.target)
        with self.assertRaises(ValueError):
            self.model(self.patches*float("nan"), self.context, self.target)

    def test_seeded_training_and_checkpoint_resume_match_continuous_run(self):
        config = self.config
        options = dict(config=config, batch_size=2, seed=19)
        complete = train_smoke(steps=4, **options)
        repeated = train_smoke(steps=4, **options)
        self.assertEqual(complete["history"], repeated["history"])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"model.pt"
            first = train_smoke(steps=2, checkpoint=path, **options)
            second = train_smoke(steps=2, resume=path, **options)
            self.assertEqual(first["history"]+second["history"], complete["history"])
            self.assertEqual(second["completed_step"], 4)
            with self.assertRaises(ValueError):
                train_smoke(steps=1, resume=path, batch_size=3, seed=19)


if __name__ == "__main__":
    unittest.main()
