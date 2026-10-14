import unittest
from pathlib import Path
import hashlib
import tempfile
from unittest.mock import patch

import torch

from jepa_lab.pretrained import load_pretrained, pretrained_embeddings, pretrained_input, transform_metadata


class PretrainedInputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_constant_rgb_channels_follow_imagenet_scaling(self):
        images = torch.zeros(2, 3, 32, 32, dtype=torch.uint8)
        images[1].fill_(255)
        before = images.clone()
        rng = torch.get_rng_state().clone()
        values = pretrained_input(images)
        expected = torch.tensor([[-.485/.229, -.456/.224, -.406/.225],
                                 [(1-.485)/.229, (1-.456)/.224, (1-.406)/.225]])
        self.assertEqual(values.shape, (2, 3, 224, 224))
        torch.testing.assert_close(values[:, :, 100, 100], expected)
        self.assertTrue(torch.equal(images, before))
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertFalse(values.requires_grad)

    def test_batch_isolation_and_repeatability(self):
        image = torch.arange(3*32*32).remainder(256).to(torch.uint8).reshape(1, 3, 32, 32)
        first = pretrained_input(image)
        paired = pretrained_input(torch.cat((image, 255-image)))
        torch.testing.assert_close(first[0], paired[0], rtol=0, atol=0)
        torch.testing.assert_close(first, pretrained_input(image), rtol=0, atol=0)

    def test_invalid_layout_dtype_device_and_bounds(self):
        for images in (None, torch.zeros(1, 3, 32, 32),
                       torch.zeros(1, 32, 32, 3, dtype=torch.uint8),
                       torch.zeros(0, 3, 32, 32, dtype=torch.uint8),
                       torch.zeros(17, 3, 32, 32, dtype=torch.uint8),
                       torch.empty(1, 3, 32, 32, dtype=torch.uint8, device="meta")):
            with self.subTest(images=type(images)):
                with self.assertRaises(ValueError):
                    pretrained_input(images)

    def test_metadata_copy_does_not_change_recipe(self):
        metadata = transform_metadata()
        metadata["mean"][0] = 99
        self.assertEqual(transform_metadata()["mean"][0], .485)


class PretrainedEncoderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_local_loader_is_frozen_offline_and_rng_preserving(self):
        from torchvision.models import resnet18
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"synthetic.pth"
            torch.save(resnet18(weights=None).state_dict(), path)
            sha = hashlib.sha256(path.read_bytes()).hexdigest()
            rng = torch.get_rng_state().clone()
            # Synthetic state is accepted only under a test-specific hash substitution.
            with patch("jepa_lab.pretrained.WEIGHTS_SHA256", sha), \
                    patch("torch.hub.download_url_to_file", side_effect=AssertionError("No downloads")):
                model, metadata = load_pretrained(path)
            self.assertFalse(model.training)
            self.assertTrue(all(not p.requires_grad for p in model.parameters()))
            self.assertIsInstance(model.fc, torch.nn.Identity)
            self.assertEqual(metadata["weights_sha256"], sha)
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))

    def test_weight_identity_and_malformed_state_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"bad.pth"
            for state in ({"not_a_resnet": torch.ones(2)}, {"bad": torch.tensor(float("nan"))}):
                torch.save(state, path)
                with self.assertRaisesRegex(ValueError, "declared"):
                    load_pretrained(path)
                with patch("jepa_lab.pretrained.WEIGHTS_SHA256", hashlib.sha256(path.read_bytes()).hexdigest()):
                    with self.assertRaisesRegex(ValueError, "invalid"):
                        load_pretrained(path)
            path.write_bytes(b"not a torch state")
            with patch("jepa_lab.pretrained.WEIGHTS_SHA256", hashlib.sha256(path.read_bytes()).hexdigest()):
                with self.assertRaises(ValueError):
                    load_pretrained(path)

    def test_inference_preserves_buffers_mixed_modes_and_gradients(self):
        from torchvision.models import resnet18
        model = resnet18(weights=None)
        model.fc = torch.nn.Identity()
        model.train()
        model.layer1.eval()
        modes = [m.training for m in model.modules()]
        state = {k:v.clone() for k,v in model.state_dict().items()}
        rng = torch.get_rng_state().clone()
        images = torch.zeros(1, 3, 32, 32, dtype=torch.uint8)
        result = pretrained_embeddings(model, images)
        self.assertEqual(result.shape, (1, 512))
        self.assertFalse(result.requires_grad)
        self.assertEqual(modes, [m.training for m in model.modules()])
        self.assertTrue(all(torch.equal(v,model.state_dict()[k]) for k,v in state.items()))
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        torch.testing.assert_close(result, pretrained_embeddings(model, images), rtol=0, atol=0)

    def test_failure_restores_mode(self):
        class Bad(torch.nn.Module):
            def forward(self, values):
                return torch.full((len(values), 512), float("nan"))
        model = Bad().train()
        with self.assertRaises(ValueError):
            pretrained_embeddings(model, torch.zeros(1,3,32,32,dtype=torch.uint8))
        self.assertTrue(model.training)
