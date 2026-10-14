import unittest

import torch

from jepa_lab.pretrained import pretrained_input, transform_metadata


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
