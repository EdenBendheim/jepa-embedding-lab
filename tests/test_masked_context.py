import unittest
from copy import deepcopy
from unittest.mock import patch

import torch

from jepa_lab.masked_context import masked_context_report
from jepa_lab.model import JEPA, ModelConfig


class MaskedContextTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def setUp(self):
        with torch.random.fork_rng():
            torch.manual_seed(29)
            self.model = JEPA(ModelConfig(patch_dim=48, embedding_dim=8, heads=2, encoder_depth=1))
            self.images = torch.randint(256, (4, 3, 32, 32), dtype=torch.uint8)

    def test_repeatability_variance_identity_and_caller_isolation(self):
        self.model.train(); self.model.context_encoder.blocks.eval()
        modes = [m.training for m in self.model.modules()]
        state = deepcopy(self.model.state_dict()); images = self.images.clone()
        rng = torch.get_rng_state().clone()
        a = masked_context_report(self.model, self.images, batch_size=2)
        self.assertEqual(a, masked_context_report(self.model, self.images, batch_size=2))
        self.assertEqual(45, a['visible_tokens']); self.assertEqual(4, a['views'])
        self.assertAlmostEqual(a['total_variance'], a['within_image_mask_variance'] + a['between_image_mean_variance'])
        self.assertGreater(a['within_image_mask_variance'], 0)
        self.assertEqual(modes, [m.training for m in self.model.modules()])
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertTrue(torch.equal(images, self.images))
        self.assertTrue(all(torch.equal(v, self.model.state_dict()[k]) for k,v in state.items()))
        self.assertTrue(all(p.grad is None for p in self.model.parameters()))
        self.assertNotEqual(a['mask_recipe_sha256'], masked_context_report(
            self.model, self.images, mask_seeds=(5, 7))['mask_recipe_sha256'])

    def test_attention_control_and_hidden_patch_boundary(self):
        class Encoder(torch.nn.Module):
            def forward(self, patches, indices):
                visible = patches.gather(1, indices[..., None].expand(-1, -1, 48))
                # Deliberate attention effect: all visible tokens share a mean.
                return visible[..., :8] + visible.mean((1, 2))[:, None, None]
        self.model.context_encoder = Encoder()
        report = masked_context_report(self.model, self.images, mask_seeds=(29, 31))
        for view in report['per_view']:
            self.assertGreater(view['full_to_subset']['rmse'], 0)
            self.assertGreater(view['subset_to_masked']['rmse'], 0)
        from jepa_lab.cifar10 import patchify_rgb
        from jepa_lab.masking import block_mask
        patches = patchify_rgb(self.images)
        mask = block_mask(8, 8, seed=29)
        indices = torch.tensor(mask.context).expand(len(patches), -1)
        changed = patches.clone(); changed[:, list(mask.target + mask.unused)] += 9
        self.assertTrue(torch.equal(self.model.context_encoder(patches, indices),
                                    self.model.context_encoder(changed, indices)))

    def test_reject_invalid_inputs_and_restore_modes_on_error(self):
        for kwargs in ({'batch_size':True}, {'mask_seeds':(1,1)}, {'mask_seeds':(1,)},
                       {'mask_seeds':(-1,2)}, {'context_fraction':0}, {'target_rows':8,'target_cols':8}):
            with self.assertRaises(ValueError): masked_context_report(self.model, self.images, **kwargs)
        for images in (self.images.float(), self.images[:0], self.images[:,:,:16], torch.ones(2,3,32)):
            with self.assertRaises(ValueError): masked_context_report(self.model, images)
        self.model.train(); modes = [m.training for m in self.model.modules()]
        with patch.object(self.model.context_encoder, 'forward', side_effect=RuntimeError('encoder')):
            with self.assertRaisesRegex(RuntimeError,'encoder'): masked_context_report(self.model,self.images)
        self.assertEqual(modes,[m.training for m in self.model.modules()])
