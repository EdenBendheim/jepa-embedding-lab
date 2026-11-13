import unittest
from unittest.mock import patch

import torch

from jepa_lab.model import JEPA, ModelConfig
from jepa_lab.variants.multiblock_masks import multi_block_mask
from jepa_lab.variants.multiblock_model import multi_block_forward, multi_block_loss


class MultiBlockModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        with torch.random.fork_rng():
            torch.manual_seed(29)
            self.model = JEPA(ModelConfig(rows=4, cols=4, embedding_dim=16, encoder_depth=1))
            self.patches = torch.randn(2, 16, 4)
        self.masks = [multi_block_mask(4, 4, seed=seed) for seed in (29, 31)]

    def test_shared_context_teacher_and_exact_independent_block_outputs_and_loss(self):
        rng = torch.get_rng_state().clone()
        with patch.object(self.model.context_encoder, 'forward', wraps=self.model.context_encoder.forward) as context, \
                patch.object(self.model.target_encoder, 'forward', wraps=self.model.target_encoder.forward) as teacher, \
                patch.object(self.model.predictor, 'forward', wraps=self.model.predictor.forward) as predictor:
            predicted, targets = multi_block_forward(self.model, self.patches, self.masks)
        self.assertEqual((2, 2, 4, 16), predicted.shape)
        self.assertEqual(1, context.call_count)
        self.assertEqual(1, teacher.call_count)
        self.assertEqual(2, predictor.call_count)
        self.assertFalse(targets.requires_grad)
        context_indices = [mask.context for mask in self.masks]
        for block in range(2):
            expected = self.model(self.patches, context_indices, [mask.blocks[block] for mask in self.masks])
            self.assertTrue(torch.equal(predicted[:, block], expected[0]))
            self.assertTrue(torch.equal(targets[:, block], expected[1]))
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))
        loss = multi_block_loss(self.model, self.patches, self.masks)
        expected_loss = torch.stack([torch.nn.functional.smooth_l1_loss(predicted[:, b], targets[:, b])
                                     for b in range(2)]).mean()
        self.assertTrue(torch.allclose(loss, expected_loss))

    def test_joint_predictor_matches_one_forward_over_the_ordered_target_union(self):
        with patch.object(self.model.predictor, 'forward', wraps=self.model.predictor.forward) as predictor:
            predicted, targets = multi_block_forward(self.model, self.patches, self.masks, predictor_grouping='joint')
        self.assertEqual(1, predictor.call_count)
        contexts = [mask.context for mask in self.masks]
        unions = [tuple(index for block in mask.blocks for index in block) for mask in self.masks]
        expected_predictions, expected_targets = self.model(self.patches, contexts, unions)
        self.assertTrue(torch.equal(predicted.flatten(1, 2), expected_predictions))
        self.assertTrue(torch.equal(targets.flatten(1, 2), expected_targets))
        self.assertFalse(targets.requires_grad)

    def test_hidden_union_cannot_change_predictions_or_receive_gradients(self):
        altered = self.patches.clone()
        for image, mask in enumerate(self.masks):
            hidden = sorted(set(range(16)) - set(mask.context))
            altered[image, hidden] += 50
        for grouping in ('separate', 'joint'):
            with self.subTest(grouping=grouping):
                options = dict(predictor_grouping=grouping)
                first = multi_block_forward(self.model, self.patches, self.masks, **options)[0]
                self.assertTrue(torch.equal(first, multi_block_forward(self.model, altered, self.masks, **options)[0]))
                self.model.zero_grad()
                patches = self.patches.clone().requires_grad_(True)
                multi_block_loss(self.model, patches, self.masks, **options).backward()
                for image, mask in enumerate(self.masks):
                    hidden = sorted(set(range(16)) - set(mask.context))
                    self.assertEqual(0., float(patches.grad[image, hidden].abs().sum()))
                    self.assertGreater(float(patches.grad[image, list(mask.context)].abs().sum()), 0)
                self.assertTrue(all(p.grad is None for p in self.model.target_encoder.parameters()))
                self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in self.model.predictor.parameters()))

    def test_incompatible_batches_fail_before_attention(self):
        with patch.object(self.model.context_encoder, 'forward', side_effect=AssertionError('attention')):
            for patches, masks in ((self.patches.double(), self.masks), (self.patches, self.masks[:1]),
                    (self.patches, [self.masks[0], multi_block_mask(4, 4, blocks=1)]),
                    (self.patches, [multi_block_mask(8, 8)]*2), (self.patches*float('nan'), self.masks)):
                with self.assertRaises(ValueError):
                    multi_block_forward(self.model, patches, masks)
