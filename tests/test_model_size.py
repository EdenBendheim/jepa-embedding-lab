import unittest
import torch
from jepa_lab.model import JEPA,ModelConfig
from jepa_lab.model_size import model_size,validate_local_config


class ModelSizeTests(unittest.TestCase):
    def test_exact_groups_memory_and_unchanged_rng_across_widths(self):
        state=torch.get_rng_state().clone();previous=0
        for width in (32,64,96):
            config=ModelConfig(patch_dim=48,embedding_dim=width)
            report=model_size(config)
            self.assertTrue(torch.equal(state,torch.get_rng_state()))
            with torch.random.fork_rng(devices=[]):model=JEPA(config)
            self.assertEqual(sum(p.numel() for p in model.parameters()),report['parameters'])
            self.assertEqual(0,report['groups']['target']['trainable_parameters'])
            self.assertGreater(report['parameters'],previous);previous=report['parameters']
            self.assertEqual(4*previous,report['fp32_parameter_bytes'])
            self.assertEqual(report['groups']['context']['parameters'],report['groups']['target']['parameters'])

    def test_invalid_local_capacity_is_rejected_before_model_allocation(self):
        from unittest.mock import patch
        for config in (ModelConfig(patch_dim=48,embedding_dim=1024),ModelConfig(),ModelConfig(patch_dim=48,encoder_depth=10)):
            with patch('jepa_lab.model_size.JEPA',side_effect=AssertionError('allocated huge model')):
                with self.assertRaises(ValueError):model_size(config)
        with self.assertRaises(ValueError):ModelConfig(heads=True)
