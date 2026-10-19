from dataclasses import replace
from pathlib import Path
import tempfile,unittest
import torch
from jepa_lab.model import JEPA,ModelConfig,config_dict
from jepa_lab.model_size import model_size
from jepa_lab.train import train_smoke


class ModelPositionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_nonposition_weights_and_rng_match_and_fixed_grid_is_not_optimized(self):
        config=ModelConfig(patch_dim=48);torch.manual_seed(19);learned=JEPA(config);rng=torch.get_rng_state().clone()
        torch.manual_seed(19);fixed=JEPA(replace(config,position_encoding='sincos2d'))
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        for key,value in learned.state_dict().items():
            if 'positions' not in key:self.assertTrue(torch.equal(value,fixed.state_dict()[key]),key)
        self.assertNotIn('context_encoder.positions',dict(fixed.named_parameters()))
        before=fixed.context_encoder.positions.clone();patches=torch.randn(2,64,48,requires_grad=True)
        optimizer=torch.optim.AdamW((p for p in fixed.parameters() if p.requires_grad),lr=.01)
        loss=fixed.loss(patches,[0,1,2],[3,4]);loss.backward();optimizer.step();fixed.update_target(.9)
        self.assertTrue(torch.equal(before,fixed.context_encoder.positions))
        self.assertTrue(torch.equal(before,fixed.target_encoder.positions))
        self.assertEqual(0.,float(patches.grad[:,3:].abs().sum()))
        self.assertTrue(all(p.grad is None for p in fixed.target_encoder.parameters()))
        self.assertNotIn('position_encoding',config_dict(config))
        self.assertEqual('sincos2d',config_dict(fixed.config)['position_encoding'])
        a,b=model_size(config),model_size(fixed.config)
        self.assertEqual(a['parameters']-3*64*32,b['parameters'])
        self.assertEqual(3*64*32*4,b['fp32_buffer_bytes'])

    def test_fixed_position_checkpoint_resumes_exactly_and_rejects_mode_change(self):
        config=ModelConfig(rows=4,cols=4,embedding_dim=16,encoder_depth=1,position_encoding='sincos2d')
        options=dict(config=config,batch_size=2,seed=19)
        continuous=train_smoke(steps=4,**options)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'fixed.pt';first=train_smoke(steps=2,checkpoint=path,**options)
            second=train_smoke(steps=2,resume=path,**options)
            self.assertEqual(continuous['history'],first['history']+second['history'])
            with self.assertRaises(ValueError):train_smoke(steps=1,resume=path,**(options|dict(config=replace(config,position_encoding='learned'))))
        for changes in (dict(position_encoding='unknown'),dict(embedding_dim=6,heads=2,position_encoding='sincos2d')):
            with self.assertRaises(ValueError):ModelConfig(**changes)
