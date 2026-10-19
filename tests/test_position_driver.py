import hashlib,importlib.util,json
from pathlib import Path
import unittest
from unittest.mock import patch
import torch
import test_budget_driver as fixtures


class PositionDriverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def setUp(self):
        fixtures.BudgetDriverTests.setUp(self)
        path=Path(__file__).resolve().parents[1]/'experiments/run_position_20261018.py'
        spec=importlib.util.spec_from_file_location('position_driver',path);self.position=importlib.util.module_from_spec(spec);spec.loader.exec_module(self.position)
        d=json.loads(self.driver.PROTOCOL.read_text());d|=dict(positions=['learned','sincos2d'],embedding_dim=16,heads=4,encoder_depth=1,predictor_depth=1,
            target_rows=2,target_cols=2,reused_checkpoints={},neighbor_counts=[1,3])
        self.driver.PROTOCOL.write_text(json.dumps(d));self.position.PROTOCOL=self.driver.PROTOCOL
        self.position.PROTOCOL_SHA=hashlib.sha256(self.position.PROTOCOL.read_bytes()).hexdigest();self.position.ROOT=self.root
        p=patch.object(self.position,'CIFAR10Binary',self.driver.CIFAR10Binary);p.start();self.addCleanup(p.stop)

    def test_modes_match_actual_order_and_cache_retry_never_trains_or_reads_images(self):
        result=self.position.training(self.directory)
        self.assertEqual(result['positionlearned-seed29']['train_image_order_sha256'],result['positionsincos2d-seed29']['train_image_order_sha256'])
        self.assertLess(result['positionsincos2d-seed29']['model_size']['trainable_parameters'],result['positionlearned-seed29']['model_size']['trainable_parameters'])
        with patch('jepa_lab.training_budget.train_cifar10',side_effect=AssertionError('training')),patch.object(self.position.CIFAR10Binary,'__getitem__',side_effect=AssertionError('images')):
            self.assertEqual(result,self.position.training(self.directory,reuse_only=True))
        path=self.position.checkpoint_path(self.directory,json.loads(self.position.PROTOCOL.read_text()),'sincos2d',29)
        state=torch.load(path,weights_only=True);state['training_history'][0]['train_indices'].reverse();torch.save(state,path)
        with patch.object(self.position,'run_training_budget',side_effect=AssertionError('trained before preflight')),self.assertRaises(ValueError):self.position.training(self.directory)
