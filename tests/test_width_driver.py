import hashlib,importlib.util,json
from pathlib import Path
import unittest
from unittest.mock import patch
import torch
import test_budget_driver as fixtures


class WidthDriverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def setUp(self):
        fixtures.BudgetDriverTests.setUp(self)
        path=Path(__file__).resolve().parents[1]/'experiments/run_width_20261015.py'
        spec=importlib.util.spec_from_file_location('width_driver',path);self.width=importlib.util.module_from_spec(spec);spec.loader.exec_module(self.width)
        d=json.loads(self.driver.PROTOCOL.read_text());d|=dict(widths=[16,32],heads=4,encoder_depth=1,predictor_depth=1,
            target_rows=2,target_cols=2,reused_checkpoints={})
        self.driver.PROTOCOL.write_text(json.dumps(d));self.width.PROTOCOL=self.driver.PROTOCOL
        self.width.PROTOCOL_SHA=hashlib.sha256(self.width.PROTOCOL.read_bytes()).hexdigest();self.width.ROOT=self.root
        p=patch.object(self.width,'CIFAR10Binary',self.driver.CIFAR10Binary);p.start();self.addCleanup(p.stop)

    def test_widths_match_image_order_and_completed_cache_does_not_train_or_read_images(self):
        result=self.width.training(self.directory)
        self.assertEqual(result['width16-seed29']['train_image_order_sha256'],result['width32-seed29']['train_image_order_sha256'])
        self.assertGreater(result['width32-seed29']['model_size']['parameters'],result['width16-seed29']['model_size']['parameters'])
        with patch('jepa_lab.training_budget.train_cifar10',side_effect=AssertionError('training')),patch.object(self.width.CIFAR10Binary,'__getitem__',side_effect=AssertionError('images')):
            self.assertEqual(result,self.width.training(self.directory,reuse_only=True))
