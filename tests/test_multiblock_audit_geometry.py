import unittest

import torch

from experiments.multiblock_audit_geometry import measure_geometry


class AuditGeometryTests(unittest.TestCase):
    def test_layout_pairs_same_image_and_milestones_pair_same_seed(self):
        values = torch.tensor([[1., 1.], [1., -1.], [-1., 1.], [-1., -1.]])
        sources = dict(separate=values, joint_matched=values, joint_extended=values,
                       random=torch.ones_like(values))
        panel = {'rgb': torch.zeros(4, 6)}
        for name, tensor in sources.items():
            panel[f'29/{name}/original'] = tensor
            panel[f'29/{name}/shuffled'] = tensor.flip(0)
        result = measure_geometry(panel, [29])
        self.assertEqual(0, result['paired_milestones']['29']['joint_matched-vs-separate']['rmse'])
        self.assertIsNone(result['paired_milestones']['29']['joint_extended-vs-random']['centered_linear_cka'])
        self.assertGreater(result['paired_layout']['29/separate']['rmse'], 0)
        self.assertEqual(0, result['paired_layout']['29/random']['rmse'])
        self.assertEqual(set(panel)-{'rgb'}, {f'{source}/{view}' for source, views in result['views'].items()
                                               for view in views})
