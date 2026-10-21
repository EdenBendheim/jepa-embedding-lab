from copy import deepcopy
from dataclasses import replace
import tempfile
import unittest
from unittest.mock import patch

import torch

from jepa_lab.compare import fit_representation
from jepa_lab.probes import ProbeConfig
from jepa_lab.scale_compare import compare_scales, audit_scales
from test_position_compare import position_arms


def scale_arms():
    base = position_arms()['positionsincos2d']
    result = {}
    for scale, checkpoint in ((.02, 'a'), (1., 'b')):
        data = dict(pixels=base['pixels'])
        for rep in ('checkpoint', 'random'):
            config = base[rep][0].metadata['config'] | dict(position_scale=scale)
            metadata = base[rep][0].metadata | dict(config=config)
            if rep == 'checkpoint':
                metadata['checkpoint_sha256'] = checkpoint * 64
            data[rep] = tuple(replace(item, metadata=deepcopy(metadata)) for item in base[rep])
        result[f'scale{scale}'] = data
    return result


class ScaleComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_shared_baselines_saved_state_audit_and_unchanged_sources(self):
        arms = scale_arms()
        before = {name: {rep: [item.features.clone() for item in pair] for rep, pair in data.items()}
                  for name, data in arms.items()}
        rng = torch.get_rng_state().clone()
        candidates = (ProbeConfig(steps=3),)
        with tempfile.TemporaryDirectory() as folder, \
                patch('jepa_lab.width_compare.fit_representation', wraps=fit_representation) as fitted:
            report = compare_scales(arms, candidates, [29, 31], probe_dir=folder, ridge_alphas=[.1])
            self.assertEqual(10, fitted.call_count)
            self.assertEqual(list(arms), report['position_scales'])
            with patch('jepa_lab.width_compare.fit_representation', side_effect=AssertionError('no refit')):
                self.assertEqual(15, audit_scales(report, arms, candidates, [29, 31],
                                                 ridge_alphas=[.1])['verified_classifiers'])
            edited = deepcopy(report)
            edited['summary']['scale0.02-checkpoint']['mean_accuracy'] += .1
            with self.assertRaises(ValueError):
                audit_scales(edited, arms, candidates, [29, 31], ridge_alphas=[.1])
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        for name, data in arms.items():
            for rep, pair in data.items():
                for item, original in zip(pair, before[name][rep]):
                    self.assertTrue(torch.equal(item.features, original))

    def test_confounding_changes_fail_before_fitting(self):
        for change in ('mode', 'depth', 'scale', 'steps', 'seed', 'rows', 'pixels', 'checkpoint'):
            arms = scale_arms()
            data = arms['scale1.0']
            if change in ('mode', 'depth', 'scale'):
                updates = dict(mode=dict(position_encoding='learned'), depth=dict(encoder_depth=2),
                               scale=dict(position_scale=.02))[change]
                for rep in ('checkpoint', 'random'):
                    data[rep] = tuple(replace(i, metadata=i.metadata | dict(config=i.metadata['config'] | updates))
                                      for i in data[rep])
            if change == 'steps':
                data['checkpoint'] = tuple(replace(i, metadata=i.metadata | dict(completed_steps=4)) for i in data['checkpoint'])
            if change == 'seed':
                data['random'] = tuple(replace(i, metadata=i.metadata | dict(seed=31)) for i in data['random'])
            if change == 'rows':
                data['checkpoint'] = tuple(replace(i, indices=i.indices[::-1]) for i in data['checkpoint'])
            if change == 'pixels':
                data['pixels'] = (replace(data['pixels'][0], features=data['pixels'][0].features + .1), data['pixels'][1])
            if change == 'checkpoint':
                data['checkpoint'] = tuple(replace(i, metadata=i.metadata | dict(checkpoint_sha256='a' * 64)) for i in data['checkpoint'])
            with self.subTest(change=change), \
                    patch('jepa_lab.width_compare.fit_representation', side_effect=AssertionError('invalid fit')), \
                    self.assertRaises(ValueError):
                compare_scales(arms, (ProbeConfig(steps=2),), [29], probe_dir='unused', ridge_alphas=[.1])
