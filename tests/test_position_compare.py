from copy import deepcopy
from dataclasses import replace
import tempfile,unittest
from unittest.mock import patch
import torch

from jepa_lab.compare import fit_representation
from jepa_lab.position_compare import compare_positions,audit_positions
from jepa_lab.probes import ProbeConfig
from test_compare import sets


def position_arms():
    base=sets();result={}
    for mode,checkpoint in [('learned','a'),('sincos2d','b')]:
        data=dict(pixels=base['pixels'])
        for rep in ('checkpoint','random'):
            config=base[rep][0].metadata['config']|({} if mode=='learned' else dict(position_encoding=mode))
            metadata=base[rep][0].metadata|dict(config=config)
            metadata|=dict(completed_steps=3,checkpoint_sha256=checkpoint*64,training_settings=dict(seed=29)) if rep=='checkpoint' else dict(seed=29)
            data[rep]=tuple(replace(item,metadata=deepcopy(metadata)) for item in base[rep])
        result['position'+mode]=data
    return result


class PositionComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_shared_fits_exact_saved_states_and_tamper_rejection(self):
        arms=position_arms();candidates=(ProbeConfig(steps=3),)
        with tempfile.TemporaryDirectory() as folder,patch('jepa_lab.width_compare.fit_representation',wraps=fit_representation) as fitted:
            report=compare_positions(arms,candidates,[29,31],probe_dir=folder,ridge_alphas=[.1])
            self.assertEqual(10,fitted.call_count);self.assertEqual(list(arms),report['positions'])
            self.assertEqual(15,audit_positions(report,arms,candidates,[29,31],ridge_alphas=[.1])['verified_classifiers'])
            bad=deepcopy(report);bad['summary']['positionsincos2d-checkpoint']['mean_accuracy']+=.1
            with self.assertRaises(ValueError):audit_positions(bad,arms,candidates,[29,31],ridge_alphas=[.1])

    def test_width_depth_budget_control_seed_and_row_changes_fail_before_fitting(self):
        for change in ('width','depth','steps','seed','rows','pixels'):
            arms=position_arms();data=arms['positionsincos2d']
            if change in ('width','depth'):
                for rep in ('checkpoint','random'):
                    field='embedding_dim' if change=='width' else 'encoder_depth'
                    data[rep]=tuple(replace(i,metadata=i.metadata|dict(config=i.metadata['config']|{field:32 if change=='width' else 2})) for i in data[rep])
            if change=='steps':data['checkpoint']=tuple(replace(i,metadata=i.metadata|dict(completed_steps=4)) for i in data['checkpoint'])
            if change=='seed':data['random']=tuple(replace(i,metadata=i.metadata|dict(seed=31)) for i in data['random'])
            if change=='rows':data['checkpoint']=tuple(replace(i,indices=i.indices[::-1]) for i in data['checkpoint'])
            if change=='pixels':data['pixels']=(replace(data['pixels'][0],features=data['pixels'][0].features+.1),data['pixels'][1])
            with patch('jepa_lab.width_compare.fit_representation',side_effect=AssertionError('fit invalid arms')),self.assertRaises(ValueError):
                compare_positions(arms,(ProbeConfig(steps=2),),[29],probe_dir='unused',ridge_alphas=[.1])
