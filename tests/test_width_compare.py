from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import tempfile,unittest
from unittest.mock import patch
import torch

from jepa_lab.compare import fit_representation
from jepa_lab.probes import ProbeConfig
from jepa_lab.width_compare import compare_widths,validate_widths,audit_widths
from test_compare import sets


def width_arms():
    base=sets();result={}
    for width in (16,32):
        current=dict(pixels=base['pixels'])
        for name in ('checkpoint','random'):
            metadata=base[name][0].metadata|dict(config=base[name][0].metadata['config']|dict(embedding_dim=width))
            metadata|=dict(completed_steps=3,checkpoint_sha256=str(width%10)*64,training_settings=dict(seed=29)) if name=='checkpoint' else dict(seed=29)
            current[name]=tuple(replace(item,features=item.features.repeat(1,width//16),metadata=deepcopy(metadata)) for item in base[name])
        result[f'width{width}']=current
    return result


class WidthComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_pixel_fits_are_shared_and_saved_states_pairs_and_aggregates_reproduce(self):
        arms=width_arms();candidates=(ProbeConfig(steps=3),)
        with tempfile.TemporaryDirectory() as folder,patch('jepa_lab.width_compare.fit_representation',wraps=fit_representation) as fitted:
            report=compare_widths(arms,candidates,[29,31],probe_dir=folder,ridge_alphas=[.1])
            self.assertEqual(10,fitted.call_count)
            self.assertEqual(15,audit_widths(report,arms,candidates,[29,31],ridge_alphas=[.1])['verified_classifiers'])
            bad=deepcopy(report);bad['runs'][0]['paired_validation']['adamw']['width32']['vs_pixels']['both_correct']+=1
            with self.assertRaises(ValueError):audit_widths(bad,arms,candidates,[29,31],ridge_alphas=[.1])

    def test_changed_steps_architecture_seed_pixels_or_order_fail_before_fitting(self):
        for change in ('steps','architecture','seed','pixels','order'):
            arms=width_arms();data=arms['width32']
            if change=='steps':data['checkpoint']=tuple(replace(i,metadata=i.metadata|dict(completed_steps=4)) for i in data['checkpoint'])
            if change=='architecture':
                for name in ('checkpoint','random'):data[name]=tuple(replace(i,metadata=i.metadata|dict(config=i.metadata['config']|dict(encoder_depth=2))) for i in data[name])
            if change=='seed':data['random']=tuple(replace(i,metadata=i.metadata|dict(seed=31)) for i in data['random'])
            if change=='pixels':data['pixels']=(replace(data['pixels'][0],features=data['pixels'][0].features+.1),data['pixels'][1])
            if change=='order':data['checkpoint']=tuple(replace(i,indices=i.indices[::-1]) for i in data['checkpoint'])
            with patch('jepa_lab.width_compare.fit_representation',side_effect=AssertionError('fit invalid arms')),self.assertRaises(ValueError):
                compare_widths(arms,(ProbeConfig(steps=2),),[29],probe_dir='unused',ridge_alphas=[.1])
