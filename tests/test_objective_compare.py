from copy import deepcopy
from dataclasses import replace
import unittest
import torch

from jepa_lab.objective_compare import validate_objective_features
from jepa_lab.probes import ProbeConfig
from test_position_compare import position_arms


def objective_sources():
    base = position_arms()['positionsincos2d']
    out = dict(prediction=base['checkpoint'], random=base['random'], pixels=base['pixels'])
    out['context_variance'] = tuple(replace(i, metadata=i.metadata | dict(checkpoint_sha256='c'*64,
        training_settings=i.metadata['training_settings'] | dict(objective=dict(
            context_variance_weight=.1, context_std_floor=1.)))) for i in base['checkpoint'])
    return out


class ObjectiveComparisonTests(unittest.TestCase):
    def test_shared_controls_and_only_objective_changes_leave_inputs_intact(self):
        sources = objective_sources()
        before = {k: [i.features.clone() for i in p] for k, p in sources.items()}
        rng = torch.get_rng_state().clone()
        self.assertEqual(sources['prediction'], validate_objective_features(sources, (ProbeConfig(steps=2),)))
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        for key, pair in sources.items():
            self.assertTrue(all(torch.equal(i.features, old) for i, old in zip(pair, before[key])))

    def test_confounding_recipe_objective_seed_or_row_changes_fail(self):
        for mutation in ('budget', 'recipe', 'architecture', 'objective', 'checkpoint', 'rows', 'random'):
            data = objective_sources()
            pair = data['context_variance']
            def update(change):
                data['context_variance'] = tuple(replace(i, metadata=deepcopy(i.metadata) | change) for i in pair)
            if mutation == 'budget': update(dict(completed_steps=4))
            if mutation == 'recipe': update(dict(training_settings=pair[0].metadata['training_settings'] | dict(seed=31)))
            if mutation == 'architecture': update(dict(config=pair[0].metadata['config'] | dict(encoder_depth=3)))
            if mutation == 'objective': update(dict(training_settings=dict(seed=29)))
            if mutation == 'checkpoint': update(dict(checkpoint_sha256=data['prediction'][0].metadata['checkpoint_sha256']))
            if mutation == 'rows': data['context_variance'] = tuple(replace(i, indices=i.indices[::-1]) for i in pair)
            if mutation == 'random': data['random'] = tuple(replace(i, metadata=i.metadata | dict(seed=31)) for i in data['random'])
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate_objective_features(data, (ProbeConfig(steps=2),))
