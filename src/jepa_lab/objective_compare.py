"""Fairness boundary for an objective-only contrast with shared controls."""
import torch

from .compare import validate_comparison_inputs
from .features import _sha
from .objective import objective_from_settings

SOURCES = ('prediction', 'context_variance', 'random', 'pixels')


def validate_objective_features(sources, candidates):
    if not isinstance(sources, dict) or set(sources) != set(SOURCES):
        raise ValueError('Objective contrast needs both trained arms and shared random/pixel controls')
    pairs = []
    for name in SOURCES[:2]:
        pairs.append(validate_comparison_inputs(dict(checkpoint=sources[name],
            random=sources['random'], pixels=sources['pixels']), candidates))
    first, second = [p[0].metadata for p in pairs]
    keys = ('config', 'completed_steps', 'pooling', 'encoder', 'normalization')
    if ({k: first.get(k) for k in keys} != {k: second.get(k) for k in keys}
            or type(first.get('completed_steps')) is not int or first['completed_steps'] < 1):
        raise ValueError('Objective arms must share architecture, training budget and extraction recipe')
    objectives = [objective_from_settings(m.get('training_settings')) for m in (first, second)]
    settings = [{k: v for k, v in m['training_settings'].items() if k != 'objective'} for m in (first, second)]
    hashes = [m.get('checkpoint_sha256') for m in (first, second)]
    if (objectives[0].variance_weight != 0 or objectives[1].variance_weight <= 0
            or settings[0] != settings[1] or any(not _sha(h) for h in hashes) or hashes[0] == hashes[1]
            or sources['random'][0].metadata.get('seed') != settings[0].get('seed')):
        raise ValueError('Vary only the objective, retain distinct trained checkpoints and exact-initialization control')
    for value, expected in zip(pairs[1], pairs[0]):
        if (value.indices != expected.indices or not torch.equal(value.labels, expected.labels)
                or value.manifest_sha256 != expected.manifest_sha256 or value.selection_sha256 != expected.selection_sha256):
            raise ValueError('Objective arms must use identical ordered examples')
    return pairs[0]
