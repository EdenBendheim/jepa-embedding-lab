"""Compare positional amplitude with fixed architecture and matched controls."""
from dataclasses import asdict

import torch

from .compare import validate_comparison_inputs
from .features import _sha
from .model import ModelConfig
from .width_compare import compare_arms, audit_arms


def validate_scales(arms, candidates):
    if not isinstance(arms, dict) or not 2 <= len(arms) <= 3:
        raise ValueError('Declare two or three increasing positional-scale arms')
    first = next(iter(arms.values()))
    reference = validate_comparison_inputs(first, candidates)
    anchor = reference[0].metadata
    fields = ('completed_steps', 'training_settings', 'pooling', 'encoder', 'normalization')
    fixed = {key: anchor.get(key) for key in fields}
    if (type(fixed['completed_steps']) is not int or fixed['completed_steps'] < 1
            or not isinstance(fixed['training_settings'], dict)):
        raise ValueError('Scale arms require recorded training identity')
    configs, scales, checkpoints = [], [], []
    for name, data in arms.items():
        pair = validate_comparison_inputs(data, candidates)
        metadata = pair[0].metadata
        config = asdict(ModelConfig(**metadata['config']))
        scale = config.pop('position_scale')
        configs.append(config)
        scales.append(scale)
        checkpoints.append(metadata.get('checkpoint_sha256'))
        if (name != f'scale{scale}' or {key: metadata.get(key) for key in fields} != fixed
                or data['random'][0].metadata.get('seed') != fixed['training_settings'].get('seed')):
            raise ValueError('Scale arms must share training identity and exact-initialization control seeds')
        for item, expected in zip(pair, reference):
            if (item.indices != expected.indices or not torch.equal(item.labels, expected.labels)
                    or item.manifest_sha256 != expected.manifest_sha256
                    or item.selection_sha256 != expected.selection_sha256):
                raise ValueError('Scale arms require identical ordered examples')
        for item, expected in zip(data['pixels'], first['pixels']):
            if (item.artifact_sha256 != expected.artifact_sha256 or item.metadata != expected.metadata
                    or not torch.equal(item.features, expected.features)):
                raise ValueError('Scale arms must share exact pixel artifacts')
    if (scales != sorted(set(scales)) or any(config != configs[0] for config in configs)
            or any(not _sha(value) for value in checkpoints) or len(set(checkpoints)) != len(arms)):
        raise ValueError('Arms must vary only scale, in increasing order, with distinct checkpoints')
    return reference


LIMITATION = ('Only positional amplitude varies in the recorded architecture; parameter counts stay fixed. '
              'Shared pixel heads and exact-initialization random controls use identical evaluation budgets. '
              'Metadata alone does not prove training ancestry or actual image-order equality.')


def compare_scales(arms, candidates, seeds, *, probe_dir, ridge_alphas):
    return compare_arms(arms, candidates, seeds, probe_dir=probe_dir, ridge_alphas=ridge_alphas,
                        validator=validate_scales, axis='position_scales', limitation=LIMITATION)


def audit_scales(report, arms, candidates, seeds, *, ridge_alphas):
    return audit_arms(report, arms, candidates, seeds, ridge_alphas=ridge_alphas,
                      validator=validate_scales, axis='position_scales')
