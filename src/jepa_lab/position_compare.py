"""Compare only position encodings at fixed architecture/data/probe budgets."""
from dataclasses import asdict
import torch

from .compare import validate_comparison_inputs
from .features import _sha
from .model import ModelConfig
from .width_compare import compare_arms,audit_arms


def validate_positions(arms,candidates):
    if not isinstance(arms,dict) or list(arms)!=['positionlearned','positionsincos2d']:
        raise ValueError('Declare learned and sincos2d position arms in that order')
    reference=validate_comparison_inputs(arms['positionlearned'],candidates)
    anchor=reference[0].metadata
    fixed={key:anchor.get(key) for key in ('completed_steps','training_settings','pooling','encoder','normalization')}
    if (type(fixed['completed_steps']) is not int or fixed['completed_steps']<1 or not isinstance(fixed['training_settings'],dict)):
        raise ValueError('Positions require recorded training identity')
    configs=[];checkpoints=[]
    for name,data in arms.items():
        pair=validate_comparison_inputs(data,candidates);metadata=pair[0].metadata
        config=ModelConfig(**metadata['config']);wire=asdict(config);mode=wire.pop('position_encoding');configs.append(wire)
        checkpoints.append(metadata.get('checkpoint_sha256'))
        if (name!='position'+mode or {key:metadata.get(key) for key in fixed}!=fixed
                or data['random'][0].metadata.get('seed')!=fixed['training_settings'].get('seed')):
            raise ValueError('Positions must share training/pooling and exact-initialization control seeds')
        for item,expected in zip(pair,reference):
            if (item.indices!=expected.indices or not torch.equal(item.labels,expected.labels)
                    or item.manifest_sha256!=expected.manifest_sha256 or item.selection_sha256!=expected.selection_sha256):
                raise ValueError('Position arms require identical ordered examples')
        for item,expected in zip(data['pixels'],arms['positionlearned']['pixels']):
            if item.artifact_sha256!=expected.artifact_sha256 or item.metadata!=expected.metadata or not torch.equal(item.features,expected.features):
                raise ValueError('Position arms must share exact pixel artifacts')
    if configs[0]!=configs[1] or any(not _sha(h) for h in checkpoints) or len(set(checkpoints))!=2:
        raise ValueError('Position arms must differ only in position mode and have distinct checkpoint identities')
    return reference


LIMITATION='Only positional encoding varies in the declared architecture. Fixed buffers reduce trainable parameters; equal steps are not equal capacity. Shared pixels and exact-initialization random controls use identical evaluation budgets.'


def compare_positions(arms,candidates,seeds,*,probe_dir,ridge_alphas):
    return compare_arms(arms,candidates,seeds,probe_dir=probe_dir,ridge_alphas=ridge_alphas,
        validator=validate_positions,axis='positions',limitation=LIMITATION)


def audit_positions(report,arms,candidates,seeds,*,ridge_alphas):
    return audit_arms(report,arms,candidates,seeds,ridge_alphas=ridge_alphas,validator=validate_positions,axis='positions')
