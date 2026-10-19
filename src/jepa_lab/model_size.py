"""Count model capacity and bound the public-data local training architecture."""
from dataclasses import asdict
import torch
from .model import JEPA,ModelConfig, config_dict


def validate_local_config(config):
    if (not isinstance(config,ModelConfig) or (config.rows,config.cols,config.patch_dim)!=(8,8,48)
            or not 8<=config.embedding_dim<=128 or config.heads>8
            or config.encoder_depth>4 or config.predictor_depth>2):
        raise ValueError('Local RGB training allows width 8-128, <=8 heads, <=4 encoder and <=2 predictor blocks')
    return config


def model_size(config):
    validate_local_config(config)
    with torch.random.fork_rng(devices=[]):model=JEPA(config)
    groups={name:dict(parameters=0,trainable_parameters=0) for name in ('context','target','predictor')}
    for name,p in model.named_parameters():
        group='context' if name.startswith('context_encoder.') else 'target' if name.startswith('target_encoder.') else 'predictor'
        groups[group]['parameters']+=p.numel()
        if p.requires_grad:groups[group]['trainable_parameters']+=p.numel()
    total=sum(g['parameters'] for g in groups.values());trainable=sum(g['trainable_parameters'] for g in groups.values())
    buffers=sum(b.numel() for b in model.buffers())
    result=dict(config=config_dict(config),groups=groups,parameters=total,trainable_parameters=trainable,
        frozen_parameters=total-trainable,fp32_parameter_bytes=4*total,
        estimated_fp32_parameters_gradients_adam_moments_bytes=4*(total+buffers+3*trainable),
        limitation='Persistent tensor estimate only; excludes activations, temporary buffers, optimizer scalars and allocator overhead. Equal steps do not mean equal compute.')

    if buffers:result.update(buffer_elements=buffers,fp32_buffer_bytes=4*buffers)
    return result
