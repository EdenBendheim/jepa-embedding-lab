"""Inspect frozen scale-run encoders on the original fitting images only."""

import hashlib
import json

import torch

from .cifar10 import CIFAR10Binary, patchify_rgb
from .embeddings import load_encoder, pixel_features, random_encoder
from .feature_geometry import feature_geometry
from .model import config_dict
from .stability import paired_features, parameter_lag, token_spread


@torch.inference_mode()
def _tokens(model, images, batch_size):
    model.eval()
    context, target = [], []
    for start in range(0, len(images), batch_size):
        patches = patchify_rgb(images[start:start+batch_size])
        positions = torch.arange(64).expand(len(patches), -1)
        context.append(model.context_encoder(patches, positions))
        target.append(model.target_encoder(patches, positions))
    return torch.cat(context), torch.cat(target)


def inspect_training(run, *, batch_size=16):
    """Reuse completed training, then compare fixed full-image encoder endpoints.

    No optimization, masked prediction, class scoring, validation extraction or
    test extraction occurs. Labels are used by existing split/selection replay,
    never by these representation diagnostics.
    """
    if type(batch_size) is not int or not 1 <= batch_size <= 32:
        raise ValueError('Audit batch size must be 1-32')
    reports = run.training(reuse_only=True)
    d, manifest, reference, _, _ = run.inputs()
    indices = reference['partitions']['train']['indices']
    if not 1 <= len(indices) <= 1000:
        raise ValueError('Audit needs 1-1000 original fitting images')
    if not set(indices).issubset(manifest['partitions']['train']['indices']):
        raise ValueError('Audit selection must be training-only')
    train = CIFAR10Binary(run.root)
    images = torch.stack([train[index][0] for index in indices])
    arms = {}
    for scale in d['position_scales']:
        for seed in d['training_seeds']:
            name = f'scale{scale}-seed{seed}'
            model, metadata = load_encoder(run.checkpoint(d, scale, seed), manifest['manifest_sha256'])
            if (metadata['checkpoint_sha256'] != reports[name]['checkpoint_sha256']
                    or metadata['config'] != config_dict(run.config(d, scale))
                    or metadata['completed_steps'] != d['total_steps']):
                raise ValueError('Audit checkpoint differs from the verified completed run')
            random = random_encoder(run.config(d, scale), seed)
            context, target = _tokens(model, images, batch_size)
            random_context, random_target = _tokens(random, images, batch_size)
            arms[name] = dict(checkpoint=metadata,
                context=token_spread(context), target=token_spread(target),
                random_context=token_spread(random_context), random_target=token_spread(random_target),
                context_target=paired_features(context.mean(1), target.mean(1)),
                context_random=paired_features(context.mean(1), random_context.mean(1)),
                parameter_lag=parameter_lag(model.context_encoder, model.target_encoder))
    return dict(version=1, protocol_sha256=run.expected_sha, manifest_sha256=manifest['manifest_sha256'],
        training_selection_sha256=reference['selection_sha256'],
        ordered_training_indices_sha256=hashlib.sha256(json.dumps(indices, separators=(',', ':')).encode()).hexdigest(),
        count=len(indices), batch_size=batch_size, runtime=dict(torch=str(torch.__version__), cpu_threads=torch.get_num_threads()),
        arms=arms, pixels=feature_geometry(pixel_features(images)),
        images_sampled=True, partition='train', classifiers_fitted=False, models_trained=False,
        validation_images_read=False, test_evaluated=False,
        limitation='Exploratory training-only endpoint diagnostics, not downstream quality, temporal stability or a new held-out comparison')
