"""Inspect matched objective checkpoints on the original fitting images."""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from ..cifar10 import CIFAR10Binary, patchify_rgb
from ..embeddings import image_embeddings, load_encoder, random_encoder
from ..feature_geometry import feature_geometry
from ..masking import block_mask
from ..model import ModelConfig
from ..objective_run import ObjectiveRun
from ..stability import paired_features
from .mask_variance import crossed_variance
from .prediction_controls import prediction_controls

MASK_SEEDS = (29, 31, 37, 41)
BATCH_SIZE = 16


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


@torch.inference_mode()
def _measure(model, images, masks):
    context = []
    target = []
    for batch in images.split(BATCH_SIZE):
        context.append(image_embeddings(model, batch))
        target.append(image_embeddings(model, batch, encoder='target'))
    context, target = torch.cat(context), torch.cat(target)
    batches = [patchify_rgb(batch) for batch in images.split(BATCH_SIZE)]
    pooled_views = []
    controls = []
    for seed, mask in zip(MASK_SEEDS, masks, strict=True):
        predictions, targets, pooled = [], [], []
        for patches in batches:
            prediction, teacher, visible = model(patches, mask.context, mask.target, return_context=True)
            predictions.append(prediction)
            targets.append(teacher)
            pooled.append(visible.mean(dim=1))
        pooled_views.append(torch.cat(pooled))
        controls.append(dict(seed=seed, **prediction_controls(torch.cat(predictions), torch.cat(targets))))
    views = torch.stack(pooled_views, dim=1)
    return dict(full_context=feature_geometry(context), full_target=feature_geometry(target),
                full_context_target=paired_features(context, target),
                mean_visible_context=feature_geometry(views.mean(dim=1)),
                crossed_variance=crossed_variance(views), prediction_controls=controls)


def inspect_training(run: ObjectiveRun) -> dict:
    """Replay the locked checkpoints and read only the pinned fitting selection."""
    reports = run.training(reuse_only=True)
    recipe, manifest, fitting, _, _, _ = run.inputs()
    indices = fitting['partitions']['train']['indices']
    if (not 2 <= len(indices) <= 1000 or len(indices) != len(set(indices))
            or not set(indices).issubset(manifest['partitions']['train']['indices'])):
        raise ValueError('Diagnostics require 2-1000 distinct rows from the fixed training split')
    config = ModelConfig(**recipe['config'])
    if len(indices) * config.embedding_dim * max(len(MASK_SEEDS),
            recipe['mask']['target_rows'] * recipe['mask']['target_cols']) > 4_000_000:
        raise ValueError('Diagnostic tensors exceed the element budget')
    masks = [block_mask(config.rows, config.cols, **recipe['mask'], seed=seed) for seed in MASK_SEEDS]
    models = {}
    for seed in recipe['training_seeds']:
        for arm in recipe['arms']:
            name = f'{arm["name"]}-seed{seed}'
            model, metadata = load_encoder(run.checkpoint(arm, seed), manifest['manifest_sha256'])
            if (metadata['checkpoint_sha256'] != reports[name]['checkpoint_sha256']
                    or metadata['config'] != recipe['config']
                    or metadata['completed_steps'] != recipe['total_steps']
                    or metadata['training_settings'] != run.settings(recipe, arm, seed)):
                raise ValueError('Diagnostic checkpoint differs from the verified completed run')
            models[name] = (model, metadata)
        models[f'random-seed{seed}'] = (random_encoder(config, seed), dict(seed=seed, config=recipe['config']))
    dataset = CIFAR10Binary(run.root)
    images = torch.stack([dataset[index][0] for index in indices])
    arms = {name: dict(identity=identity, **_measure(model, images, masks))
            for name, (model, identity) in models.items()}
    return dict(version=1, protocol_sha256=run.expected_sha, manifest_sha256=manifest['manifest_sha256'],
                training_selection_sha256=fitting['selection_sha256'],
                ordered_training_indices_sha256=_digest(indices), count=len(indices),
                mask_seeds=list(MASK_SEEDS), mask_recipe=recipe['mask'],
                mask_schedule_sha256=_digest([[mask.context, mask.target] for mask in masks]),
                batch_size=BATCH_SIZE, runtime=recipe['runtime'], arms=arms,
                partition='train', models_trained=False, classifiers_fitted=False,
                validation_images_read=False, test_images_read=False,
                limitation='Fixed-view, fitting-image endpoint diagnostics. Each model has its own teacher targets. '
                           'Cross-model loss levels cannot establish downstream quality or generalization.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--expected-sha256', required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    try:
        run = ObjectiveRun(args.root, args.directory, protocol=args.protocol, expected_sha=args.expected_sha256)
        print(json.dumps(inspect_training(run), indent=2, allow_nan=False))
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
