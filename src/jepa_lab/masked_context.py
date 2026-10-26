"""Training-row diagnostics for fixed-mask pooling and context attention."""

import hashlib
import json

import torch

from .cifar10 import patchify_rgb
from .feature_geometry import feature_geometry
from .masking import block_mask
from .model import gather
from .stability import paired_features


@torch.inference_mode()
def masked_context_report(model, images, *, mask_seeds=(29, 31, 37, 41),
                          target_rows=2, target_cols=2, context_fraction=.75,
                          batch_size=16):
    """Compare full pooling, subset pooling and re-encoded visible context.

    Every image receives the same fixed views. Subset pooling takes visible
    positions from full-attention outputs, so it is a diagnostic control and
    MUST NOT be used as a leak-free training context. Re-encoding gathers raw
    visible patches before attention, matching the model's training boundary.
    """
    if (type(batch_size) is not int or not 1 <= batch_size <= 32
            or not isinstance(images, torch.Tensor) or images.device.type != 'cpu'
            or images.dtype != torch.uint8 or images.ndim != 4
            or not 1 <= len(images) <= 1000 or images.shape[1:] != (3, 32, 32)):
        raise ValueError('Use 1-1000 CPU uint8 RGB 32x32 images and batch size 1-32')
    if (not isinstance(mask_seeds, (tuple, list)) or not 2 <= len(mask_seeds) <= 8
            or any(type(s) is not int or not 0 <= s < 2**63 for s in mask_seeds)
            or len(set(mask_seeds)) != len(mask_seeds)):
        raise ValueError('Use 2-8 distinct nonnegative integer mask seeds below 2**63')
    config = model.config
    if (config.rows, config.cols, config.patch_dim) != (8, 8, 48):
        raise ValueError('Masked RGB diagnostic requires an 8x8 grid of 48-value patches')
    masks = [block_mask(8, 8, target_rows=target_rows, target_cols=target_cols,
                        context_fraction=context_fraction, seed=s) for s in mask_seeds]
    positions = [torch.tensor(m.context) for m in masks]
    recipe = dict(mask_seeds=list(mask_seeds), target_rows=target_rows,
                  target_cols=target_cols, context_fraction=context_fraction,
                  context_positions=[list(m.context) for m in masks])
    recipe_hash = hashlib.sha256(json.dumps(recipe, sort_keys=True,
                                separators=(',', ':')).encode()).hexdigest()
    full, subsets, masked = [], [[] for _ in masks], [[] for _ in masks]
    modes = [(module, module.training) for module in model.modules()]
    try:
        model.eval()
        for start in range(0, len(images), batch_size):
            patches = patchify_rgb(images[start:start+batch_size])
            all_positions = torch.arange(64).expand(len(patches), -1)
            tokens = model.context_encoder(patches, all_positions)
            full.append(tokens.mean(1))
            for i, pos in enumerate(positions):
                index = pos.expand(len(patches), -1)
                subsets[i].append(gather(tokens, index).mean(1))
                masked[i].append(model.context_encoder(patches, index).mean(1))
    finally:
        # Preserve even callers with mixed child train/eval modes.
        for module, training in modes:
            module.training = training
    full = torch.cat(full)
    subsets = [torch.cat(x) for x in subsets]
    masked = [torch.cat(x) for x in masked]
    values = torch.stack(masked, dim=1).double()
    image_means = values.mean(1)
    within = float((values - image_means[:, None]).square().mean())
    between = float((image_means - image_means.mean(0)).square().mean())
    total = float((values - values.mean((0, 1))).square().mean())
    return dict(count=len(images), views=len(masks), visible_tokens=len(masks[0].context),
        mask_recipe_sha256=recipe_hash, mask_seeds=list(mask_seeds),
        target_rows=target_rows, target_cols=target_cols, context_fraction=context_fraction,
        full_geometry=feature_geometry(full),
        masked_mean_geometry=feature_geometry(image_means.float()),
        within_image_mask_variance=within, between_image_mean_variance=between,
        total_variance=total, mask_variance_fraction=within/total if total else None,
        per_view=[dict(seed=seed, masked_geometry=feature_geometry(m),
            full_to_subset=paired_features(full, s),
            full_to_masked=paired_features(full, m),
            subset_to_masked=paired_features(s, m))
            for seed, s, m in zip(mask_seeds, subsets, masked)],
        limitation='Fixed-mask training-row endpoint diagnosis only; subset control sees full-image attention; no accuracy, robustness or temporal-stability claim')
