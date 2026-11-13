"""Multi-block predictions sharing one visible-context encoding."""

import torch
from torch.nn import functional as F

from ..model import JEPA, gather, scaled_positions
from .multiblock_masks import MultiBlockMask


def multi_block_forward(model, patches, masks, *, predictor_grouping='separate'):
    """Return [batch, blocks, tokens, width] predictions and stopped targets.

    The context and teacher each run once. Separate passes isolate blocks'
    mask tokens. Joint prediction lets all target mask tokens attend together.
    """
    if not isinstance(model, JEPA):
        raise ValueError('Use a JEPA model')
    if predictor_grouping not in ('separate', 'joint'):
        raise ValueError('Choose separate or joint predictor grouping')
    config = model.config
    if (not isinstance(patches, torch.Tensor) or patches.dtype != torch.float32
            or patches.device.type != 'cpu' or patches.ndim != 3 or not 1 <= len(patches) <= 32
            or patches.shape[1:] != (config.rows*config.cols, config.patch_dim)
            or not torch.isfinite(patches).all() or not isinstance(masks, (list, tuple))
            or len(masks) != len(patches) or any(not isinstance(mask, MultiBlockMask) for mask in masks)):
        raise ValueError('Use finite CPU float32 patches and one validated multi-block mask per image')
    first = masks[0]
    shapes = {(mask.rows, mask.cols, len(mask.context), len(mask.blocks), len(mask.blocks[0])) for mask in masks}
    if (len(shapes) != 1 or (first.rows, first.cols) != (config.rows, config.cols)):
        raise ValueError('Batch masks must share the model grid, context count, block count and block area')
    context = torch.tensor([mask.context for mask in masks], dtype=torch.long)
    targets = [torch.tensor([mask.blocks[block] for mask in masks], dtype=torch.long)
               for block in range(len(first.blocks))]
    encoded = model.context_encoder(patches, context)
    visible = model.predictor_projection(encoded) + scaled_positions(
        model.predictor_positions, context, config.position_scale)
    groups = [torch.cat(targets, dim=1)] if predictor_grouping == 'joint' else targets
    predictions = []
    for target in groups:
        tokens = model.mask_token.expand(len(patches), target.shape[1], -1) + scaled_positions(
            model.predictor_positions, target, config.position_scale)
        predictions.append(model.output(model.predictor(torch.cat((visible, tokens), dim=1))[:, -target.shape[1]:]))
    with torch.no_grad():
        full = torch.arange(patches.shape[1]).expand(len(patches), -1)
        teacher = model.target_encoder(patches, full)
        teacher_targets = torch.stack([F.layer_norm(gather(teacher, target), (config.embedding_dim,))
                                       for target in targets], dim=1)
    if predictor_grouping == 'joint':
        predictions = predictions[0].reshape_as(teacher_targets)
    else:
        predictions = torch.stack(predictions, dim=1)
    if not torch.isfinite(predictions).all() or not torch.isfinite(teacher_targets).all():
        raise ValueError('Multi-block model produced non-finite embeddings')
    return predictions, teacher_targets


def multi_block_loss(model, patches, masks, *, predictor_grouping='separate'):
    predictions, targets = multi_block_forward(model, patches, masks, predictor_grouping=predictor_grouping)
    # Equal-sized blocks make this an equal mean over images, blocks, tokens and channels.
    return F.smooth_l1_loss(predictions, targets)
