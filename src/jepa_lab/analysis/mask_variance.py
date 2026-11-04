"""Separate image, mask, and interaction effects in crossed fixed views."""

import torch


@torch.inference_mode()
def crossed_variance(views: torch.Tensor) -> dict:
    """Summarize [image, mask, coordinate] means on a balanced crossed design.

    Each mask must be applied to every image in the same column order. The
    residual measures image-dependent responses to masks. Main effects and the
    residual are orthogonal under uniform weighting of this particular grid.
    """
    if (not isinstance(views, torch.Tensor) or views.ndim != 3
            or views.dtype != torch.float32 or views.device.type != 'cpu'
            or not 2 <= views.shape[0] <= 1000 or not 2 <= views.shape[1] <= 8
            or not 1 <= views.shape[2] <= 4096 or views.numel() > 4_000_000
            or not torch.isfinite(views).all()):
        raise ValueError('Use finite bounded CPU float32 [2-1000 images, 2-8 masks, coordinates]')
    values = views.double()
    centered = values - values.mean(dim=(0, 1))
    image_effect = centered.mean(dim=1, keepdim=True)
    mask_effect = centered.mean(dim=0, keepdim=True)
    interaction = centered - image_effect - mask_effect
    total = float(centered.square().mean())
    components = {
        'image': float(image_effect.square().mean()),
        'mask': float(mask_effect.square().mean()),
        'interaction': float(interaction.square().mean()),
    }
    return {
        'images': values.shape[0],
        'masks': values.shape[1],
        'coordinates': values.shape[2],
        'total_variance': total,
        'variance': components,
        'fraction': {name: value / total if total else None for name, value in components.items()},
    }
