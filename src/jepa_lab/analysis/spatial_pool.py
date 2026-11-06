"""Fixed spatial pooling for full 8-by-8 image-token grids."""

import torch


@torch.inference_mode()
def spatial_mean(tokens, *, bins=2):
    """Concatenate cell means in row-major order, preserving channel order."""
    if (not isinstance(tokens, torch.Tensor) or tokens.device.type != 'cpu'
            or tokens.dtype != torch.float32 or tokens.ndim != 3
            or not 1 <= tokens.shape[0] <= 1000 or tokens.shape[1] != 64
            or not 1 <= tokens.shape[2] <= 512 or not torch.isfinite(tokens).all()
            or type(bins) is not int or bins not in (1, 2)):
        raise ValueError('Use finite CPU float32 [1-1000, 64, 1-512] tokens and bins 1 or 2')
    if bins == 1:
        return tokens.mean(1)
    batch, _, width = tokens.shape
    # Grid rows = cell-row, within-cell row; columns follow the same layout.
    grid = tokens.reshape(batch, bins, 8 // bins, bins, 8 // bins, width)
    return grid.mean(dim=(2, 4)).reshape(batch, bins * bins * width)
