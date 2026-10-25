"""Aggregate representation diagnostics; no labels, fitting, or quality thresholds."""

import torch


def _matrix(values):
    if (not isinstance(values, torch.Tensor) or values.device.type != 'cpu'
            or values.dtype != torch.float32 or values.ndim != 2
            or not 1 <= values.shape[0] <= 1000 or not 1 <= values.shape[1] <= 4096
            or not torch.isfinite(values).all()):
        raise ValueError('Use finite CPU float32 matrices with 1-1000 rows and 1-4096 columns')
    return values.detach().double()


@torch.inference_mode()
def paired_features(context, target):
    """Compare aligned rows; caller must verify image identity and ordering.

    Centered linear CKA ignores translation, rotation and global scale. Raw
    RMSE/cosines retain complementary coordinate information. Undefined ratios
    are null, including CKA for constant or singleton representations.
    """
    x, y = _matrix(context), _matrix(target)
    if x.shape != y.shape:
        raise ValueError('Paired features require identical row and coordinate shapes')
    xc, yc = x - x.mean(0), y - y.mean(0)
    gx, gy = xc @ xc.T, yc @ yc.T
    denominator = float(gx.norm() * gy.norm())
    cka = max(0., min(1., float((gx * gy).sum()) / denominator)) if denominator else None
    nx, ny = x.norm(dim=1), y.norm(dim=1)
    valid = (nx > 0) & (ny > 0)
    cosines = ((x[valid] * y[valid]).sum(1) / (nx[valid] * ny[valid])).clamp(-1, 1)
    return dict(count=len(x), feature_dim=x.shape[1], rmse=float((x-y).square().mean().sqrt()),
        mean_paired_cosine=float(cosines.mean()) if valid.any() else None,
        cosine_defined_rows=int(valid.sum()), centered_linear_cka=cka,
        limitation='Aligned supplied rows only; similarity is not downstream quality or evidence of temporal stability')
