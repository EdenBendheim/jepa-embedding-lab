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


@torch.inference_mode()
def token_spread(tokens):
    """Decompose population variance into within-image and between-image parts.

    All images have the same token count. Therefore total variance equals the
    sum of mean within-image variance and variance of image means. Large token
    variance can coexist with identical pooled image embeddings.
    """
    if (not isinstance(tokens, torch.Tensor) or tokens.device.type != 'cpu'
            or tokens.dtype != torch.float32 or tokens.ndim != 3
            or not 1 <= tokens.shape[0] <= 1000 or not 1 <= tokens.shape[1] <= 64
            or not 1 <= tokens.shape[2] <= 4096 or tokens.numel() > 4_000_000
            or not torch.isfinite(tokens).all()):
        raise ValueError('Use finite bounded CPU float32 [images, 1-64 tokens, dimensions]')
    from .feature_geometry import feature_geometry
    values = tokens.detach().double()
    pooled = values.mean(1)
    within = float((values - pooled[:, None]).square().mean())
    between = float((pooled - pooled.mean(0)).square().mean())
    total = float((values - values.mean((0, 1))).square().mean())
    return dict(count=len(values), tokens_per_image=values.shape[1], feature_dim=values.shape[2],
        within_image_population_variance=within, between_image_population_variance=between,
        total_population_variance=total, between_fraction=between/total if total else None,
        pooled_geometry=feature_geometry(pooled.float()),
        limitation='Full-image spatial decomposition at one checkpoint; no classification or temporal-stability guarantee')


@torch.inference_mode()
def parameter_lag(context, target):
    """Measure an encoder pair's current weight gap, without performing EMA.

    This is an endpoint comparison, not an update-rate or training-trajectory
    estimate. Fixed positional buffers are checked separately from parameters.
    """
    left, right = dict(context.named_parameters()), dict(target.named_parameters())
    lb, rb = dict(context.named_buffers()), dict(target.named_buffers())
    if not left or left.keys() != right.keys() or lb.keys() != rb.keys():
        raise ValueError('Encoders must have matching named parameters and buffers')
    count, squared_gap, squared_target = 0, 0., 0.
    rows = {}
    for name, x in left.items():
        y = right[name]
        if (x.shape != y.shape or x.device.type != 'cpu' or y.device.type != 'cpu'
                or x.dtype != torch.float32 or y.dtype != torch.float32
                or not x.numel() or not torch.isfinite(x).all() or not torch.isfinite(y).all()):
            raise ValueError('Encoder parameters must be matching finite CPU float32 tensors')
        gap = float((x.detach().double()-y.detach().double()).square().sum())
        target_energy = float(y.detach().double().square().sum())
        count += x.numel(); squared_gap += gap; squared_target += target_energy
        rows[name] = dict(count=x.numel(), rmse=(gap/x.numel())**.5,
            relative_l2=(gap/target_energy)**.5 if target_energy else None)
    for name, x in lb.items():
        y = rb[name]
        if (x.shape != y.shape or x.dtype != y.dtype or x.device.type != 'cpu'
                or y.device.type != 'cpu' or not torch.isfinite(x).all() or not torch.isfinite(y).all()):
            raise ValueError('Encoder buffers must have matching finite CPU layouts')
    return dict(parameter_count=count, rmse=(squared_gap/count)**.5,
        relative_l2=(squared_gap/squared_target)**.5 if squared_target else None,
        parameters=rows, buffer_count=len(lb), buffers_identical=all(torch.equal(x, rb[n]) for n, x in lb.items()),
        limitation='Current student-teacher weight gap only; not temporal stability, EMA correctness, or feature quality')
