"""Row-identity-stable quadrant permutations for a layout negative control."""

import hashlib

import torch


def _validate_indices(indices, count):
    if (not isinstance(indices, (list, tuple)) or len(indices) != count
            or any(type(i) is not int or not 0 <= i < 50000 for i in indices)
            or len(set(indices)) != count):
        raise ValueError('Use one distinct official training index per feature row')


@torch.inference_mode()
def shuffle_quadrants(features, indices, *, seed=29):
    """Permute four whole cells per image, independent of batch order or labels."""
    if (not isinstance(features, torch.Tensor) or features.dtype != torch.float32
            or features.device.type != 'cpu' or features.ndim != 2
            or not 1 <= len(features) <= 1000 or not 4 <= features.shape[1] <= 2048
            or features.shape[1] % 4 or not torch.isfinite(features).all()
            or type(seed) is not int or not 0 <= seed < 2**63):
        raise ValueError('Use finite CPU float32 four-cell vectors and an integer seed')
    _validate_indices(indices, len(features))
    cells = features.reshape(len(features), 4, -1)
    output = []
    for row, index in enumerate(indices):
        local_seed = int.from_bytes(hashlib.sha256(f'quadrants-v1:{seed}:{index}'.encode()).digest()[:8], 'big') % (2**63)
        order = torch.randperm(4, generator=torch.Generator().manual_seed(local_seed))
        output.append(cells[row, order].reshape(-1))
    return torch.stack(output)
