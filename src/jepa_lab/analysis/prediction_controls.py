"""Check whether a fixed-mask predictor benefits from its own image context."""

import torch
from torch.nn.functional import smooth_l1_loss


@torch.inference_mode()
def prediction_controls(predictions: torch.Tensor, targets: torch.Tensor) -> dict:
    """Compare aligned, reassigned, and leave-one-out target template errors.

    Inputs are [image, target slot, coordinate] for a single shared mask. With
    independent rows, reassigning a prediction changes its image context while
    retaining target slots and mask geometry. The template uses the other
    supplied targets at each slot. It is a descriptive control on this sample.
    """
    for tensor in (predictions, targets):
        if (not isinstance(tensor, torch.Tensor) or tensor.ndim != 3
                or tensor.dtype != torch.float32 or tensor.device.type != 'cpu'
                or not 2 <= tensor.shape[0] <= 1000 or not 1 <= tensor.shape[1] <= 64
                or not 1 <= tensor.shape[2] <= 4096 or tensor.numel() > 4_000_000
                or not torch.isfinite(tensor).all()):
            raise ValueError('Use finite bounded CPU float32 [2-1000 images, targets, coordinates]')
    if predictions.shape != targets.shape:
        raise ValueError('Prediction and target shapes must match')
    pred, target = predictions.double(), targets.double()
    count = len(target)
    shifts = sorted({1, count // 2, count - 1})
    matched = float(smooth_l1_loss(pred, target))
    reassigned = [float(smooth_l1_loss(pred.roll(shift, dims=0), target)) for shift in shifts]
    template = (target.sum(dim=0) - target) / (count - 1)
    reassigned_mean = sum(reassigned) / len(reassigned)
    return {
        'count': count,
        'target_slots': target.shape[1],
        'matched_smooth_l1': matched,
        'reassigned_smooth_l1_mean': reassigned_mean,
        'reassignment_shifts': shifts,
        'reassigned_smooth_l1_by_shift': reassigned,
        'reassigned_minus_matched': reassigned_mean - matched,
        'leave_one_out_template_smooth_l1': float(smooth_l1_loss(template, target)),
    }
