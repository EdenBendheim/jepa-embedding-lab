"""Explicit, deterministic inputs for an optional frozen transfer baseline."""

from copy import deepcopy

import torch


TRANSFORM = {
    "name": "torchvision_resnet18_imagenet1k_v1_tensor_256_224",
    "input": "uint8 NCHW RGB 32x32",
    "resize_short_edge": 256,
    "crop_size": 224,
    "interpolation": "bilinear",
    "antialias": True,
    "scale": "x / 255",
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
}


def transform_metadata():
    """Return an independent description of the fixed tensor preprocessing."""
    return deepcopy(TRANSFORM)


@torch.inference_mode()
def pretrained_input(images: torch.Tensor) -> torch.Tensor:
    """Apply the named ImageNet weight recipe without augmentations or fitting."""
    if (not isinstance(images, torch.Tensor) or images.dtype != torch.uint8
            or images.device.type != "cpu" or images.ndim != 4
            or images.shape[1:] != (3, 32, 32) or not 1 <= images.shape[0] <= 16):
        raise ValueError("Pretrained inputs require 1-16 CPU uint8 NCHW RGB 32x32 images")
    try:
        from torchvision.models import ResNet18_Weights
    except ImportError as exc:
        raise RuntimeError("Install the optional pretrained extra to use ResNet-18") from exc
    result = ResNet18_Weights.IMAGENET1K_V1.transforms()(images)
    if result.shape != (len(images), 3, 224, 224) or not torch.isfinite(result).all():
        raise ValueError("Fixed pretrained transform produced invalid inputs")
    return result
