"""Explicit, deterministic inputs for an optional frozen transfer baseline."""

from copy import deepcopy
import hashlib
import io
from pathlib import Path
import pickle

import torch

WEIGHTS_SHA256 = "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"
WEIGHTS_URL = "https://download.pytorch.org/models/resnet18-f37072fd.pth"
NORMALIZATION = "ImageNet1K tensor recipe"


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


def pretrained_metadata():
    import torchvision
    if torchvision.__version__.split("+")[0] != "0.23.0":
        raise ValueError("The declared baseline requires torchvision 0.23.0")
    return dict(representation="pretrained", architecture="torchvision ResNet-18",
                weights="IMAGENET1K_V1", weights_sha256=WEIGHTS_SHA256,
                weights_url=WEIGHTS_URL, pretraining="supervised ImageNet-1K",
                feature_dim=512, pooling="frozen global-average pool before classification head",
                normalization=NORMALIZATION, transform=transform_metadata(),
                torchvision=str(torchvision.__version__))


def load_pretrained(path: str | Path):
    """Load exactly the named local weights. Never download or unpickle custom objects."""
    metadata = pretrained_metadata()
    path = Path(path)
    if path.stat().st_size > 64*1024*1024:
        raise ValueError("Local pretrained weights exceed the 64 MiB bound")
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != WEIGHTS_SHA256:
        raise ValueError("Local weights differ from the declared ImageNet ResNet-18 bytes")
    try:
        from torchvision.models import resnet18
        state = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
        if (not isinstance(state, dict) or not state
                or any(not isinstance(t, torch.Tensor) or not torch.isfinite(t).all()
                       for t in state.values())):
            raise ValueError("Pretrained state must contain finite tensors")
        with torch.random.fork_rng(devices=[]):
            model = resnet18(weights=None)
        model.load_state_dict(state, strict=True)
        model.fc = torch.nn.Identity()
        model.eval().requires_grad_(False)
    except (ValueError, RuntimeError, TypeError, EOFError, pickle.UnpicklingError) as exc:
        raise ValueError("Local pretrained architecture or state is invalid") from exc
    return model, metadata


@torch.inference_mode()
def pretrained_embeddings(model: torch.nn.Module, images: torch.Tensor):
    """Extract frozen features while restoring every submodule's original mode."""
    inputs = pretrained_input(images)
    modes = [(module, module.training) for module in model.modules()]
    try:
        model.eval()
        with torch.random.fork_rng(devices=[]):
            values = model(inputs)
        if (values.dtype != torch.float32 or values.device.type != "cpu"
                or values.shape != (len(images), 512) or not torch.isfinite(values).all()):
            raise ValueError("Pretrained encoder must return finite CPU float32 512-dimensional rows")
        return values.detach().clone()
    finally:
        for module, mode in modes:
            module.training = mode
