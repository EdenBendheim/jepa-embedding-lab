"""Frozen full-image RGB features and provenance-bound CIFAR-10 extraction."""

import argparse
from dataclasses import asdict
import hashlib
import io
import json
from pathlib import Path

import torch

from .cifar10 import CIFAR10Binary, make_manifest, patchify_rgb
from .cifar10_train import atomic_checkpoint
from .model import JEPA, ModelConfig
from .selection import validate_pilot


def rgb_config(config: ModelConfig):
    if (config.rows, config.cols, config.patch_dim) != (8, 8, 48):
        raise ValueError("Full-image CIFAR-10 embeddings require an 8x8 RGB grid (patch_dim=48)")


@torch.inference_mode()
def image_embeddings(model: JEPA, images: torch.Tensor, *, encoder: str = "context") -> torch.Tensor:
    """Mean-pool all 64 encoded patches; do not use masks or the predictor."""
    if encoder not in ("context", "target"):
        raise ValueError("Encoder must be context or target")
    rgb_config(model.config)
    patches = patchify_rgb(images)
    if patches.shape[1:] != (64, 48):
        raise ValueError("Protocol requires 32x32 RGB images")
    was_training = model.training
    try:
        model.eval()
        module = model.context_encoder if encoder == "context" else model.target_encoder
        positions = torch.arange(64, device=patches.device).expand(images.shape[0], -1)
        features = module(patches, positions).mean(dim=1)
        if not torch.isfinite(features).all():
            raise ValueError("Encoder produced non-finite embeddings")
        return features
    finally:
        model.train(was_training)


@torch.inference_mode()
def pixel_features(images: torch.Tensor) -> torch.Tensor:
    # Reuse the RGB adapter's dtype/layout checks and fixed normalization contract.
    patchify_rgb(images)
    if images.shape[1:] != (3, 32, 32):
        raise ValueError("Raw-pixel protocol requires 32x32 RGB images")
    return images.float().div(127.5).sub(1).flatten(1)


def random_encoder(config: ModelConfig, seed: int) -> JEPA:
    rgb_config(config)
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("Random-encoder seed must be an integer in [0, 2**63)")
    # Baseline construction must not change the caller's training RNG sequence.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return JEPA(config).eval().requires_grad_(False)


def load_encoder(checkpoint: str | Path, manifest_sha256: str) -> tuple[JEPA, dict]:
    path = Path(checkpoint)
    payload = path.read_bytes()
    saved = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
    if (not isinstance(saved, dict) or saved.get("format_version") != 1 or saved.get("dataset") != "CIFAR-10"
            or saved.get("manifest_sha256") != manifest_sha256
            or not isinstance(saved.get("runtime"), dict)
            or saved.get("runtime", {}).get("torch") != str(torch.__version__)
            or type(saved.get("step")) is not int or saved["step"] < 1):
        raise ValueError("Encoder checkpoint dataset, manifest, runtime, or completed steps are incompatible")
    try:
        config = ModelConfig(**saved["config"])
        rgb_config(config)
        with torch.random.fork_rng(devices=[]):
            model = JEPA(config).eval().requires_grad_(False)
        model.load_state_dict(saved["model"])
        settings = saved["settings"]
        if not isinstance(settings, dict):
            raise ValueError("Training settings must be a recorded dictionary")
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        raise ValueError("Checkpoint encoder configuration or weights are invalid") from exc
    metadata = dict(checkpoint_sha256=hashlib.sha256(payload).hexdigest(),
                    completed_steps=saved["step"], config=asdict(config), training_settings=settings)
    return model, metadata


def extract_cifar10(root: str | Path, manifest_path: str | Path, *, partition: str = "train",
                    representation: str = "checkpoint", checkpoint: str | Path | None = None,
                    encoder: str = "context", seed: int = 29, limit: int | None = None, batch_size: int = 16,
                    selection_path: str | Path | None = None,
                    config: ModelConfig | None = None, output: str | Path | None = None) -> dict:
    if partition not in ("train", "validation", "test") or representation not in ("checkpoint", "random", "pixels"):
        raise ValueError("Choose a fixed partition and checkpoint/random/pixels representation")
    if selection_path is not None and (limit is not None or partition == "test"):
        raise ValueError("Balanced pilot extraction uses all selected train/validation indices; no limit or test")
    limit = 64 if limit is None else limit
    if type(limit) is not int or not 1 <= limit <= 1000 or type(batch_size) is not int or not 1 <= batch_size <= 32:
        raise ValueError("Bounded extraction allows 1-1000 images and batch size 1-32")
    if encoder not in ("context", "target"):
        raise ValueError("Encoder must be context or target")
    if (representation == "checkpoint") != (checkpoint is not None):
        raise ValueError("Only checkpoint representations require a checkpoint path")
    if representation == "checkpoint" and config is not None:
        raise ValueError("Checkpoint architecture comes from its recorded configuration")
    manifest = json.loads(Path(manifest_path).read_text())
    train, test = CIFAR10Binary(root), CIFAR10Binary(root, train=False)
    if manifest != make_manifest(train, test, seed=manifest["seed"]):
        raise ValueError("Manifest does not match current data or the fixed split protocol")
    selection = validate_pilot(json.loads(Path(selection_path).read_text()), manifest, train.labels) if selection_path else None
    metadata = dict(representation=representation, pooling="mean of all 64 patch tokens" if representation != "pixels"
                    else "flatten NCHW RGB values", normalization="x / 127.5 - 1")
    model = None
    if representation == "checkpoint":
        model, details = load_encoder(checkpoint, manifest["manifest_sha256"])
        metadata |= details | dict(encoder=encoder)
    elif representation == "random":
        config = config if config is not None else ModelConfig(patch_dim=48)
        model = random_encoder(config, seed)
        metadata |= dict(encoder=encoder, seed=seed, config=asdict(config))
    selected = manifest["partitions"][partition]
    indices = selection["partitions"][partition]["indices"] if selection else selected["indices"][:limit]
    dataset = train if selected["official_split"] == "train" else test
    features, labels = [], []
    for start in range(0, len(indices), batch_size):
        batch = [dataset[index] for index in indices[start:start+batch_size]]
        images = torch.stack([image for image, _ in batch])
        labels.extend(label for _, label in batch)
        values = pixel_features(images) if model is None else image_embeddings(model, images, encoder=encoder)
        features.append(values.cpu())
    values = torch.cat(features)
    fingerprint = hashlib.sha256(values.contiguous().numpy().tobytes()).hexdigest()
    identity = dict(format_version=1, dataset="CIFAR-10", manifest_sha256=manifest["manifest_sha256"],
                    partition=partition, official_split=selected["official_split"], indices=indices,
                    feature_metadata=metadata, features_sha256=fingerprint,
                    selection_sha256=selection["selection_sha256"] if selection else None,
                    extraction_settings=dict(limit=len(indices), batch_size=batch_size),
                    runtime=dict(torch=str(torch.__version__), cpu_threads=torch.get_num_threads()))
    if output:
        atomic_checkpoint(Path(output), identity | dict(features=values, labels=torch.tensor(labels, dtype=torch.long)))
    return identity | dict(count=len(indices), feature_dim=values.shape[1], finite=bool(torch.isfinite(values).all()),
                           feature_std_mean=float(values.std(dim=0, unbiased=False).mean()),
                           limitation="Bounded feature extraction only; no fitted probe or classification-quality result")


def main():
    parser = argparse.ArgumentParser(description="Extract frozen full-image embeddings or a matched-data baseline")
    parser.add_argument("root", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--partition", choices=("train", "validation", "test"), default="train")
    parser.add_argument("--representation", choices=("checkpoint", "random", "pixels"), default="checkpoint")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--encoder", choices=("context", "target"), default="context")
    parser.add_argument("--seed", type=int, default=29, help="Random-encoder seed")
    parser.add_argument("--limit", type=int, help="First N fixed indices (default 64); cannot combine with --selection")
    parser.add_argument("--selection", type=Path, help="Shared balanced pilot manifest; extracts its full selected partition")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True, help="Local feature artifact; use ignored runs/ and .pt")
    args = parser.parse_args()
    torch.set_num_threads(1)
    try:
        report = extract_cifar10(args.root, args.manifest, partition=args.partition, representation=args.representation,
                                checkpoint=args.checkpoint, encoder=args.encoder, seed=args.seed, limit=args.limit,
                                batch_size=args.batch_size, selection_path=args.selection, output=args.output)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
