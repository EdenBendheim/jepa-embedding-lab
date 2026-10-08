"""Verify local feature artifacts before fitting or comparing a frozen probe."""

import argparse
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path
import pickle
import re

import torch

from .cifar10 import CIFAR10Binary, make_manifest
from .embeddings import rgb_config
from .model import ModelConfig
from .selection import validate_pilot


@dataclass(frozen=True)
class FeatureSet:
    features: torch.Tensor
    labels: torch.Tensor
    indices: tuple[int, ...]
    partition: str
    manifest_sha256: str
    selection_sha256: str
    metadata: dict
    artifact_sha256: str

    def summary(self) -> dict:
        return dict(partition=self.partition, official_split="train", count=len(self.indices),
                    feature_dim=self.features.shape[1], manifest_sha256=self.manifest_sha256,
                    selection_sha256=self.selection_sha256, artifact_sha256=self.artifact_sha256,
                    feature_metadata=deepcopy(self.metadata))


def _sha(value):
    return isinstance(value, str) and re.fullmatch(r"[a-f0-9]{64}", value) is not None


def load_features(path: str | Path, *, manifest: dict, selection: dict, training_labels: list[int],
                  partition: str) -> FeatureSet:
    if partition not in ("train", "validation"):
        raise ValueError("Pilot probes accept training and validation artifacts only; test is reserved")
    validate_pilot(selection, manifest, training_labels)
    path = Path(path)
    if path.stat().st_size > 64*1024*1024:
        raise ValueError("Bounded pilot artifacts must be at most 64 MiB")
    payload = path.read_bytes()
    try:
        saved = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
    except (ValueError, RuntimeError, EOFError, pickle.UnpicklingError) as exc:
        raise ValueError("Invalid local feature artifact") from exc
    if (not isinstance(saved, dict) or type(saved.get("format_version")) is not int or saved.get("format_version") != 1 or saved.get("dataset") != "CIFAR-10"
            or saved.get("manifest_sha256") != manifest["manifest_sha256"]
            or saved.get("selection_sha256") != selection["selection_sha256"]
            or saved.get("partition") != partition or saved.get("official_split") != "train"
            or saved.get("indices") != selection["partitions"][partition]["indices"]):
        raise ValueError("Artifact identity, exact indices, or namespace differs from the balanced pilot")
    indices = saved["indices"]
    values, labels = saved.get("features"), saved.get("labels")
    if (not isinstance(values, torch.Tensor) or values.dtype != torch.float32 or values.ndim != 2
            or values.shape[0] != len(indices) or not 1 <= values.shape[1] <= 4096
            or not torch.isfinite(values).all() or values.requires_grad):
        raise ValueError("Features must be finite frozen float32 rows matching the selected examples")
    expected = torch.tensor([training_labels[index] for index in indices], dtype=torch.long)
    if not isinstance(labels, torch.Tensor) or labels.dtype != torch.long or not torch.equal(labels, expected):
        raise ValueError("Artifact labels differ from the selected official training labels")
    fingerprint = hashlib.sha256(values.contiguous().numpy().tobytes()).hexdigest()
    if saved.get("features_sha256") != fingerprint:
        raise ValueError("Feature content fingerprint differs from the recorded artifact")
    settings, runtime = saved.get("extraction_settings"), saved.get("runtime")
    if (not isinstance(settings, dict) or settings.get("limit") != len(indices)
            or type(settings.get("batch_size")) is not int or not 1 <= settings["batch_size"] <= 32
            or not isinstance(runtime, dict) or runtime.get("torch") != str(torch.__version__)
            or type(runtime.get("cpu_threads")) is not int or runtime["cpu_threads"] < 1):
        raise ValueError("Artifact extraction settings or PyTorch runtime are incompatible")
    metadata = saved.get("feature_metadata")
    if not isinstance(metadata, dict) or metadata.get("normalization") != "x / 127.5 - 1":
        raise ValueError("Feature metadata must record the fixed input normalization")
    try:
        json.dumps(metadata, allow_nan=False)
    except (ValueError, TypeError) as exc:
        raise ValueError("Feature metadata must be finite JSON-compatible provenance") from exc
    representation = metadata.get("representation")
    if representation == "pixels":
        if (metadata.get("pooling") != "flatten NCHW RGB values" or values.shape[1] != 3072
                or (values.abs() > 1).any()):
            raise ValueError("Pixel baseline must use 3,072 normalized NCHW values")
    elif representation in ("checkpoint", "random"):
        try:
            config = ModelConfig(**metadata["config"])
            rgb_config(config)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Encoder architecture metadata is invalid") from exc
        if (values.shape[1] != config.embedding_dim or metadata.get("encoder") not in ("context", "target")
                or metadata.get("pooling") != "mean of all 64 patch tokens"):
            raise ValueError("Encoder dimension, choice, or pooling differs from its feature contract")
        if representation == "checkpoint":
            if (not _sha(metadata.get("checkpoint_sha256")) or type(metadata.get("completed_steps")) is not int
                    or metadata["completed_steps"] < 1 or not isinstance(metadata.get("training_settings"), dict)):
                raise ValueError("Checkpoint feature metadata lacks valid weight/training provenance")
        elif type(metadata.get("seed")) is not int or not 0 <= metadata["seed"] < 2**63:
            raise ValueError("Random-encoder features require a valid initialization seed")
    else:
        raise ValueError("Unsupported feature representation")
    return FeatureSet(values.detach().clone(), labels.clone(), tuple(indices), partition,
                      manifest["manifest_sha256"], selection["selection_sha256"], deepcopy(metadata),
                      hashlib.sha256(payload).hexdigest())


def main():
    parser = argparse.ArgumentParser(description="Verify a local balanced-pilot feature artifact without fitting")
    parser.add_argument("root", type=Path)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--partition", choices=("train", "validation"), required=True)
    args = parser.parse_args()
    try:
        train, test = CIFAR10Binary(args.root), CIFAR10Binary(args.root, train=False)
        manifest = json.loads(args.manifest.read_text())
        if manifest != make_manifest(train, test, seed=manifest["seed"]):
            raise ValueError("Dataset manifest differs from current files or the locked protocol")
        result = load_features(args.artifact, manifest=manifest, selection=json.loads(args.selection.read_text()),
                               training_labels=train.labels, partition=args.partition)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    print(json.dumps(result.summary(), indent=2))


if __name__ == "__main__":
    main()
