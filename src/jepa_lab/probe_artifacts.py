"""Persist fitted probes bound to exact verified train/validation artifacts."""

from dataclasses import asdict
import hashlib
import io
import json
from pathlib import Path
import pickle

import torch

from .cifar10_train import atomic_checkpoint
from .features import FeatureSet, _sha
from .probes import LinearProbe, ProbeConfig, validate_probe_data

from .ridge import RidgeProbe, RidgeConfig

STATE_FIELDS = ("mean", "scale", "constant_columns", "weight", "bias")


def probe_context(train: FeatureSet, validation: FeatureSet) -> dict:
    if not isinstance(train, FeatureSet) or not isinstance(validation, FeatureSet):
        raise ValueError("Probe persistence requires verified feature sets")
    validate_probe_data(train.features, train.labels, fitting=True)
    validate_probe_data(validation.features, validation.labels, dimension=train.features.shape[1])
    if (train.partition != "train" or validation.partition != "validation"
            or set(train.indices).intersection(validation.indices)
            or train.manifest_sha256 != validation.manifest_sha256
            or train.selection_sha256 != validation.selection_sha256 or train.metadata != validation.metadata
            or len(train.indices) != len(train.labels) or len(validation.indices) != len(validation.labels)
            or any(not _sha(value) for value in (train.manifest_sha256, train.selection_sha256,
                                                train.artifact_sha256, validation.artifact_sha256))):
        raise ValueError("Probe sources must be disjoint matching train/validation artifacts with fingerprints")
    context = dict(manifest_sha256=train.manifest_sha256, selection_sha256=train.selection_sha256,
                   train_artifact_sha256=train.artifact_sha256, validation_artifact_sha256=validation.artifact_sha256,
                   feature_dim=train.features.shape[1], feature_metadata=train.metadata,
                   counts=dict(train=len(train.labels), validation=len(validation.labels)))
    json.dumps(context, allow_nan=False)
    return context


def validate_probe_state(state: dict, dimension: int):
    if not isinstance(state, dict) or set(state) != set(STATE_FIELDS):
        raise ValueError("Fitted probe state is incomplete")
    for name, shape in (("mean", (dimension,)), ("scale", (dimension,)),
                        ("constant_columns", (dimension,)), ("weight", (10, dimension)), ("bias", (10,))):
        value = state[name]
        dtype = torch.bool if name == "constant_columns" else torch.float32
        if (not isinstance(value, torch.Tensor) or value.device.type != "cpu" or value.dtype != dtype
                or tuple(value.shape) != shape or value.requires_grad or not torch.isfinite(value).all()):
            raise ValueError(f"Invalid fitted probe tensor: {name}")
    if (state["scale"] <= 0).any() or not torch.equal(
            state["scale"][state["constant_columns"]], torch.ones_like(state["scale"][state["constant_columns"]])):
        raise ValueError("Probe scales must be positive; constant columns use scale one")


def state_fingerprint(state: dict, config: dict, context: dict) -> str:
    digest = hashlib.sha256(json.dumps(dict(config=config, context=context), sort_keys=True,
                                      separators=(",", ":"), allow_nan=False).encode())
    for name in STATE_FIELDS:
        digest.update(name.encode())
        digest.update(state[name].contiguous().numpy().tobytes())
    return digest.hexdigest()


def save_probe(path: str | Path, probe: LinearProbe, train: FeatureSet, validation: FeatureSet) -> dict:
    if not isinstance(probe, LinearProbe) or not isinstance(probe.config, (ProbeConfig, RidgeConfig)):
        raise ValueError("Save a fitted linear probe with a validated configuration")
    context = probe_context(train, validation)
    state = {name: getattr(probe, name).detach().clone() for name in STATE_FIELDS}
    validate_probe_state(state, context["feature_dim"])
    config = asdict(probe.config)
    scores = dict(training=probe.score(train.features, train.labels),
                  validation=probe.score(validation.features, validation.labels))
    saved = dict(format_version=1, kind="CIFAR-10 fitted ridge probe" if isinstance(probe, RidgeProbe) else "CIFAR-10 fitted linear probe",
                 context=context, config=config,
                 state=state, state_sha256=state_fingerprint(state, config, context), scores=scores,
                 runtime=dict(torch=str(torch.__version__), cpu_threads=torch.get_num_threads()), test_evaluated=False)
    atomic_checkpoint(Path(path), saved)
    return load_probe(path, train, validation)[1]


def load_probe(path: str | Path, train: FeatureSet, validation: FeatureSet) -> tuple[LinearProbe, dict]:
    context = probe_context(train, validation)
    path = Path(path)
    if path.stat().st_size > 4 * 1024 * 1024:
        raise ValueError("Local fitted probes are bounded at 4 MiB")
    payload = path.read_bytes()
    try:
        saved = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
    except (ValueError, RuntimeError, EOFError, pickle.UnpicklingError) as exc:
        raise ValueError("Invalid local fitted probe artifact") from exc
    if (not isinstance(saved, dict) or type(saved.get("format_version")) is not int or saved["format_version"] != 1
            or saved.get("kind") not in ("CIFAR-10 fitted linear probe", "CIFAR-10 fitted ridge probe") or saved.get("context") != context
            or saved.get("runtime") != dict(torch=str(torch.__version__), cpu_threads=torch.get_num_threads()) or saved.get("test_evaluated") is not False):
        raise ValueError("Probe artifact identity, sources, runtime, or evaluation boundary differ")
    state, config = saved.get("state"), saved.get("config")
    validate_probe_state(state, context["feature_dim"])
    try:
        settings = (RidgeConfig if saved["kind"] == "CIFAR-10 fitted ridge probe" else ProbeConfig)(**config)
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid saved probe configuration") from exc
    if saved.get("state_sha256") != state_fingerprint(state, asdict(settings), context):
        raise ValueError("Probe state fingerprint differs from the saved artifact")
    probe_class = RidgeProbe if isinstance(settings, RidgeConfig) else LinearProbe
    probe = probe_class(*(state[name].clone() for name in STATE_FIELDS), settings)
    scores = dict(training=probe.score(train.features, train.labels),
                  validation=probe.score(validation.features, validation.labels))
    if saved.get("scores") != scores:
        raise ValueError("Saved probe scores do not reproduce on the bound feature artifacts")
    return probe, dict(context=context, config=asdict(settings), scores=scores,
                       state_sha256=saved["state_sha256"], artifact_sha256=hashlib.sha256(payload).hexdigest(),
                       test_evaluated=False)
