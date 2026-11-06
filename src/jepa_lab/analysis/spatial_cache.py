"""Local-only immutable feature caches for matched spatial summaries."""

import hashlib
import io
import os
from pathlib import Path
import tempfile

import torch

from ..objective_run import file_sha
from ..probes import _values
from .color_cache import code_fingerprints
from .spatial_sources import extract_fitting, prepare_sources


def tensor_sha(value):
    header = f'{value.dtype}:{tuple(value.shape)}:'.encode()
    return hashlib.sha256(header + value.detach().contiguous().numpy().tobytes()).hexdigest()


def validate_features(features, inputs):
    expected = {'pixels': 3072} | {f'{key}/{mode}': width for key in inputs.identity['sources']
                                   for mode, width in inputs.identity['feature_dims'].items()}
    if not isinstance(features, dict) or features.keys() != expected.keys():
        raise ValueError('Spatial feature sources differ from the declared paired controls')
    for key, width in expected.items():
        _values(features[key], dimension=width)
        if len(features[key]) != len(inputs.indices):
            raise ValueError('Spatial feature count differs from fitting indices')
    if not torch.equal(features['pixels'], inputs.references['pixels'].features):
        raise ValueError('Spatial pixel control differs from the verified original')
    for key in inputs.identity['sources']:
        if not torch.allclose(features[f'{key}/mean'], inputs.references[key].features, atol=1e-6, rtol=1e-5):
            raise ValueError('Spatial mean control differs from the verified original')


def _load(path, inputs, identity):
    if path.stat().st_size > 64 * 1024 * 1024:
        raise ValueError('Spatial cache exceeds 64 MiB')
    state = torch.load(io.BytesIO(path.read_bytes()), map_location='cpu', weights_only=True)
    if (not isinstance(state, dict) or set(state) != {'version', 'identity', 'features', 'features_sha256', 'labels'}
            or type(state['version']) is not int or state['version'] != 1 or state['identity'] != identity
            or not isinstance(state['labels'], torch.Tensor) or state['labels'].dtype != torch.long
            or not torch.equal(state['labels'], inputs.labels)):
        raise ValueError('Spatial cache identity or labels differ; preserve it and choose a new path')
    validate_features(state['features'], inputs)
    if state['features_sha256'] != {key: tensor_sha(value) for key, value in state['features'].items()}:
        raise ValueError('Spatial feature content differs from its recorded digest')
    return state['features']


def cached_features(run, output, *, reuse_only=False, expected_identity=None):
    if type(reuse_only) is not bool:
        raise ValueError('Reuse-only must be boolean')
    output = Path(output)
    package = Path(__file__).resolve().parent.parent
    if (output.suffix != '.pt' or any(output.resolve().is_relative_to(folder.resolve())
                                    for folder in (run.root, run.directory, package))):
        raise ValueError('Spatial caches must be outside raw data, completed experiments and code')
    inputs = prepare_sources(run)
    identity = dict(version=1, inputs=inputs.identity, code_sha256=code_fingerprints())
    if expected_identity is not None and expected_identity != identity:
        raise ValueError('Declared spatial protocol differs from the current sources or code')
    if output.exists():
        return inputs, _load(output, inputs, identity), True, file_sha(output)
    if reuse_only:
        raise ValueError('Reuse-only spatial feature cache is missing')
    features = extract_fitting(run.root, inputs)
    validate_features(features, inputs)
    state = dict(version=1, identity=identity, features=features,
                 features_sha256={key: tensor_sha(value) for key, value in features.items()}, labels=inputs.labels)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=output.parent, delete=False) as handle:
            temporary = Path(handle.name)
            torch.save(state, handle)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, output)
        except FileExistsError:
            return inputs, _load(output, inputs, identity), True, file_sha(output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return inputs, features, False, file_sha(output)
