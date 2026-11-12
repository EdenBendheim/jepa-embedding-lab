"""Load a provenance-bound frozen CIFAR-10 encoder from a multi-block run."""

import torch

from ..model import JEPA, ModelConfig
from .cifar_batches import CifarTrainingData
from .multiblock_train import _model_digest, _public_sampler, _read_checkpoint, _recipe, _validate_history


def load_cifar_encoder(checkpoint, training_data, *, expected_sha256=None):
    if not isinstance(training_data, CifarTrainingData):
        raise ValueError('Pass verified CIFAR-10 training data')
    checksum, saved = _read_checkpoint(checkpoint, expected_sha256=expected_sha256)
    try:
        settings = saved['settings']
        config, identity = _recipe(config=ModelConfig(**saved['config']), training_data=training_data,
            **{key: settings[key] for key in ('batch_size', 'seed', 'learning_rate', 'momentum')},
            **settings['mask'])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError('Invalid multi-block checkpoint configuration') from exc
    history = _validate_history(saved, identity, config)
    _public_sampler(training_data, settings, history, saved)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(settings['seed'])
        model = JEPA(config).eval().requires_grad_(False)
    if saved['initial_state_sha256'] != _model_digest(model):
        raise ValueError('Checkpoint initialization differs from its seed and architecture')
    try:
        model.load_state_dict(saved['model'])
    except (KeyError, TypeError, RuntimeError) as exc:
        raise ValueError('Invalid multi-block encoder state') from exc
    if any(not torch.isfinite(value).all() for value in model.state_dict().values()):
        raise ValueError('Multi-block encoder weights must be finite')
    metadata = identity | dict(checkpoint_sha256=checksum, completed_steps=saved['step'],
                               initial_state_sha256=saved['initial_state_sha256'])
    return model, metadata, history
