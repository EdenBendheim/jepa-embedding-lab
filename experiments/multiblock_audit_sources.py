"""Reopen the completed grouping milestones without training or scoring."""

from dataclasses import dataclass
import json
from pathlib import Path

from jepa_lab.embeddings import random_encoder
from jepa_lab.model import ModelConfig
from jepa_lab.objective_run import file_sha
from jepa_lab.selection import validate_pilot
from jepa_lab.variants.multiblock_train import _digest, _model_digest

from experiments.multiblock_milestones import verify_milestones
from experiments.multiblock_trajectory import run_trajectory
from experiments.run_multiblock_budget_20261118 import PROTOCOL_PATH


@dataclass(frozen=True)
class AuditSources:
    """The common ordered fitting panel and its verified frozen encoders."""

    indices: tuple[int, ...]
    models: dict
    identity: dict


def reopen_sources(data, selection, directory, budget, expected_training):
    """Recheck completed chunks against an externally pinned training record."""
    validate_pilot(selection, data.manifest, data.labels)
    indices = tuple(selection['partitions']['train']['indices'])
    if not 20 <= len(indices) <= 1000 or not set(indices).issubset(data.indices):
        raise ValueError('The representation audit requires 20–1000 fixed training rows')
    directory = Path(directory)
    training = json.loads((directory/'training-lock.json').read_text())
    if training != dict(protocol_sha256=file_sha(PROTOCOL_PATH), milestones=expected_training):
        raise ValueError('The saved training lock differs from the published experiment')
    config = ModelConfig(**budget['config'])
    endings = budget['schedule']['checkpoints']
    models = {}
    for seed, order in zip(budget['seeds'], budget['orders'], strict=True):
        folder = directory/f'seed-{seed}'
        receipts = {arm: run_trajectory(data, folder/arm, grouping=arm,
            ends=endings[arm], seed=seed, batch_size=budget['batch_size'], config=config,
            replay=True) for arm in order}
        frozen, observed = verify_milestones(data, folder, receipts)
        if observed | dict(arm_order=order) != expected_training[str(seed)]:
            raise ValueError('The replayed milestones differ from the published training record')
        initial = random_encoder(config, seed)
        if _model_digest(initial) != observed['initial_state_sha256']:
            raise ValueError('The random control differs from the recorded initialization')
        frozen['random'] = initial
        models.update({f'{seed}/{name}': model for name, model in frozen.items()})

    sources = {name: _model_digest(model) for name, model in models.items()}
    identity = dict(manifest_sha256=data.manifest['manifest_sha256'],
        selection_sha256=selection['selection_sha256'], index_order_sha256=_digest(indices),
        rows=len(indices), training_lock_sha256=file_sha(directory/'training-lock.json'),
        model_state_sha256=sources)
    return AuditSources(indices, models, identity)
