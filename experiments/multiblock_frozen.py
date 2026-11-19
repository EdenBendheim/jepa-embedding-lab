"""Fit training-only ridge heads and score them after recording fresh exposure."""

from hashlib import sha256
import io
import json
from pathlib import Path

import torch

from jepa_lab.analysis.color_statistics import rgb_statistics
from jepa_lab.cifar10_train import atomic_checkpoint
from jepa_lab.compare import write_report
from jepa_lab.diagnostics import paired_diagnostics
from jepa_lab.embeddings import image_embeddings, pixel_features
from jepa_lab.exposure import extend_exposure_bundle, exposure_summary, load_exposure_bundle, make_ledger_confirmation
from jepa_lab.objective_run import file_sha
from jepa_lab.probe_artifacts import STATE_FIELDS, validate_probe_state
from jepa_lab.ridge import RidgeConfig, RidgeProbe, fit_ridge
from jepa_lab.variants.multiblock_train import _digest


def _verify_or_save(path, expected):
    if path.exists():
        if json.loads(path.read_text()) != expected:
            raise ValueError(f'Saved evidence changed: {path.name}')
    else:
        write_report(path, expected)


def _features(read, indices, models, labels, *, batch_size=16):
    parts = {name: [] for name in (*models, 'rgb_statistics', 'pixels')}
    for start in range(0, len(indices), batch_size):
        batch = indices[start:start+batch_size]
        images, observed_labels = read(batch)
        if observed_labels.tolist() != [labels[index] for index in batch]:
            raise ValueError('Classification sample labels differ from the fixed manifest')
        for name, model in models.items():
            parts[name].append(image_embeddings(model, images))
        parts['rgb_statistics'].append(rgb_statistics(images))
        parts['pixels'].append(pixel_features(images))
    return {name: torch.cat(chunks) for name, chunks in parts.items()}


def prepare_heads(data, directory, selection, models, *, identity, alpha, replay):
    """Lock the exact fitted heads before reserving validation images."""
    directory = Path(directory)
    checkpoint = directory/'heads.pt'
    lock_path = directory/'heads-lock.json'
    indices = selection['partitions']['train']['indices']
    context = dict(identity=identity, sources=list(models), alpha=alpha,
                   manifest_sha256=data.manifest['manifest_sha256'],
                   fitting_indices_sha256=_digest(indices))
    config = RidgeConfig(alpha)
    if not lock_path.exists():
        if replay or (directory/'observation-receipt.json').exists():
            raise ValueError('Replay or prior exposure requires the original fitted-head lock')
        features = _features(data.read, indices, models, data.labels)
        labels = torch.tensor([data.labels[index] for index in indices])
        states = {}
        for name, values in features.items():
            head = fit_ridge(values, labels, config)
            states[name] = {field: getattr(head, field) for field in STATE_FIELDS}
        atomic_checkpoint(checkpoint, states)
        write_report(lock_path, context | dict(checkpoint_sha256=file_sha(checkpoint)))
    lock = json.loads(lock_path.read_text())
    if checkpoint.stat().st_size > 4*1024*1024:
        raise ValueError('Frozen classifier bundle exceeds 4 MiB')
    payload = checkpoint.read_bytes()
    if lock != context | dict(checkpoint_sha256=sha256(payload).hexdigest()):
        raise ValueError('Frozen classifier lock or fitted weights changed')
    states = torch.load(io.BytesIO(payload), map_location='cpu', weights_only=True)
    dimensions = {name: model.config.embedding_dim for name, model in models.items()} | {
        'rgb_statistics': 6, 'pixels': 3072}
    if states.keys() != dimensions.keys():
        raise ValueError('Frozen classifier sources differ from the declaration')
    for name, dimension in dimensions.items():
        validate_probe_state(states[name], dimension)
    return {name: RidgeProbe(**state, config=config) for name, state in states.items()}


def register_validation(data, directory, selection, exposure, *, per_class, seed):
    """Record the exact reservation before any selected validation image is indexed."""
    directory = Path(directory)
    if not (directory/'heads-lock.json').is_file():
        raise ValueError('Lock fitted classifier states before reserving fresh rows')
    previous = load_exposure_bundle(exposure, data.manifest, data.labels)
    ledger = previous['ledger']
    chosen = make_ledger_confirmation(data.manifest, data.labels, selection, ledger,
                                      validation_per_class=per_class, seed=seed)
    reservation = directory/'confirmation-selection.json'
    _verify_or_save(reservation, chosen)
    updated = extend_exposure_bundle(exposure, data.manifest, data.labels,
        [(reservation, file_sha(reservation))], expected_ledger_sha256=ledger['ledger_sha256'])
    _verify_or_save(directory/'observation-receipt.json', dict(
        selection_sha256=chosen['selection_sha256'], heads_lock_sha256=file_sha(directory/'heads-lock.json'),
        exposure_bundle_sha256=updated['bundle_sha256']))
    _verify_or_save(directory/'observed-exposure.json', updated)
    return chosen, exposure_summary(updated['ledger'], data.manifest, data.labels)


def score_frozen(data, directory, selected, models, heads):
    """Score fixed heads on the registered validation reservation using shared batches."""
    directory = Path(directory)
    receipt = json.loads((directory/'observation-receipt.json').read_text())
    observed = load_exposure_bundle(directory/'observed-exposure.json', data.manifest, data.labels)
    reservation = directory/'confirmation-selection.json'
    registered = any(row['path'] == str(reservation.resolve()) and
                     row['selection_sha256'] == selected['selection_sha256']
                     for row in observed['source_files'])
    if (json.loads(reservation.read_text()) != selected
            or not registered
            or receipt['selection_sha256'] != selected['selection_sha256']
            or receipt['heads_lock_sha256'] != file_sha(directory/'heads-lock.json')
            or receipt['exposure_bundle_sha256'] != observed['bundle_sha256']):
        raise ValueError('Fresh scoring requires matching frozen-head and exposure evidence')
    indices = selected['partitions']['validation']['indices']

    def read(batch):
        rows = [data._dataset[index] for index in batch]
        return torch.stack([image for image, _ in rows]), torch.tensor([label for _, label in rows])

    features = _features(read, indices, models, data.labels)
    labels = torch.tensor([data.labels[index] for index in indices])
    predictions = {name: head.logits(features[name]).argmax(1) for name, head in heads.items()}
    scores = {name: head.score(features[name], labels) for name, head in heads.items()}
    return scores, predictions, labels


def paired_scores(labels, predictions, *, seed, extended):
    """Count paired recoveries and regressions against grouping and baseline controls."""
    pairs = [('separate', 'joint_matched'), ('random', 'separate'), ('random', 'joint_matched')]
    if extended:
        pairs += [('separate', 'joint_extended'), ('joint_matched', 'joint_extended'), ('random', 'joint_extended')]
    trained = ['separate', 'joint_matched'] + (['joint_extended'] if extended else [])
    pairs.extend((control, name) for control in ('rgb_statistics', 'pixels') for name in trained)
    result = {}
    for before, after in pairs:
        baseline = predictions[before] if before in ('rgb_statistics', 'pixels') else predictions[f'{seed}/{before}']
        report = paired_diagnostics(labels, baseline, predictions[f'{seed}/{after}'])
        report['limitation'] = 'Descriptive paired errors on one reserved validation selection.'
        result[f'{after}-minus-{before}'] = report
    return result
