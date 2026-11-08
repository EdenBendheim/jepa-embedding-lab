"""Exploratory stratified cross-fitting on original encoder-training rows."""

import statistics

import torch

from ..diagnostics import paired_diagnostics
from ..feature_geometry import feature_geometry
from ..probes import _labels, validate_probe_data
from ..ridge import RidgeConfig, fit_ridge
from .color_audit import digest
from .spatial_cache import validate_features
from .spatial_embeddings import MODES


def stratified_folds(labels, *, folds=5, seed=29):
    if (not isinstance(labels, torch.Tensor) or labels.ndim != 1 or not 20 <= len(labels) <= 1000
            or type(folds) is not int or not 2 <= folds <= 10
            or type(seed) is not int or not 0 <= seed < 2**63):
        raise ValueError('Use 20-1000 class labels, 2-10 folds and an integer seed')
    _labels(labels, len(labels), all_classes=True)
    if any(int((labels == label).sum()) < folds for label in range(10)):
        raise ValueError('Every class needs at least one row in every scoring fold')
    generator = torch.Generator().manual_seed(seed)
    assignments = torch.empty(len(labels), dtype=torch.long)
    for label in range(10):
        rows = (labels == label).nonzero().flatten()
        rows = rows[torch.randperm(len(rows), generator=generator)]
        assignments[rows] = torch.arange(len(rows)) % folds
    return assignments


@torch.inference_mode()
def crossfit_ridge(features, labels, assignments, *, config=RidgeConfig(.1), fold_provider=None):
    validate_probe_data(features, labels, fitting=True)
    if (not isinstance(assignments, torch.Tensor) or assignments.device.type != 'cpu'
            or assignments.dtype != torch.long or assignments.shape != labels.shape):
        raise ValueError('Use one CPU integer fold ID per feature row')
    folds = sorted(assignments.unique().tolist())
    if (folds != list(range(len(folds))) or not 2 <= len(folds) <= 10
            or any(set(labels[assignments != fold].tolist()) != set(range(10)) for fold in folds)):
        raise ValueError('Use contiguous folds retaining all ten classes in every fitting subset')
    predictions = torch.empty_like(labels)
    rows = []
    for fold in folds:
        fitting, scoring = assignments != fold, assignments == fold
        probe = (fit_ridge(features[fitting], labels[fitting], config) if fold_provider is None else
                 fold_provider(fold, features[fitting], labels[fitting], config))
        predictions[scoring] = probe.logits(features[scoring]).argmax(1)
        rows.append(dict(fold=fold, fitting_count=int(fitting.sum()), scoring_count=int(scoring.sum()),
            scoring_class_counts=torch.bincount(labels[scoring], minlength=10).tolist(),
            correct=int((predictions[scoring] == labels[scoring]).sum())))
    confusion = torch.bincount(labels * 10 + predictions, minlength=100).reshape(10, 10)
    correct = int((predictions == labels).sum())
    return predictions, dict(count=len(labels), correct=correct, accuracy=correct/len(labels),
        alpha=config.alpha, folds=rows, confusion_matrix=confusion.tolist(),
        classifier_weight_parameters=10 * features.shape[1],
        limitation='Cross-fitted classifier scores on encoder-training images; not held-out representation quality')


def evaluate(inputs, features):
    validate_features(features, inputs)
    assignments = stratified_folds(inputs.labels)
    predictions, sources = {}, {}
    for key, values in features.items():
        predictions[key], score = crossfit_ridge(values, inputs.labels, assignments)
        sources[key] = dict(score=score, geometry=feature_geometry(values))
    paired = {}
    def pair(baseline, candidate):
        result = paired_diagnostics(inputs.labels, baseline, candidate)
        result['limitation'] = 'Descriptive paired cross-fitted errors on encoder-training rows; no independent confirmation'
        return result
    for key in inputs.identity['sources']:
        paired[f'{key}/quadrants-minus-mean'] = pair(
            predictions[f'{key}/mean'], predictions[f'{key}/quadrants'])
        paired[f'{key}/quadrants-minus-shuffled'] = pair(
            predictions[f'{key}/shuffled_quadrants'], predictions[f'{key}/quadrants'])
        paired[f'{key}/quadrants-minus-repeated-mean'] = pair(
            predictions[f'{key}/repeated_mean'], predictions[f'{key}/quadrants'])
        if key.startswith('prediction-'):
            random = key.replace('prediction-', 'random-', 1)
            for mode in MODES:
                paired[f'{key}/{mode}-minus-random'] = pair(
                    predictions[f'{random}/{mode}'], predictions[f'{key}/{mode}'])
    means = {}
    for arm in ('prediction', 'random'):
        for mode in MODES:
            scores = [row['score']['accuracy'] for key, row in sources.items()
                      if key.startswith(arm+'-seed') and key.endswith('/'+mode)]
            means[f'{arm}/{mode}'] = dict(encoder_seeds=len(scores), accuracy_mean=statistics.mean(scores),
                accuracy_sample_std=statistics.stdev(scores) if len(scores) > 1 else None)
    return dict(version=1, count=len(inputs.labels), source_identity=inputs.identity, sources=sources,
        fold_assignments_sha256=digest(assignments.tolist()), folds=5, fold_seed=29, alpha=.1,
        paired=paired, encoder_seed_means=means, models_trained=False, classifiers_fitted=True,
        validation_images_read=False, test_images_read=False,
        limitation='Exploratory classifier cross-fitting only: encoders may have pretrained on every scoring image. '
                   'No hyperparameter search, independent confirmation, test score or quality improvement established.')
