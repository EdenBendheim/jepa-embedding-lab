"""Score trained and random representations after a color-preserving layout change."""

import torch

from jepa_lab.analysis.spatial_evaluation import crossfit_ridge
from jepa_lab.diagnostics import paired_diagnostics
from jepa_lab.probes import validate_probe_data
from jepa_lab.ridge import RidgeConfig, fit_ridge


def layout_transfer(panel, labels, assignments, *, alpha):
    """Hold each fold's original-image classifier fixed when testing shuffled views."""
    config = RidgeConfig(alpha)
    comparisons = {}
    for key, original in panel.items():
        if not key.endswith('/original'):
            continue
        source = key.removesuffix('/original')
        shuffled = panel[f'{source}/shuffled']
        validate_probe_data(shuffled, labels, dimension=original.shape[1])
        heads = {}

        def fit_original(fold, features, fitting_labels, settings):
            head = fit_ridge(features, fitting_labels, settings)
            heads[fold] = head
            return head

        original_predictions, _ = crossfit_ridge(original, labels, assignments,
                                                  config=config, fold_provider=fit_original)
        shuffled_predictions = torch.empty_like(labels)
        for fold, head in heads.items():
            scoring = assignments == fold
            shuffled_predictions[scoring] = head.logits(shuffled[scoring]).argmax(1)
        stats = paired_diagnostics(labels, original_predictions, shuffled_predictions)
        comparisons[source] = {field: stats[field] for field in (
            'count', 'baseline_correct', 'candidate_correct', 'recovered', 'regressed',
            'predictions_changed', 'accuracy_delta', 'per_class')}
    return dict(comparisons=comparisons, alpha=alpha,
                interpretation='A shuffled view is a distribution shift. The score change measures '
                               'sensitivity of a fixed head to this particular intervention.')
