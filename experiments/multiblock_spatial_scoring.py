"""Apply original-fold heads to aligned spatial perturbations."""

import torch

from jepa_lab.analysis.spatial_control import shuffle_quadrants
from jepa_lab.analysis.spatial_evaluation import crossfit_ridge
from jepa_lab.diagnostics import paired_diagnostics
from jepa_lab.probes import validate_probe_data
from jepa_lab.ridge import RidgeConfig, fit_ridge


READOUTS = ('mean', 'repeated_mean', 'quadrants')


def _pair(labels, before, after):
    full = paired_diagnostics(labels, before, after)
    return {key: full[key] for key in ('count', 'baseline_correct', 'candidate_correct',
        'recovered', 'regressed', 'predictions_changed', 'accuracy_delta', 'per_class')}


def _score_views(original, alternatives, labels, folds, config):
    for values in alternatives.values():
        validate_probe_data(values, labels, dimension=original.shape[1])
    predictions = {name: torch.empty_like(labels) for name in alternatives}

    def fit_original(fold, fitting, fitting_labels, settings):
        head = fit_ridge(fitting, fitting_labels, settings)
        held = folds == fold
        for name, values in alternatives.items():
            predictions[name][held] = head.logits(values[held]).argmax(1)
        return head

    predictions['original'], base = crossfit_ridge(
        original, labels, folds, config=config, fold_provider=fit_original)
    transfer = {name: _pair(labels, predictions['original'], predictions[name])
                for name in alternatives}
    result = dict(original=base, transfer=transfer, dimensions=original.shape[1], alpha=config.alpha)
    return result, predictions


def score_spatial(panel, labels, indices, folds, *, alpha, quadrant_seed):
    """Compare global, repeated-global and ordered quadrant classifier readouts."""
    if len(labels) != len(indices):
        raise ValueError('Row labels and fixed image indices must align')
    reports, predictions = {}, {}
    sources = [name.removesuffix('/original/mean') for name in panel if name.endswith('/original/mean')]
    for source in sources:
        for readout in READOUTS:
            key = 'mean' if readout == 'repeated_mean' else readout
            original = panel[f'{source}/original/{key}']
            patched = panel[f'{source}/patch_shuffle/{key}']
            config = RidgeConfig(alpha)
            if readout == 'repeated_mean':
                original, patched = original.repeat(1, 4), patched.repeat(1, 4)
                # Four duplicates divide the minimum weight norm by four.
                # Scale alpha to preserve the underlying global ridge penalty.
                config = RidgeConfig(4*alpha)
            views = {'patch_shuffle': patched}
            if readout == 'quadrants':
                views['quadrant_shuffle'] = shuffle_quadrants(original, indices, seed=quadrant_seed)
            report, preds = _score_views(original, views, labels, folds, config)
            reports[f'{source}/{readout}'] = report
            predictions[f'{source}/{readout}'] = preds
    return reports, predictions
