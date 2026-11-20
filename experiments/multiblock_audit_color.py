"""Test how much frozen feature variation is predictable from six RGB statistics."""

import torch

from jepa_lab.analysis.crossfit import crossfit_reconstruction
from jepa_lab.analysis.reconstruction import fit_reconstruction
from jepa_lab.analysis.spatial_evaluation import crossfit_ridge
from jepa_lab.diagnostics import paired_diagnostics
from jepa_lab.feature_geometry import feature_geometry
from jepa_lab.ridge import RidgeConfig, fit_ridge


def measure_rgb_reconstruction(panel, assignments, *, alpha):
    """Score aligned and permuted fitting mappings on exactly the same folds."""
    rgb = panel['rgb']
    config = RidgeConfig(alpha)
    result = {}
    for key, target in panel.items():
        if not key.endswith('/original'):
            continue
        predicted, _, aligned = crossfit_reconstruction(rgb, target, assignments, config=config)
        _, _, permuted = crossfit_reconstruction(rgb, target, assignments,
                                                permute_training=True, config=config)
        result[key.removesuffix('/original')] = dict(
            aligned=aligned, permuted=permuted, residual_geometry=feature_geometry(target-predicted))
    return result


def residual_classification(panel, labels, assignments, *, reconstruction_alpha, classifier_alpha):
    """Compare RGB, residual embeddings and their combination on fixed folds."""
    rgb = panel['rgb']
    reconstruction = RidgeConfig(reconstruction_alpha)
    classifier = RidgeConfig(classifier_alpha)
    rgb_predictions, rgb_score = crossfit_ridge(rgb, labels, assignments, config=classifier)
    comparisons = {}
    for key, original in panel.items():
        if not key.endswith('/original'):
            continue
        residual_predictions = torch.empty_like(labels)
        combined_predictions = torch.empty_like(labels)
        for fold in assignments.unique(sorted=True).tolist():
            fitting, scoring = assignments != fold, assignments == fold
            # Fit both transformations inside this classifier fold. A single
            # panel-wide table of out-of-fold residuals could mix fold information.
            rgb_fit, rgb_scoring = rgb[fitting], rgb[scoring]
            fit_values, scoring_values = original[fitting], original[scoring]
            projection = fit_reconstruction(rgb_fit, fit_values, reconstruction)
            fit_residual = fit_values - projection.predict(rgb_fit)
            scoring_residual = scoring_values - projection.predict(rgb_scoring)
            residual_head = fit_ridge(fit_residual, labels[fitting], classifier)
            combined_head = fit_ridge(torch.cat((rgb_fit, fit_residual), dim=1), labels[fitting], classifier)
            residual_predictions[scoring] = residual_head.logits(scoring_residual).argmax(1)
            combined_predictions[scoring] = combined_head.logits(
                torch.cat((rgb_scoring, scoring_residual), dim=1)).argmax(1)
        paired = {}
        for name, predictions in (('residual', residual_predictions), ('rgb_plus_residual', combined_predictions)):
            stats = paired_diagnostics(labels, rgb_predictions, predictions)
            paired[name] = {field: stats[field] for field in
                ('count', 'baseline_correct', 'candidate_correct', 'recovered', 'regressed', 'accuracy_delta')}
        comparisons[key.removesuffix('/original')] = paired

    return dict(rgb=rgb_score, comparisons=comparisons, reconstruction_alpha=reconstruction_alpha,
                classifier_alpha=classifier_alpha)
