"""Test how much frozen feature variation is predictable from six RGB statistics."""

from jepa_lab.analysis.crossfit import crossfit_reconstruction
from jepa_lab.feature_geometry import feature_geometry
from jepa_lab.ridge import RidgeConfig


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
