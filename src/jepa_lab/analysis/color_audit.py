"""Reconstruct frozen representations from RGB controls on fitting rows only."""

import hashlib
import json

import torch

from ..cifar10 import CIFAR10Binary
from ..feature_geometry import feature_geometry
from ..objective_features import feature, source_specs
from .color_statistics import FEATURE_NAMES, rgb_statistics
from .crossfit import crossfit_reconstruction, make_folds


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def verified_sources(run):
    """Preflight every completed source before any diagnostic image indexing."""
    reports = run.training(reuse_only=True)
    recipe, manifest, fitting, _, labels, _ = run.inputs()
    indices = fitting['partitions']['train']['indices']
    if (not 5 <= len(indices) <= 1000 or len(indices) != len(set(indices))
            or not set(indices).issubset(manifest['partitions']['train']['indices'])):
        raise ValueError('Color audit requires 5-1000 distinct rows from the training split')
    sources = {key: feature(run, recipe, manifest, fitting, labels, key, 'train', reuse_only=True)
               for key in source_specs(recipe)}
    if any(value.indices != tuple(indices) for value in sources.values()):
        raise ValueError('Color audit sources must have exactly the pinned fitting order')
    identity = dict(version=1, protocol_sha256=run.expected_sha,
        manifest_sha256=manifest['manifest_sha256'], selection_sha256=fitting['selection_sha256'],
        ordered_training_indices_sha256=digest(indices), count=len(indices),
        checkpoints={name: row['checkpoint_sha256'] for name, row in reports.items()},
        sources={key: value.summary() for key, value in sources.items()},
        rgb_features=list(FEATURE_NAMES), folds=5, fold_seed=29, alpha=.01,
        permutation_seed_rule='41 + fold',
        runtime=dict(torch=str(torch.__version__), cpu_threads=torch.get_num_threads()))
    return identity, indices, sources


@torch.inference_mode()
def measure(run, identity, indices, sources):
    dataset = CIFAR10Binary(run.root)
    controls = rgb_statistics(torch.stack([dataset[index][0] for index in indices]))
    assignments = make_folds(len(indices), folds=identity['folds'], seed=identity['fold_seed'])
    arms = {}
    for key, value in sources.items():
        predicted, _, aligned = crossfit_reconstruction(controls, value.features, assignments)
        _, _, permuted = crossfit_reconstruction(controls, value.features, assignments, permute_training=True)
        arms[key] = dict(aligned=aligned, permuted=permuted,
            original_geometry=feature_geometry(value.features),
            residual_geometry=feature_geometry(value.features - predicted))
    return dict(version=1, identity=identity, arms=arms,
        fold_assignments_sha256=digest(assignments.tolist()),
        partition='train', reconstruction_heads_fitted=True, classifiers_fitted=False,
        models_trained=False, validation_images_read=False, test_images_read=False,
        limitation='Fixed five-fold diagnostics within the original fitting selection; '
                   'not a new held-out comparison, causal attribution, or downstream quality score. '
                   'Residual geometry describes out-of-fold errors, not learned residual features.')


def inspect_fitting(run):
    identity, indices, sources = verified_sources(run)
    return measure(run, identity, indices, sources)
