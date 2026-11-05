"""Fixed-fold reconstruction checks within a supplied fitting sample."""

import torch

from ..probes import _values
from ..ridge import RidgeConfig
from .reconstruction import fit_reconstruction


def make_folds(count, *, folds=5, seed=29):
    if (type(count) is not int or not 4 <= count <= 1000
            or type(folds) is not int or not 2 <= folds <= min(10, count)
            or type(seed) is not int or not 0 <= seed < 2**63):
        raise ValueError('Use 4-1000 rows, 2-10 nonempty folds and an integer seed')
    order = torch.randperm(count, generator=torch.Generator().manual_seed(seed))
    assignments = torch.empty(count, dtype=torch.long)
    assignments[order] = torch.arange(count) % folds
    return assignments


@torch.inference_mode()
def crossfit_reconstruction(source, target, assignments, *, permute_training=False, config=RidgeConfig(.01)):
    _values(source)
    _values(target)
    if (len(source) != len(target) or not isinstance(assignments, torch.Tensor)
            or assignments.device.type != 'cpu' or assignments.dtype != torch.long
            or assignments.shape != (len(source),) or type(permute_training) is not bool):
        raise ValueError('Use aligned source/target rows and one integer fold per row')
    folds = sorted(assignments.unique().tolist())
    if folds != list(range(len(folds))) or not 2 <= len(folds) <= 10:
        raise ValueError('Fold IDs must be contiguous, nonnegative and cover 2-10 folds')
    if any(int((assignments != fold).sum()) < 2 for fold in folds):
        raise ValueError('Each fold needs at least two fitting rows')
    predictions = torch.empty_like(target)
    reference = torch.empty_like(target)
    rows = []
    for fold in folds:
        fitting = (assignments != fold).nonzero().flatten()
        held = (assignments == fold).nonzero().flatten()
        source_fitting = source[fitting]
        if permute_training:
            order = torch.randperm(len(fitting), generator=torch.Generator().manual_seed(41 + fold))
            source_fitting = source_fitting[order]
        model = fit_reconstruction(source_fitting, target[fitting], config)
        predictions[held] = model.predict(source[held])
        reference[held] = model.target_mean.float()
        rows.append(dict(fold=fold, fitting_count=len(fitting), scoring_count=len(held)))
    squared_error = float((target.double() - predictions.double()).square().sum())
    reference_error = float((target.double() - reference.double()).square().sum())
    report = dict(count=len(source), target_dim=target.shape[1], alpha=config.alpha,
                  training_mapping='permuted within fitting fold' if permute_training else 'aligned',
                  permutation_seed_rule='41 + fold', folds=rows,
                  mean_squared_error=squared_error / target.numel(),
                  fitting_mean_reference_mse=reference_error / target.numel(),
                  reconstruction_r2=1 - squared_error / reference_error if reference_error else None,
                  r2_reference='Each scored row uses the target mean of its fitting folds',
                  limitation='Descriptive reconstruction within the supplied sample and fixed folds')
    return predictions, reference, report
