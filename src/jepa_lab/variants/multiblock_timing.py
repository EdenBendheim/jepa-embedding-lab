"""Measure steady-state training time with matched, counterbalanced CIFAR runs."""

import math
from statistics import median
from time import perf_counter_ns

from ..model import ModelConfig
from .cifar_batches import CifarTrainingData
from .multiblock_pair import GROUPING_ARMS
from .multiblock_train import _digest, train_multiblock


GROUPINGS = ('separate', 'joint')


def benchmark_grouping(data, *, seeds=(29, 31, 37), steps=32, warmup=4, batch_size=4):
    if not isinstance(data, CifarTrainingData):
        raise ValueError('Use verified CIFAR-10 training data')
    if (not 1 <= len(seeds) <= 3 or any(type(seed) is not int or not 0 <= seed < 2**63 for seed in seeds)
            or len(set(seeds)) != len(seeds)):
        raise ValueError('Use one to three distinct training seeds')
    if (type(steps) is not int or not 2 <= steps <= 100
            or type(warmup) is not int or not 1 <= warmup < steps):
        raise ValueError('Use 2-100 steps, with at least one warmup and one measured step')
    if type(batch_size) is not int or not 1 <= batch_size <= 8:
        raise ValueError('Use a batch size of 1-8')

    config = ModelConfig(patch_dim=48)
    trials = []
    for seed in seeds:
        reference = None
        for order in (GROUPINGS, GROUPINGS[::-1]):
            arms = {}
            for grouping in order:
                started = perf_counter_ns()
                result = train_multiblock(steps=steps, batch_size=batch_size, seed=seed,
                    config=config, training_data=data, measure_steps=True, **GROUPING_ARMS[f'{grouping}_2x2'])
                elapsed = perf_counter_ns()-started
                journal = result['history']
                identity = dict(initial_state_sha256=result['initial_state_sha256'],
                    image_order_sha256=_digest([row['train_indices'] for row in journal]),
                    mask_order_sha256=_digest([row['mask_sha256'] for row in journal]))
                if reference is None:
                    reference = identity
                elif identity != reference:
                    raise ValueError('Timing trials must share initialization, image order and masks within each seed')
                timings = result['timings'][warmup:]
                totals = [row['batch_ns']+row['update_ns']+row['diagnostics_ns'] for row in timings]
                arms[grouping] = dict(
                    trial_wall_seconds=elapsed/1e9, mean_step_ms=sum(totals)/len(totals)/1e6,
                    median_step_ms=median(totals)/1e6,
                    median_batch_ms=median(row['batch_ns'] for row in timings)/1e6,
                    median_update_ms=median(row['update_ns'] for row in timings)/1e6,
                    median_diagnostics_ms=median(row['diagnostics_ns'] for row in timings)/1e6)
            trials.append(dict(seed=seed, order=list(order), identity=reference, arms=arms,
                joint_over_separate_step_time=arms['joint']['mean_step_ms']/arms['separate']['mean_step_ms']))

    return dict(version=1, kind='predictor_grouping_cpu_timing',
                manifest_sha256=data.manifest['manifest_sha256'], steps_per_trial=steps,
                warmup_steps=warmup, measured_steps_per_trial=steps-warmup, batch_size=batch_size,
                trials=trials,
                median_joint_over_separate_step_time=median(trial['joint_over_separate_step_time'] for trial in trials),
                limitation='Steady-state CPU wall time on matched training rows. Trial wall time includes setup '
                           'and warmup. Validation, test and checkpoint I/O are outside the measurement.')


def estimate_step_budget(report, *, seconds_per_arm):
    """Size a future run from the slowest observed trial mean plus a 25% margin."""
    if (type(seconds_per_arm) not in (int, float) or not math.isfinite(seconds_per_arm)
            or seconds_per_arm <= 0):
        raise ValueError('Give a finite positive training-loop budget per arm')
    try:
        means = {grouping: [trial['arms'][grouping]['mean_step_ms'] for trial in report['trials']]
                 for grouping in GROUPINGS}
    except (KeyError, TypeError) as exc:
        raise ValueError('Timing report is missing paired trial means') from exc
    if any(not values or any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in values)
           for values in means.values()):
        raise ValueError('Timing report requires positive finite trial means')
    seconds_per_step = {grouping: (max(values) / 1000) * 1.25 for grouping, values in means.items()}
    arm_steps = {grouping: math.floor(min(1000, seconds_per_arm / duration))
                 for grouping, duration in seconds_per_step.items()}
    return dict(seconds_per_arm=seconds_per_arm, margin=1.25,
        estimated_seconds_per_step=seconds_per_step, equal_time_steps=arm_steps,
        equal_update_steps=min(arm_steps.values()),
        limitation='Planning estimate for steady-state training steps on this machine. '
                   'Measure actual elapsed time separately from setup, checkpoint writes and evaluation.')
