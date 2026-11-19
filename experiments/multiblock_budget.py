"""Plan shared checkpoints for longer matched CIFAR-10 training runs."""

from jepa_lab.variants.multiblock_timing import estimate_step_budget


def budget_schedule(timing, *, seconds_per_arm, chunk_steps):
    """Keep the equal-update checkpoint on both per-arm trajectories."""
    if type(chunk_steps) is not int or not 1 <= chunk_steps <= 100:
        raise ValueError('Use chunks of 1–100 updates')
    budget = estimate_step_budget(timing, seconds_per_arm=seconds_per_arm)
    shared = budget['equal_update_steps']
    targets = budget['equal_time_steps']
    if shared < 1 or any(steps >= 1000 for steps in targets.values()):
        raise ValueError('Choose a budget with at least one update per arm and neither arm capped')
    checkpoints = {}
    for arm, target in targets.items():
        boundaries = sorted({shared, target})
        ends = []
        start = 0
        for end in boundaries:
            ends.extend(range(start + chunk_steps, end, chunk_steps))
            ends.append(end)
            start = end
        checkpoints[arm] = ends
    return dict(planning=budget, chunk_steps=chunk_steps, checkpoints=checkpoints)
