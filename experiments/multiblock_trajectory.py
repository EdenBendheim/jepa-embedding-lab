"""Save and resume a timed multi-block trajectory at declared update boundaries."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter_ns

from jepa_lab.compare import write_report
from jepa_lab.objective_run import file_sha
from jepa_lab.variants.multiblock_pair import GROUPING_ARMS
from jepa_lab.variants.multiblock_train import _digest, _recipe, train_multiblock


def checkpoint_path(directory, step):
    return Path(directory) / f'step-{step:04d}' / 'encoder.pt'


def _existing_chunks(directory, ends, trajectory_sha):
    receipts = []
    missing = False
    start = 0
    for end in ends:
        checkpoint = checkpoint_path(directory, end)
        if not checkpoint.parent.exists():
            missing = True
        else:
            if missing:
                raise ValueError('Trajectory checkpoints must form a contiguous prefix')
            receipt = json.loads((checkpoint.parent / 'receipt.json').read_text())
            times = [receipt[key] for key in ('loop_ns', 'wall_ns')]
            if (receipt['trajectory_sha256'] != trajectory_sha
                    or (receipt['start_step'], receipt['end_step']) != (start, end)
                    or receipt['checkpoint_sha256'] != file_sha(checkpoint)
                    or any(type(value) is not int or value <= 0 for value in times)
                    or times[0] > times[1]):
                raise ValueError('Checkpoint chunk differs from its timing receipt or trajectory')
            receipts.append(receipt)
        start = end
    return receipts


def run_trajectory(data, directory, *, grouping, ends, seed, batch_size, config, replay=False):
    """Resume the first missing chunk, or verify existing artifacts without training."""
    if grouping not in ('separate', 'joint'):
        raise ValueError('Choose separate or joint prediction')
    if (not ends or any(type(step) is not int for step in ends) or ends[-1] > 1000
            or any(not 1 <= end-start <= 100 for start, end in zip([0, *ends], ends))):
        raise ValueError('Use increasing checkpoint boundaries with 1–100 updates per chunk')
    geometry = GROUPING_ARMS[f'{grouping}_2x2']
    options = dict(batch_size=batch_size, seed=seed, learning_rate=.001, momentum=.99,
                   context_fraction=.75, config=config, training_data=data) | geometry
    _, identity = _recipe(**options)
    trajectory_sha = _digest(dict(recipe=identity, checkpoint_steps=list(ends)))
    directory = Path(directory)
    receipts = _existing_chunks(directory, ends, trajectory_sha)
    if replay and len(receipts) != len(ends):
        raise ValueError('Replay requires every declared checkpoint chunk')
    for end in ends[len(receipts):]:
        start = receipts[-1]['end_step'] if receipts else 0
        previous = checkpoint_path(directory, start) if start else None
        target = checkpoint_path(directory, end)
        directory.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix='.staged-', dir=directory) as temporary:
            staged = Path(temporary) / target.parent.name
            staged.mkdir()
            checkpoint = staged / target.name
            before = perf_counter_ns()
            result = train_multiblock(steps=end-start, checkpoint=checkpoint,
                                      resume=previous, measure_steps=True, **options)
            wall = perf_counter_ns()-before
            if (result['start_step'], result['completed_step']) != (start, end):
                raise ValueError('Trainer did not complete the declared chunk')
            timings = result['timings']
            loop = sum(row['batch_ns'] + row['update_ns'] + row['diagnostics_ns'] for row in timings)
            if len(timings) != end-start or not 0 < loop <= wall:
                raise ValueError('Invalid measured chunk durations')
            receipt = dict(trajectory_sha256=trajectory_sha, start_step=start, end_step=end,
                           checkpoint_sha256=file_sha(checkpoint), loop_ns=loop, wall_ns=wall)
            write_report(staged / 'receipt.json', receipt)
            # Publish state and timing together. A completed destination is nonempty.
            staged.rename(target.parent)
            receipts.append(receipt)
    return receipts
