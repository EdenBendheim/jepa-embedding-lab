# Timing joint and separate prediction

Measure actual CPU wall time for the two November 12 predictor groupings on
the same CIFAR-10 training images. Each trial pairs four disjoint 2×2 blocks
predicted separately with the same blocks predicted jointly. Use a fresh model
in each arm. Verify model initialization, sampled image order and masks
match across all runs for each seed.

For seeds 29, 31 and 37, run each arm in both orders with 32 updates, batch
size four and one CPU thread. Use the width-32 model and the original
optimizer and EMA settings. Omit the first four updates of each run from the
per-step summary. Run sequentially. Use the sum of batch preparation, update
and diagnostic wall times for the per-step comparison. Report a paired ratio
of mean step time for each order, then the median of the six ratios. Report
full trial wall time separately, including startup. Neither timing includes
evaluation or checkpoint writes, and per-step timing excludes model setup.

To size a longer local experiment, start with a 10-second training-loop budget
per arm per seed. For each grouping, use the slowest trial mean plus a 25%
margin and round down to whole updates, within the 1,000-step trajectory
limit. Use the smaller count for an equal-update comparison. Use the separate
counts for a comparison with approximately equal elapsed training time.
Verify actual elapsed time when running either budget.

The [recipe](multiblock-timing-protocol-20261114.json) pins the sources,
runtime, verified CIFAR manifest and existing exposure receipt. Only
training images are indexed. This experiment produces CPU timing estimates
without fitting any downstream classifier or scoring validation or test rows.

```sh
PYTHONPATH=src .venv/bin/python experiments/run_multiblock_timing_20261114.py --preflight
PYTHONPATH=src .venv/bin/python experiments/run_multiblock_timing_20261114.py \
  --output runs/multiblock-timing-20261114.json
```
