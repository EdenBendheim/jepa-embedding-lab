# Longer CIFAR-10 predictor grouping comparison

The earlier timing trials suggest that joint prediction can process more images
in a similar CPU budget. This study compares frozen features at the same update
count and at an extended joint checkpoint. All budgets and classifier settings
are fixed before training and before reserving any new validation rows.

Use the November 14 timing report and a seven-second projection with the
existing 25% margin. The separate arm trains for 658 updates. The joint arm
saves a checkpoint at 658 and continues to 960 updates. Both arms use four
disjoint 2×2 target blocks, 36 context tokens, a batch of four, and the same
architecture and optimizer settings. Seeds 29, 31 and 37 each use identical
initialization, masks and image order through 658 updates. Reverse arm order
for seed 31. Keep chunks at 100 updates or fewer and retain checkpoint timing
receipts. Record actual training-loop and trainer wall time for all milestones.

Fit a fixed normalized ridge classifier with alpha 0.1 for each frozen
full-image context encoder using the same 500 existing fitting rows. Include
a matching random encoder for each seed, six RGB statistics and normalized
raw pixels. Fit all heads before reserving or indexing validation images.

Reserve 300 fixed validation images, 30 per class, with seed 20261118. Exclude
every row in the source-replayed prior exposure ledger. Register the reservation
before accessing images. Keep the official test images out of scoring. Report
per-seed accuracy, average accuracy, paired recoveries and regressions, and
actual elapsed training time. Compare joint 658 versus separate 658 first, then
joint 960 versus separate 658 and joint 658. All seed comparisons share the same
300 rows. The projected budget is a planning device, so report the realized
times before interpreting the longer trajectory as a compute comparison.

The machine-readable declaration pins the earlier timings, split, fitting
selection, exposure evidence, package sources, experiment scripts and runtime.
The replay command refuses missing checkpoints and classifier artifacts, reuses
fitted weights, reads only the registered scoring rows, and checks exact scores.

```sh
PYTHONPATH=src .venv/bin/python -m experiments.run_multiblock_budget_20261118 --preflight
PYTHONPATH=src .venv/bin/python -m experiments.run_multiblock_budget_20261118
PYTHONPATH=src .venv/bin/python -m experiments.run_multiblock_budget_20261118 --replay
```
