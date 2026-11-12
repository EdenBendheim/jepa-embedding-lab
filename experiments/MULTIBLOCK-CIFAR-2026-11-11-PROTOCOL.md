# Multi-block CIFAR-10 training-row pilot

Compare one 4×4 target tile with four disjoint 2×2 target tiles on the
fixed CIFAR-10 training split. Both arms use 16 target patches and 36 visible
context patches per image. This contrast varies target geometry and predictor
grouping together.

Use the [pinned recipe](multiblock-cifar-protocol-20261111.json) and its
[driver](run_multiblock_20261111.py). Run 32 AdamW updates per arm, with four
images per update, seed 29, learning rate .001, EMA .99, and one local CPU
thread. Both arms share the same initialized tensors, sample order, RGB
normalization, width-32 encoder and Smooth L1 reduction. Verify sampled rows
and initialization from both checkpoints. Record actual predictor calls,
sequence lengths and attention-pair counts. The encoder budgets match, and
the predictor budgets differ.

Freeze each context encoder and mean-pool all 64 tokens. Compare both with a
random encoder at the same initialization and six per-image RGB statistics.
Use the original 500 balanced fitting rows in the pinned pilot selection.
Fit five class-stratified ridge folds with seed 29 and fixed regularization .1.
Every representation uses the same images, labels and folds. Compute paired
errors and count any overlap between scoring rows and encoder-training rows.

The existing exposure receipt stays pinned. CIFAR file hashes and labels are
verified before training. The driver indexes training images only and keeps
the official test set reserved. Publish aggregates and provenance hashes.
Store checkpoints, sample journals and any local features in ignored paths.

Run from the repository root after preparing the original manifest and pilot
selection. Use the pinned Python and Torch versions.

```sh
PYTHONPATH=src .venv/bin/python experiments/run_multiblock_20261111.py --preflight
PYTHONPATH=src .venv/bin/python experiments/run_multiblock_20261111.py
```

The fixed budget tests the end-to-end pipeline on public data. A single seed
and classifier folds inside the original training split provide exploratory
accuracy. Held-out representation quality needs a separately declared
comparison with matched controls and a verified exposure reservation.
