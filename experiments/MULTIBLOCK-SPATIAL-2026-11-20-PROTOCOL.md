# Spatial readouts on the fixed fitting panel

The November 19 diagnostic found that global means changed little under patch
shuffling. Test whether pooling the same context tokens into four ordered
quadrants adds class information or makes predictions more sensitive to layout.

Reuse the November 18 checkpoints and seed-matched random initializations.
Use the same 500 CIFAR-10 fitting rows, five class-stratified folds from seed 29,
and 4×4 patch shuffle from seed 41. Reproduce the original and shuffled global
probe counts from the earlier audit. Read global means and four ordered
quadrant means from the same forward pass for each image view. Compare
regional features with a global mean repeated four times. Use ridge alpha 0.1
for mean and quadrant features. Use alpha 0.4 for the repeated mean so four
duplicate columns preserve the underlying global L2 penalty.

The low-level controls are RGB channel means and standard deviations over the
whole image and separately in each quadrant. Fit on original complement rows
only. Score original held rows, the same rows after patch shuffling, and, for
quadrant readouts, after reordering the four feature cells with seed 43. Reuse
the same fitted head for all views. The readout shuffle preserves cell contents.
The input shuffle preserves each 4×4 patch and changes which patches occupy
each region. Both perturbations can shift the distribution seen by a classifier.

Report the original quadrant minus global accuracy, paired recoveries and
regressions, and original quadrant comparisons with the matching random
encoder and regional RGB. Report both shuffled views and their differences
from the matching random control. Average seeds over the same rows. Keep all
model, feature and row-level artifacts local. This is a fitting-panel
diagnostic with possible encoder overlap. No new validation or test image is
indexed, and no encoder updates occur.

```sh
PYTHONPATH=src .venv/bin/python -m experiments.run_multiblock_spatial_20261120 --preflight
PYTHONPATH=src .venv/bin/python -m experiments.run_multiblock_spatial_20261120
PYTHONPATH=src .venv/bin/python -m experiments.run_multiblock_spatial_20261120 --replay
```

Replay verifies the existing checkpoints and feature cache, then recomputes
fold heads and compares the resulting aggregate. No image indexing or encoder
forward passes are needed during replay.
