# Color and layout audit of the longer grouping experiment

The November 18 validation run found that extended joint training improved over
the separate predictor and scored just below global RGB statistics. Before
using any more validation rows, inspect the existing checkpoint representations
on the original 500 fitting rows.

Use the completed 658-update separate and joint checkpoints, the 960-update
joint checkpoints, and their seed-matched random encoders. Reopen and verify the
checkpoint journals against the published training record. Extract full-image
context means once per original and shuffled image. Shuffle whole 4×4 patches
using a seed of 41 and an image-index-derived permutation. Every source sees
identical original and shuffled views. This preserves the pixels and local
4×4 structure while changing spatial layout.

Run these fixed diagnostics:

1. Measure feature effective rank and aligned CKA across checkpoints and views.
2. With five stratified folds fixed by seed 29, predict each original embedding
   from the RGB channel means and standard deviations using ridge alpha 0.01.
   Compare against permuted fitting pairs and report out-of-fold reconstruction
   R² and residual effective rank.
3. In each of the same folds, fit an RGB residualizer and then fit fixed ridge
   classifiers with alpha 0.1 on the training complement. Compare RGB alone,
   residual embeddings, and RGB plus residual embeddings. Each held row is
   transformed and scored by models fitted within its training complement.
4. Fit one classifier per source and fold on original images. Apply it to the
   original and shuffled held rows. Compare paired errors, including the
   matching random encoder controls.

The sample is within the training partition and can overlap encoder training.
The diagnostics describe the sample and do not establish generalization, remove
every color signal, or explain the earlier validation difference. Patch
shuffling changes the input distribution. No new validation rows are reserved
or indexed. The official test set remains reserved.

```sh
PYTHONPATH=src .venv/bin/python -m experiments.run_multiblock_audit_20261119 --preflight
PYTHONPATH=src .venv/bin/python -m experiments.run_multiblock_audit_20261119
PYTHONPATH=src .venv/bin/python -m experiments.run_multiblock_audit_20261119 --replay
```

Replay checks all source and feature fingerprints and recomputes the small
fold fits from the local cache. It reads no image rows and runs no encoder
forward passes or training updates. Publish aggregate results only.
