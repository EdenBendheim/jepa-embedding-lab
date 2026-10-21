# Positional amplitude — prospective October 20, 2026

[Pinned recipe](scale-sources-20261020.json). Declared before new scaled-model
training or scoring. This isolates amplitude after the October 18 comparison
changed positional form, scale and parameter count together.

- Compare fixed sin/cos positions at multipliers **0.02 and 1.0**, in context,
  EMA teacher and predictor. Both arms have identical trainable capacity and
  initial state/RNG consumption. The smaller multiplier is a declared contrast,
  not a claim to exactly match learned-position RMS or an optimized setting.
- Width 32, four heads, encoder/predictor depths 2/1; seeds
  20261003/20261010/20261011; 2,223 updates, batch four, AdamW LR .001, EMA .99;
  2x2 target and .75 visible-complement fraction. Reuse pinned scale-one runs
  from October 18; new .02 runs must match actual image-order journals. Do not
  resume a scale-one checkpoint at .02. Source hashes alone do not prove this
  trajectory match; the training/evaluation driver must check it before scoring.
- Same historical 500/200 fitting/tuning rows. Full-image context mean pooling,
  exact-initialization random controls at each scale, shared pixels. AdamW:
  LR .0001/.001/.01/.05 × decay .01/.1, 200 steps, seeds 29/31/37. Ridge:
  .01/.1/1/10. Cosine neighbors: 1/5/15. These are reused selection settings,
  not a new independent hyperparameter search.
- Retain all **2,246** observed validation rows through October 18, replayed
  from the pinned six-source exposure bundle. Use the existing **unscored**
  October 19 reservation check: 50 new rows/class, seed 20261020, selection
  `a89a88162421638681ecf1c435a1f902720ecad019b024f784fc67818441dda2`.
  Reservation is not observation. Do not register these as observed until used
  for evaluation/tuning; then extend the complete ledger with original evidence.
- Lock selected heads/indices before fresh extraction. No fresh-row fitting,
  normalization changes, tuning or picking favorable seeds. Primary descriptive
  contrast: trained scale .02 minus 1.0 AdamW accuracy on those same rows, averaged
  over encoder/classifier seeds. Report matching random/pixel results, ridge,
  neighbors and paired recoveries/regressions, including unfavorable outcomes.
  Shared rows across seeds are not independent samples. No claim of an isolated
  learning advantage without the matched random controls.
- Official test remains reserved. Local one-thread CPU only, bounded chunks;
  no paid compute/APIs. Keep images, features, checkpoints, source bundles and
  row-level predictions/journals under ignored paths. Publish aggregates only.

## Preparation and current status

`PYTHONPATH=src .venv/bin/python experiments/prepare_scale_20261020.py` rechecks
the exact protocol/source bytes, dataset manifest and original history files,
then recreates the same unscored reservation. It refuses to replace evidence or
a conflicting cached reservation. Its stdout is aggregate-only. It does not
train, read individual images, fit a classifier, or score the held-out rows.

Model scaling, CLI/resume binding and saved-feature scale comparisons now work.
[Preparation aggregates](scale-2026-10-20-preparation.json) retain the source,
ledger and selection fingerprints, 500 unscored rows and equal 29,472 trainable
parameters per arm. All 167 local tests, the mask demo and three-step scaled
synthetic training smoke passed. Synthetic loss is an execution check only.
Real scaled-model training and the scale-specific end-to-end frozen evaluation
driver remain next. No new real-image accuracy or quality improvement is claimed.
