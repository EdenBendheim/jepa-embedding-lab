# Controlled embedding-width experiment — October 15, 2026

Declared before new training or scores. The previous 32-D 2,223-step model did
not establish a useful advantage. This experiment isolates width rather than
changing position/input features, optimization or mask geometry simultaneously.

- Widths 32, 64 and 96; four heads, two encoder and one predictor block. Three
  seeds each: 20261003/20261010/20261011. Fixed 2x2 target / 0.75 context.
- Each model gets 2,223 steps, batch four, AdamW lr 0.001, default decay 0.01,
  EMA 0.99 and clip 1.0. Reuse the three verified October 14 width-32 checkpoints
  at exactly that budget; train the six wider models from scratch. Check every
  recorded image draw and match actual order across widths within each seed.
- Random encoders now use exactly the corresponding training initialization
  seed/config, improving the within-width control. This differs from the older
  29/31/37 random encoders; do not attribute changed baseline scores to width alone.
- Fixed 500 training / 200 tuning examples, context mean pooling. All widths and
  baselines get the same 200-step eight-setting AdamW grid, classifier seeds
  29/31/37, and ridge penalties 0.01/0.1/1/10. Add standardized cosine kNN with
  counts 1/5/15, selecting by tuning accuracy then declared order. Equal neighbor
  similarities use fitting-row order; vote ties select the lowest class ID.
- Lock all selected states before reading new confirmation pixels. Reserve 500
  balanced validation examples, seed 20261015, excluding every one of the 1,246
  previously observed validation examples. Score all states without changing
  normalization, refitting or retuning. Official test data stays reserved.
- Report individual seeds and encoder means; do not choose a width winner from
  the confirmation scores. Count parameters by component and report persistent
  FP32 state-memory estimates. Equal steps do not mean equal compute, capacity
  or downstream dimensions. This remains a small non-augmented pilot.

Exact source hashes, config and grids are in `width-sources-20261015.json`.
Public outputs contain metrics/configuration only. Actual raw data, checkpoints,
features, neighbor indices, classifier weights and journals stay ignored/local.
No API/cloud compute is used. These confirmation rows become observed after this
run and must not be reused as untuned confirmation for later choices.
