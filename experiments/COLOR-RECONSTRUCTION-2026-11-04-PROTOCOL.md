# Fitting-only RGB reconstruction diagnostic — November 4, 2026

Declared before inspecting real-image reconstruction outcomes. This follows the
[November 3 audit](OBJECTIVE-REPRESENTATIONS-2026-11-03.md): image-level variance
increased with the objective penalty, but effective rank and downstream scores
fell. A simple explanation worth checking is that the spread is largely
predictable from image color and contrast.

## Fixed inputs and method

- Reuse all ten cached **training** feature sources from the completed objective
  comparison: shared raw pixels and random, prediction-only and variance-penalty
  context encoders at each of three existing encoder seeds. No model updates,
  new feature extraction, classifier selection or fresh-row access.
- Use exactly the original 500 probe-fitting rows, in their pinned order.
- Extract six controls per uint8 RGB image: each channel's mean and population
  standard deviation after division by 255. No spatial information or labels.
- Assign five fixed folds with a local seed-29 permutation, round-robin IDs.
  Each source uses identical assignments: 400 fitting and 100 scoring rows per
  fold. Score each row once using a head that did not fit that row.
- Fit a continuous ridge reconstruction from six controls to the source's
  feature vector. Standardize controls from fitting rows only, center targets
  using fitting rows only, retain an unpenalized intercept, and solve in float64
  with alpha .01 and an N-scaled penalty. No alpha search.
- As a negative control, repeat after permuting **only fitting controls** within
  each fold using seed `41 + fold`; keep scoring controls and all targets aligned.
  This is one fixed control permutation per fold, not a permutation p-value.
- Report aggregate per-element MSE and reconstruction R² against a reference
  using each row's fitting-fold target mean. Report original and out-of-fold
  residual feature geometry, including covariance entropy effective rank.
- Summarize the three encoder seeds descriptively. Do not use the original
  validation or confirmation rows to choose settings; official test stays reserved.

The JSON protocol binds every feature artifact, completed checkpoint, ordered
selection digest, recipe, runtime and recursively hashed local Python source.
A cache is immutable: mismatches must use a new path. Replay revalidates source
artifacts and hashes without image indexing or reconstruction fitting. Checksums
detect accidental changes; they are not authentication against deliberate forgery.

## Interpretation and limits

High R² indicates a linear mapping from these six statistics explains much of
this source's feature variation on the fixed folds. It does not show that color
is the only learned signal, that color causes the earlier accuracy drop, or that
removing it improves classification. Residual effective rank describes errors of
five different fitting heads; it does not define a deployable residual encoder.
Pixels provide a spatially detailed reference, not an expected perfect positive
control. Low R² would only reject this particular linear explanation.

Do not tune a follow-up on already observed confirmation outcomes. Use these
fitting-only observations to choose a hypothesis, then predeclare its training
and evaluation, retaining the existing exposure history.

No raw images, row-level vectors, fitted reconstruction coefficients, feature
artifacts or model weights will be published. Public outputs contain aggregates
and provenance hashes only. Local CPU compute only.
