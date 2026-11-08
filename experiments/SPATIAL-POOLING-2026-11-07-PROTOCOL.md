# Fitting-only spatial pooling comparison — November 7, 2026

This protocol is declared before real-image spatial comparison outcomes. The
[RGB reconstruction audit](COLOR-RECONSTRUCTION-2026-11-04.md) left a simple
linear-color explanation incomplete. The next question is whether global mean
pooling hides class-relevant layout in the existing frozen patch tokens.

## Fixed comparison

- Reuse the three completed **prediction-only** context encoders from the
  October 31 objective comparison, each at 2,223 updates, and its exact seeded
  random-encoder controls. No encoder training or checkpoint selection.
- Use only the original 500 probe-fitting images, with their verified cached
  global-mean and raw-pixel controls. No validation, confirmation or official-test
  image tensors will be indexed for this comparison.
- Extract four views from the same full-image context-token forward pass:
  global mean (32 dimensions), four quadrant means concatenated in row-major
  order (128), image-specific shuffled quadrant means (128), and the global mean
  repeated four times (128). Every 4-by-4-patch quadrant contains 16 tokens.
- The shuffle permutes whole cell vectors, not channels, using the existing
  `quadrants-v1` SHA-256 image-index seed rule at seed 29. It is stable across
  batch ordering and identical across encoders for the same image. This tests
  consistent cell layout, not image-patch shuffling inside the encoder.
- Repeated means add classifier coordinates without spatial information. They
  match the number of weight parameters but not independent feature rank.
  At fixed ridge alpha, repetition can also change the effective penalty on
  shared signal; this control is not a capacity-equivalence proof.
- Include shared raw pixels (3,072 dimensions) and six RGB channel mean/std
  statistics. RGB statistics use population standard deviation after division
  by 255; their cache is checked against the original verified pixel features.
- Stratify five folds by all ten classes using local RNG seed 29: 400 classifier
  fitting and 100 scoring rows per fold, with ten scoring images per class.
  All 26 feature sources use exactly the same folds.
- Fit one normalized multiclass ridge head per source/fold with alpha **.1**,
  fitting-only mean/std and an unpenalized intercept. No hyperparameter search,
  classifier selection, alternative fold seed or extra encoder run.
- Report aggregate accuracy, confusion matrices, fold counts, descriptive
  encoder-seed means/sample standard deviations, feature geometry and paired
  recoveries/regressions. Include quadrants versus mean, shuffled quadrants,
  repeated mean, and matching random encoders for every mode.

There are 130 fixed classifier fits (26 sources × five folds). Each head is
saved against its exact disjoint fitting/scoring subsets and verified to
reproduce its scores. A no-fit replay must load all saved heads and reproduce
the aggregate report. Missing or changed evidence cannot silently refit or
replace an existing report.

## Provenance and limits

The accompanying JSON binds original source artifacts, checkpoints, fitting
order, fold assignment digest, runtime, all local Python source files and the
fixed recipe. Original experiment files and their exposure ledger are retained.
The earlier 3,246 observed validation rows and 1,754 unused rows do not change.

These are **exploratory cross-fitted classifier scores on encoder-training
images**. Although each scoring row is excluded from its classifier's fit, the
frozen encoder may have pretrained on that image. These results are not an
independent representation evaluation or confirmation of an accuracy gain.
The matched random controls and increased classifier dimensions must accompany
any comparison. Do not select a new experiment from old confirmation scores.
A follow-up would need its own prospective protocol and then-current exposure
history before new held-out access.

Checksums detect accidental differences, not intentional forgery. Raw images,
row-level features/labels, fitted classifier states, encoder weights and local
caches stay in ignored directories. Public output will contain aggregates and
provenance hashes only. Use bounded local CPU compute; no paid APIs or cloud.
