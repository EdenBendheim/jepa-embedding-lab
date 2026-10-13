# Fixed classifier confirmation — declared October 12, 2026

## Question

Do the already-selected pooling/mask classifiers retain their relative results
on examples never scored by the previous declared pilots? This scores fixed
classifiers; it changes neither model training nor classifier fitting.

## Fixed inputs before observing new scores

- Same official CIFAR-10 binary files and locked v1 manifest, SHA-256
  `0333e4c2400a89a694e5b2f3b51ff39482e3cdf9397c5e6969fc9995ad1cec44`.
- Preserve all 500 fitted training examples from October 9 selection
  `e0c4645e619917b1f001ea3382961906349995f8df8c9e0d3307569ef73e0e78`.
- Exclude every validation index in both the October 7 100/50 pilot and the
  October 9 500/200 pilot. Later encoder-age/seed/pooling/mask runs used the
  latter same selection; they add no new previously evaluated validation rows.
- Select **500 fresh validation examples, 50 per class**, using confirmation
  selector seed **20261012**, SHA-256 ordering and the exclusions above.
  These remain within the original 5,000-image validation partition.
  Do not extract or evaluate official test images.
- Freeze the 223-step default checkpoint
  `9943decda69020dc6e96174f5d5c6587390d207f000cbe6ca1255da731097b9e`
  and the already completed, image-order-matched 223-step 4x4 checkpoint.
- Six declared feature arms: default learned mean, default learned mean_std,
  learned 4x4-mask mean, random seed-29 mean, random seed-29 mean_std, and pixels.
  Their dimensions are respectively 32, 64, 32, 32, 64 and 3,072.
- Reuse the **exact selected states** from October 11's pooling/mask reports:
  AdamW classifier seeds **29, 31, 37**, plus the one deterministic selected
  ridge classifier in each arm. Total: **24 classifier evaluations**.
  The pixel states come from the pooling report once and are shared conceptually
  by all encoder comparisons. No classifier is retrained or retuned.
- Keep the original training-only mean/std and original settings in every
  classifier. Reproduce original train/tuning scores and verify source hashes
  before scoring fresh rows. Reject different weights, pooling, encoder choice,
  normalization, examples, class labels or declared source reports.
- Local CPU only, one thread, feature-extraction batch 16, no paid API/cloud.

## Known results informing this declared comparison

October 11's reused-tuning AdamW means: default learned mean 26.67%, default
learned mean_std 28.67%, learned 4x4 mean 27.50%, matched random mean 25.17%,
matched random mean_std 33.83%, pixels 28.00%. These are already known and
influenced which variants are being checked; the new rows have not been scored
before this protocol is committed. The max arm is not selected for this check.

## Report and limits

Publish all six arms: original tuning accuracy, confirmation per-class counts
and confusion matrices, AdamW mean/sample standard deviation, ridge scores,
selection/model/state/source fingerprints, no-refit flags and evaluated count.
Do not select a new winner, settings or publication claim from confirmation.
Raw images, features and weights remain ignored and local.

These are fresh examples relative to the recorded earlier pilots, rather than
an official test result. They still come from one dataset and the same original
validation partition, with one learned encoder seed per mask setting. The
mean_std dimensionality and the changed mask-token budgets remain confounds.
Classifier-seed spread is not encoder/data uncertainty. After inspection, these
rows are no longer an untouched source for future confirmation of new tuning.
