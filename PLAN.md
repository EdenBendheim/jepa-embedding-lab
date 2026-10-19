# Experiment plan

Started September 29, 2026; model implementation completed its first working synthetic training/checkpoint milestone October 2. CIFAR-10 RGB images and frozen-embedding classification were selected October 3 as the first public-data experiment. Geospatial experiments remain a later direction.

## Foundation

- [x] Synthetic single-channel image patchification with shape and finite-value checks.
- [x] Seeded target-block/context masking; target/context separation and index tests.
- [x] CLI inspection of patch and mask contracts.
- [x] Select CIFAR-10 and frozen-embedding ten-class classification as the first public experiment.
- [x] Write fixed train/validation/test manifests while preserving the official test partition.
- [x] Add RGB input adaptation with tested channel/patch order and a fixed transform.
- [x] Run a bounded public-data training smoke check using training images only.
- [x] Extend resumable training/checkpoints to public data, preserving manifest identity and sample RNG state.
- [x] Run a bounded public-data pilot and resume it while preserving training state.

## Compact model — begin October 1

- [x] Add PyTorch context encoder, stop-gradient target encoder, and predictor.
- [x] Implement EMA updates and checkpoint/config persistence.
- [x] Verify gradient boundaries, mask gather shapes, and EMA behavior.
- [x] Run a small smoke experiment; log embedding variance and training loss.
- [x] Establish a frozen pretrained baseline appropriate to the selected modality.

## Custom embeddings and useful comparisons

- [x] Extract frozen full-image context/target embeddings with a documented pooling rule.
- [x] Add seeded random-encoder and raw-pixel feature representations on identical fixed examples.
- [x] Verify feature artifacts against selected examples/labels, numerical integrity, architecture, and weight/runtime provenance before evaluation.
- [x] Add a shared class-balanced pilot manifest and exact-subset feature extraction without test selection.
- [x] Persist features/labels with exact partition indices, manifest/checkpoint identity, and configuration.
- [x] Compare three embedding widths at the same update/data/evaluation budget.
- [x] Compare learned versus fixed spatial positions at width 32 under matched initialization, budgets and fresh rows.
- [ ] Predeclare and compare a position-scale/input-feature alternative with retained exposure history.
- [x] Hold the data split, downstream task, update budget, and evaluation code fixed.
- [x] Add frozen cosine-neighbor selection, saved indices and untuned confirmation.
- [x] Persist selected probe weights/normalization with source identity and reproduced scores.
- [x] Add a matched regularized ridge head and a larger predeclared pilot.
- [x] Fit bounded linear probes with training-only normalization and validation-only hyperparameter selection.
- [x] Run a matched real-image checkpoint/random/pixel comparison and report its exploratory pilot limitations.
- [x] Repeat classifier initialization on matched frozen features and label the limited uncertainty scope.
- [x] Repeat encoder-training/random-encoder seeds in a bounded pilot and distinguish descriptive seed variation from test/data uncertainty.
- [x] Compare fixed-checkpoint mean/max/mean_std with matching random pooling and shared pixel classifiers.
- [x] Inspect verified feature geometry and reproduce selected classifiers/paired class errors.
- [x] Implement resume-bound custom masks and run a declared 4x4-target pilot matched to actual historical image order.
- [x] Reserve balanced confirmation rows excluding all declared earlier tuning examples.
- [x] Score frozen selected AdamW/ridge classifiers on fresh examples without refitting or retuning.
- [x] Publish configurations, provenance, plots, a results table, and a small row-and-label embedding explorer.

## Next session

The [October 18 position comparison](experiments/POSITION-2026-10-18.md) is
complete: three new fixed-position runs and three historical learned-position
runs, 2,223 updates each with matched non-position initialization and actual
image order. On 500 fresh rows, trained fixed AdamW 29.04% versus learned-position
27.47%, but fixed random 28.93%. A general trained-feature advantage remains
unproven; pixel kNN is still higher. Reproduced 60 linear artifacts and 13 indices,
plus paired fresh recoveries/regressions, with no-refit cache replay.

Before another confirmation round, implement a source-verified flattened
exposure ledger: the current bounded nested representation rejects another
round. Retain all 2,246 now-observed validation rows rather than dropping a
parent to bypass the guard. Then predeclare a position-scale/input-feature
contrast at fixed width with a fresh reservation. Do not tune on these newly
observed scores or report absolute changes from older row sets as improvement.
Keep the official test reserved. The row/label neighborhood explorer already
works; a local-only thumbnail view remains optional. No raw dataset images,
features, model weights or row-level journals belong in public commits.
