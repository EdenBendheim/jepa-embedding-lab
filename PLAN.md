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
- [ ] Compare position/input-feature choices at a fixed width.
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
- [ ] Publish configurations, provenance, plots, a results table, and a small embedding explorer.

## Next session

The [October 15 controlled-width experiment](experiments/WIDTH-2026-10-15.md)
is complete: 32/64/96 widths, three seeds each, 2,223 steps, same actual image
order and exact-initialization random controls. Reproduced 84 linear-state
artifacts and 19 neighbor indices; scored 500 new confirmation rows excluding
1,246 declared earlier rows. Fresh learned AdamW means: 28.16%/27.80%/25.47%.
The 32-D AdamW head beats its random/pixel controls here, while ridge/kNN still
favor pixels. Wider did not consistently help; no general advantage or best
width is established. Resource bounds, exact parameter accounting and custom
architecture CLIs work. All weights/features/indices/journals remain local.

Next: predeclare one position/input-feature change at a fixed width, keep the
training/evaluation recipe fixed, and reserve different untuned rows before
confirmation. The current ancestry depth is bounded; retain the history of all
scored selections when designing later reservations. Official test, embedding
explorer and further quality evidence remain unfinished.
