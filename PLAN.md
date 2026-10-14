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
- [ ] Compare two or three embedding widths and position/input-feature choices.
- [ ] Hold the data split, downstream task, training budget, and evaluation code fixed.
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

The [October 14 longer-budget experiment](experiments/BUDGET-2026-10-14.md) is
complete: two target masks, three encoder seeds each, 2,223 steps, matched image
order, 72 reproduced selected states and 500 new confirmation examples excluding
746 observed rows. Fresh AdamW: 2x2 learned 24.24%, 4x4 learned 23.11%, matching
random 24.11%, pixels 25.73%; pixel ridge 30.20%. Longer training alone did not
establish an advantage. Periodic recovery and complete-journal/cache verification
work; images, weights and per-image records remain ignored.

Next: predeclare a controlled model/input change (width and position/input choices
remain planned), hold the training/evaluation budget fixed, and reserve a new
untuned subset before confirmation. All prior confirmation rows are now observed.
Retain the official test set. Training-dynamics plots and verified aggregate points now work; an embedding explorer remains planned;
quality claims still require independent evidence.
