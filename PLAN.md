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
- [ ] Establish a frozen pretrained baseline appropriate to the selected modality.

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
- [ ] Publish configurations, provenance, plots, a results table, and a small embedding explorer.

## Next session

Both October 10 protocol parts completed: controlled 23/223-step comparison and three encoder/random initialization seeds. Under matched 500/200 examples and grids, the later-age mean gained 1 pp in one trajectory. Across three 223-step encoder runs, AdamW learned/random/pixel means were 27.11%/25.67%/28.00%; ridge 25.67%/25.33%/30.00%. All 36 selected classifiers for the seed comparison reproduce, along with paired class errors. One favorable seed/head is not a reliable improvement; reused validation and short training remain limiting.

Next: inspect class confusions, mean pooling and mask geometry; select an appropriate frozen pretrained baseline; predeclare a tightly bounded pooling/masking ablation. Then extend training with explicit budgets and reserve the official test set for a final fixed protocol. Width/input-feature comparisons, plots and an embedding explorer remain pending. Résumé quality claims require stronger evidence.
