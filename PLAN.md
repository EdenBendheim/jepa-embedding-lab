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
- [x] Fit bounded linear probes with training-only normalization and validation-only hyperparameter selection.
- [x] Run a matched real-image checkpoint/random/pixel comparison and report its exploratory pilot limitations.
- [ ] Run multiple seeds where compute allows; report uncertainty and collapse/failure cases.
- [ ] Publish configurations, provenance, plots, a results table, and a small embedding explorer.

## Next session

The balanced 100-train/50-validation pilot now supports verified frozen artifacts, matched architectures/examples, training-only normalization, common probe grids/budgets, and validation-only selection. Its 23-step checkpoint scored 24%, random encoder 30%, and pixels 28%; this small exploratory result does not establish learned-feature improvement. Next: predeclare a larger balanced selection and regularization grid, run a bounded longer JEPA training budget on training data only, compare several seeds/checkpoints with the same downstream protocol, and diagnose pixel overfitting. Keep test reserved for a final fixed evaluation. Add an appropriate pretrained comparison and probe-state persistence before full evaluation. Sustained quality results enter the résumé only when supported. Record actual work/checks in DEVLOG; small coherent daily steps are sufficient.
