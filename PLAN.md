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
- [x] Persist features/labels with exact partition indices, manifest/checkpoint identity, and configuration.
- [ ] Compare two or three embedding widths and position/input-feature choices.
- [ ] Hold the data split, downstream task, training budget, and evaluation code fixed.
- [ ] Fit a linear probe or small task head to frozen representations.
- [ ] Run multiple seeds where compute allows; report uncertainty and collapse/failure cases.
- [ ] Publish configurations, provenance, plots, a results table, and a small embedding explorer.

## Next session

Add a linear-probe evaluation path over the frozen feature artifacts, validating partition identity and fitting on training examples only. Choose hyperparameters using validation, compare raw-pixel/random-encoder/checkpoint representations on the same selected examples, and reserve test for final reporting. The feature extractor currently handles bounded first-index subsets, not the full 45k/5k protocol; add a shared balanced pilot-selection manifest or full extraction before making quality comparisons. Then choose a bounded sustained training budget and an appropriate pretrained comparison. Downstream accuracy is still unmeasured. Record actual experiments and checks in DEVLOG; measured quality results enter the résumé only after evaluation. The daily development run implements, checks, commits, and pushes an update; small useful steps are sufficient.
