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
- [ ] Extend resumable training/checkpoints to public data, preserving manifest identity and sample RNG state.

## Compact model — begin October 1

- [x] Add PyTorch context encoder, stop-gradient target encoder, and predictor.
- [x] Implement EMA updates and checkpoint/config persistence.
- [x] Verify gradient boundaries, mask gather shapes, and EMA behavior.
- [x] Run a small smoke experiment; log embedding variance and training loss.
- [ ] Establish a frozen pretrained baseline appropriate to the selected modality.

## Custom embeddings and useful comparisons

- [ ] Compare two or three embedding widths and position/input-feature choices.
- [ ] Hold the data split, downstream task, training budget, and evaluation code fixed.
- [ ] Fit a linear probe or small task head to frozen representations.
- [ ] Run multiple seeds where compute allows; report uncertainty and collapse/failure cases.
- [ ] Publish configurations, provenance, plots, a results table, and a small embedding explorer.

## Next session

Add a resumable CIFAR-10 training path that binds every checkpoint to the fixed data-manifest fingerprint and preserves sampling state. Add full-image frozen embeddings and a linear probe, with validation-only hyperparameter selection and raw-pixel/random-encoder baselines on the same splits. Run a bounded pilot before choosing a sustained compute budget, then add an appropriate pretrained comparison. Only the three-step real-data wiring check has run; downstream accuracy is still unmeasured. Record actual experiments and checks in DEVLOG; measured quality results enter the résumé only after evaluation. The daily development run implements, checks, commits, and pushes an update; small useful steps are sufficient.
