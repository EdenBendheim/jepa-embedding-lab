# Experiment plan

Started September 29, 2026; begin the model implementation October 1. Image/geospatial scope is provisional until a dataset and downstream task are selected.

## Foundation

- [x] Synthetic single-channel image patchification with shape and finite-value checks.
- [x] Seeded target-block/context masking; target/context separation and index tests.
- [x] CLI inspection of patch and mask contracts.
- [ ] Select one small public dataset and a fixed downstream task.
- [ ] Write train/evaluation manifests, with event-level splits if using fire data.
- [ ] Add multi-channel input support if needed by the selected dataset.

## Compact model — begin October 1

- [ ] Add PyTorch context encoder, stop-gradient target encoder, and predictor.
- [ ] Implement EMA updates and checkpoint/config persistence.
- [ ] Verify gradient boundaries, mask gather shapes, and EMA behavior.
- [ ] Run a small smoke experiment; log embedding variance and training loss.
- [ ] Establish a frozen pretrained baseline appropriate to the selected modality.

## Custom embeddings and useful comparisons

- [ ] Compare two or three embedding widths and position/input-feature choices.
- [ ] Hold the data split, downstream task, training budget, and evaluation code fixed.
- [ ] Fit a linear probe or small task head to frozen representations.
- [ ] Run multiple seeds where compute allows; report uncertainty and collapse/failure cases.
- [ ] Publish configurations, provenance, plots, a results table, and a small embedding explorer.

## Next session

Choose the dataset/task and write the split manifest before training. On October 1, start a compact encoder/predictor implementation. Record actual experiments and checks in DEVLOG; measured results enter the résumé only after the experiments run.
