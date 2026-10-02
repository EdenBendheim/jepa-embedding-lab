# Development log

## October 2, 2026

- Implemented compact Transformer context/target encoders, positional embeddings, and a masked-target predictor. The teacher is frozen and evaluated without gradients; an EMA update moves it toward the context encoder.
- Context encoding gathers raw visible patches before attention. Per-example masks are checked for bounds, uniqueness, and target/context overlap.
- Added bounded synthetic CPU training, loss and embedding standard-deviation/covariance diagnostics, and resumable model/optimizer/configuration/RNG checkpoints. Added a pinned optional PyTorch training dependency and CPU training checks to CI.
- Verification: 11 tests passed, including gradient boundaries, prediction invariance to hidden-pixel changes, exact EMA interpolation/endpoints, mask validation, seeded training, and split-versus-continuous checkpoint resume. Ran 20 synthetic steps and resumed for three more steps successfully; these are execution checks, not a public-data quality result.
- Interview explanation: "I built masked embedding prediction with a frozen EMA teacher, verified hidden pixels cannot reach the context encoder, and made the training state reproducibly resumable."
- Still pending: dataset/task selection, data splits, real-data training, pretrained comparison, downstream probes, and useful-embedding results. Checkpoints and metrics remain in ignored local directories.

## September 29, 2026

- Started the image experiment foundation with non-overlapping patch extraction and position metadata.
- Added deterministic target-block masks, visible context sampling, and a synthetic CLI inspection demo.
- Added tests for target/context separation, complete grid coverage, input validation, pixel preservation, and seed isolation.
- Model training and custom learned embeddings remain future work, beginning October 1.
- Verification: six unit tests passed; the seeded CLI demo generated 64 patches, 45 context patches, and four target patches. Added GitHub Actions checks for Python 3.11 and 3.13.
