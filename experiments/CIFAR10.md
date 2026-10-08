# CIFAR-10 experiment protocol, version 1

## Dataset and task

Use the binary version of [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html): 32x32 RGB images, ten classes, official 50,000-image training and 10,000-image test sets. Cite Alex Krizhevsky, *Learning Multiple Layers of Features from Tiny Images* (2009), as requested by the dataset authors. Raw images remain local in ignored `data/`; this repository distributes code and provenance metadata.

The downstream task is ten-class classification from a frozen, full-image embedding. Predicting masked target embeddings is the pretraining objective, not the downstream classification score. This compact model is not a reproduction of the official I-JEPA recipe.

## Locked split and input contract

- Original training indices are ordered within each class by SHA-256 of `cifar10-v1:20261003:<index>`. The first 500 per class enter validation; the remaining 4,500 enter training.
- Training indices apply only to the concatenation of binary batches 1-5. Test indices apply only to `test_batch.bin`. Preserve those namespaces in every dataset/subset reference.
- Labels determine stratification and later probe targets; the JEPA loss never receives them.
- Fit the encoder and probe on training data only. Choose any hyperparameters/checkpoints on validation. Reserve the official test set for final reporting; do not tune against its scores.
- Input layout is NCHW uint8 RGB. Scale with `x / 127.5 - 1`, without fitting statistics to the full dataset. Use 4x4 patches: 8x8 positions, 48 values per patch.
- Batch SHA-256 hashes, exact partition indices, seed, algorithm and input configuration are recorded in the generated manifest. Rebuild and compare that manifest before using a checkpoint or running a comparison.

## Comparison protocol

The primary metric will be top-1 classification accuracy, with per-class outcomes and failure examples. Compare learned JEPA embeddings against a frozen random encoder of the same architecture and a raw-pixel linear classifier. Use identical probe-fitting data, validation selection, and evaluation examples. Add an appropriate pretrained visual encoder as a separate external-pretraining baseline, documenting its source/pretraining data and input adaptation.

Full-image feature extraction, balanced pilot selection, validated artifacts and linear probes are implemented. The [October 7 exploratory pilot](PILOT-2026-10-07.md) measures validation-selected checkpoint/random/pixel scores; official test and pretrained comparisons remain pending. Select a bounded pilot budget before sustained training; retain shared step/image budgets, seeds and settings for later ablations. Report training cost and embedding variance/covariance alongside downstream results. Lower JEPA loss alone does not establish better embeddings.

## Frozen representation contract

- Context or target encoder receives all 64 RGB patches; mean-pool the final patch-token embeddings. Do not run the masked predictor for full-image evaluation.
- Disable gradients and keep checkpoint weights fixed. Record encoder choice, pooling, architecture, training step/settings, and a hash of the exact checkpoint bytes loaded.
- A seeded random encoder of the same architecture and flattened normalized NCHW pixels supply baseline features. They use the same selected indices and labels as checkpoint features; no normalization statistics are fitted.
- Feature artifacts preserve manifest fingerprint, official split namespace, exact indices, feature hash, configuration, and extraction settings. Labels are used only for split stratification and downstream classification probes.
- The current CLI extracts the first 1–1,000 fixed partition indices for bounded wiring checks. These subsets are not guaranteed class-balanced. The shared balanced pilot manifest is now implemented for downstream comparisons; do not silently compare different subsets/architectures.

## Completed check

October 3: verified the official archive, produced the version-one manifest, and ran three CPU JEPA updates from four RGB training images per step. The smoke command uses no held-out images and rejects an edited/stale manifest.

October 4: implemented manifest-bound public-data checkpoints and ran 20 CPU steps (batch four), followed by three additional steps from the saved checkpoint. Unit tests compare continuous and resumed sampling, diagnostics, weights, optimizer, and RNG states. The manifest fingerprint is unchanged. Checkpoints save at successful invocation boundaries and require matching model/settings/PyTorch version/thread count. These checks establish training and resume behavior; they do not provide classification accuracy or validate representation quality.

October 5: extracted checkpoint/random-encoder/pixel features on the same 64 real training examples, and checkpoint features on 64 validation examples. Features were finite and artifacts retain exact example/weight identity. No classifier was fitted or test score measured; representation quality remains unmeasured.

October 7: completed a matched balanced pilot with 100 training and 50 validation examples, eight shared probe candidates, and the existing 23-step context checkpoint. Selected validation accuracy: checkpoint 24%, random 30%, pixels 28%. All 43 tests passed. An initial grid exposed pixel overfitting and guided the shared grid expansion. These are exploratory validation-selected scores; test remains reserved, no reliable representation ranking or improvement is established, and multiple seeds/longer training remain pending.
