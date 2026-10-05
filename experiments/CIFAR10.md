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

## Planned comparisons (not yet measured)

The primary metric will be top-1 classification accuracy, with per-class outcomes and failure examples. Compare learned JEPA embeddings against a frozen random encoder of the same architecture and a raw-pixel linear classifier. Use identical probe-fitting data, validation selection, and evaluation examples. Add an appropriate pretrained visual encoder as a separate external-pretraining baseline, documenting its source/pretraining data and input adaptation.

Full-image embedding extraction, linear-probe fitting, and these measurements are not implemented yet. Select a bounded pilot budget before sustained training; retain shared step/image budgets, seeds and settings for later ablations. Report training cost and embedding variance/covariance alongside downstream results. Lower JEPA loss alone does not establish better embeddings.

## Completed check

October 3: verified the official archive, produced the version-one manifest, and ran three CPU JEPA updates from four RGB training images per step. The smoke command uses no held-out images and rejects an edited/stale manifest.

October 4: implemented manifest-bound public-data checkpoints and ran 20 CPU steps (batch four), followed by three additional steps from the saved checkpoint. Unit tests compare continuous and resumed sampling, diagnostics, weights, optimizer, and RNG states. The manifest fingerprint is unchanged. Checkpoints save at successful invocation boundaries and require matching model/settings/PyTorch version/thread count. These checks establish training and resume behavior; they do not provide classification accuracy or validate representation quality.
