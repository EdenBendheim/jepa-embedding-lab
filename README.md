# JEPA Embedding Lab

A small, reproducible representation-learning experiment: predict target embeddings from visible context, then test whether custom embeddings help a downstream task.

**Status — October 3, 2026:** patch preparation, reproducible masking, a compact PyTorch JEPA model with EMA updates, synthetic checkpoint resume, and an RGB CIFAR-10 input/split pipeline work. A three-step public-data CPU smoke run completed. A sustained public-data trainer, downstream embedding evaluation, and comparison results remain pending; smoke runs do not establish useful learned representations.

## Why this experiment

I want to explore the relationship between the input representation, masking geometry, and what an embedding retains. The first public experiment uses [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) and frozen-embedding image classification. Its small RGB images make bounded local iteration practical. Satellite/geospatial imagery remains a later direction because it connects to my research; no unreleased research data is used here.

The design reference is [I-JEPA](https://arxiv.org/abs/2301.08243) and its [official implementation](https://github.com/facebookresearch/ijepa). The current single-target-block sampler is a simplified experiment foundation, not a reproduction of the official multi-block training recipe.

## Working now

- Non-overlapping patch extraction with explicit positions and strict shape/finite-value validation.
- Seeded rectangular target blocks with context sampled outside the target.
- Separate context, target, and unused indices: the context encoder must never receive target pixels.
- Reproducible CLI mask inspection and tests covering partition integrity, preserved pixels, invalid shapes, and random-seed isolation.
- Context/target Transformer encoders and a masked-target predictor with learned patch positions. Context gathers visible raw patches before encoding; target encoding is frozen and uses no gradients.
- EMA target updates, training loss plus embedding standard-deviation/covariance diagnostics, and CPU checkpoint resume including optimizer and random-generator state.
- Official CIFAR-10 binary ingestion without pickle, a fixed RGB patch adapter, dataset fingerprints, and disjoint stratified train/validation indices while retaining the official test partition.
- A bounded RGB training smoke command that validates its manifest, samples training images only, and records loss/variance diagnostics without claiming downstream accuracy.

## Run the mask demo

Python 3.11+, no runtime dependencies:

```sh
export PYTHONPATH=src
python3 -m jepa_lab.demo --seed 29
python3 -m unittest discover -s tests -p test_foundation.py -v
```

`C` marks visible context, `T` marks prediction targets, and `.` marks unused patches. The command also prints indices and patch dimensions for inspection. All demo intensities are generated locally.

## Run the model and synthetic training check

The training extra pins PyTorch 2.8.0; the lightweight mask demo remains dependency-free. On Linux, install the CPU PyTorch wheel from its official index first if CPU-only training is intended.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[training]'
export PYTHONPATH=src
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m jepa_lab.train --steps 20 --checkpoint checkpoints/smoke.pt --output runs/smoke.json
.venv/bin/python -m jepa_lab.train --steps 3 --resume checkpoints/smoke.pt
```

The default model uses an 8x8 patch grid, four values per patch, width 32, two encoder blocks, and one predictor block. The CLI runs on CPU with one thread and locally generated spatial patterns. Every sample receives a deterministic mask. Checkpoints include model/configuration, optimizer, step count, and data/PyTorch RNG state; resume requires matching seed, batch size, learning rate, and momentum. `--steps` means additional steps. Tests compare a resumed run with an uninterrupted run in the same environment. Reproducibility across hardware or library versions is not guaranteed.

Checkpoints and run metrics are ignored by Git. No weights or private research inputs are committed. Small synthetic loss changes and nonzero embedding variance are execution diagnostics, not evidence of downstream accuracy or immunity to collapse. The public dataset/protocol below is selected; quality claims still require downstream evaluation.

## Prepare the first public dataset

Download the **binary version** from the [official CIFAR-10 page](https://www.cs.toronto.edu/~kriz/cifar.html). Its published archive MD5 is `c32a1d4ab5d03f1284b67883e8d87530`; verify the download and unpack it beneath ignored `data/`. Images are not redistributed in this repository. The loader requires the six official `.bin` files, validates record sizes/labels, and hashes their contents. No network request occurs inside the loader.

```sh
export PYTHONPATH=src
.venv/bin/python -m jepa_lab.cifar10 data/cifar-10-batches-bin \
  --manifest runs/cifar10-manifest.json
.venv/bin/python -m jepa_lab.cifar10_smoke data/cifar-10-batches-bin \
  --manifest runs/cifar10-manifest.json --steps 3 \
  --output runs/cifar10-adapter-smoke.json
```

The version-one protocol uses 45,000 official training images for JEPA/probe fitting, 5,000 for validation (500 per class), and all 10,000 official test images for final evaluation. SHA-256 ordering of original training indices with seed `20261003` makes the split independent of global RNG state or Python's shuffle implementation. The manifest preserves official split names, exact indices, batch hashes, transform configuration, and its own fingerprint. Training and test indices have separate namespaces; numeric overlap alone does not mean shared examples. The smoke command rebuilds the manifest from local files and rejects stale or edited splits.

The adapter accepts uint8 NCHW RGB images and applies the fixed `x / 127.5 - 1` transform. A 32x32 image becomes an 8x8 grid of 4x4 RGB patches (`patch_dim=48`); patches follow row-major grid order with channel-major pixels. Class labels are used for stratification and future probe evaluation, not the JEPA training objective. The three-step smoke run uses only training images, one CPU thread, and no paid API. Its command is capped at 50 steps / batch size 32; it is not the resumable long-run trainer.

See [the experiment protocol](experiments/CIFAR10.md) and [version-one provenance](experiments/cifar10-v1.json). Frozen-embedding extraction, probes, and baseline measurements are the next steps. No CIFAR-10 classification accuracy has been measured yet.

## Implemented model contract

1. A context encoder receives only visible patches and their positions.
2. A stop-gradient target encoder encodes the full input; target positions select prediction targets.
3. A predictor combines context embeddings with target positions to predict target representations.
4. An exponential moving average updates the target encoder from the context encoder.
5. Log loss **and** representation variance/covariance to diagnose collapse; evaluate frozen representations on a held-out downstream task.

The model uses PyTorch through an explicit optional training dependency. Baselines and ablations will share fixed splits, seeds, compute accounting, and an evaluation protocol. Improved latent prediction loss is not itself proof of useful embeddings.

For a fire-specific dataset, split by distinct fire event, time, and geography before sampling windows. A fire progression prediction head would be a separate experiment; latent prediction alone does not output fire polygons.

See [PLAN.md](PLAN.md) for the staged experiment and [DEVLOG.md](DEVLOG.md) for completed work.
