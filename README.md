# JEPA Embedding Lab

A small, reproducible representation-learning experiment: predict target embeddings from visible context, then test whether custom embeddings help a downstream task.

**Status — October 7, 2026:** the compact JEPA model, EMA updates, fixed CIFAR-10 input/splits, and resumable public-data training work. Frozen full-image features, seeded random-encoder features, and raw-pixel baseline features now extract on matching examples. A 20+3-step training pilot and bounded real-image feature checks completed. Probe fitting, sustained training, and classification comparison results remain pending; execution checks do not establish useful learned representations.

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
- A resumable public-data CPU trainer with atomic checkpoints, matching manifest/model/settings/runtime checks, optimizer and sampling/PyTorch RNG restoration, and split-versus-continuous reproducibility checks.
- Frozen context/target full-image embeddings with mean pooling across all 64 patches, seeded random-encoder and raw-pixel feature baselines, and atomic feature artifacts bound to exact example indices, data identity, and checkpoint bytes.

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

See [the experiment protocol](experiments/CIFAR10.md) and [version-one provenance](experiments/cifar10-v1.json). No CIFAR-10 classification accuracy has been measured yet.

## Resume a public-data training pilot

```sh
export PYTHONPATH=src
.venv/bin/python -m jepa_lab.cifar10_train data/cifar-10-batches-bin \
  --manifest runs/cifar10-manifest.json --steps 20 \
  --checkpoint checkpoints/cifar10-pilot.pt --output runs/cifar10-pilot.json
.venv/bin/python -m jepa_lab.cifar10_train data/cifar-10-batches-bin \
  --manifest runs/cifar10-manifest.json --steps 3 \
  --resume checkpoints/cifar10-pilot.pt --checkpoint checkpoints/cifar10-pilot.pt \
  --output runs/cifar10-pilot-resumed.json
```

`--steps` is the number of **additional** optimizer steps, capped at 1,000 per invocation with batch size 1–32. Defaults are batch four, learning rate 0.001, EMA momentum 0.99, the manifest seed, and one CPU thread. This is a bounded local trainer, with no download, API request, paid compute, or automatic prolonged run. Reports record the start/completed step counts, exact sampled training indices, loss, gradient norm, and embedding variance/covariance for that invocation.

Checkpoint version one binds state to the verified manifest fingerprint, model configuration, seed, batch size, learning rate, momentum, PyTorch version, and CPU thread count. Resume rejects mismatches before sampling. Model, optimizer, sample RNG, and PyTorch RNG are restored; mask seeds advance from the saved global step. Tests require a resumed run to match uninterrupted diagnostics, weights, optimizer, and RNG state in the same environment. This does not guarantee identical results across platforms or library versions; incompatible recorded runtimes are rejected.

Checkpoints load with `weights_only=True` on CPU and are written via temporary file plus atomic replacement. A failed save leaves the previous checkpoint intact. Saving occurs at the end of a successful invocation, so an interrupted invocation can lose its unsaved steps. Data/weights/run metrics remain ignored. The 20+3-step real-image pilot had finite losses/gradients but is too small to establish useful representations. Frozen feature extraction below works; validation-tuned probes and measured raw-pixel/random-encoder/pretrained comparisons remain the next milestone.

## Extract frozen image features and baseline features

```sh
export PYTHONPATH=src
.venv/bin/python -m jepa_lab.embeddings data/cifar-10-batches-bin \
  --manifest runs/cifar10-manifest.json --checkpoint checkpoints/cifar10-pilot.pt \
  --partition train --limit 64 --output runs/features-checkpoint-train.pt
.venv/bin/python -m jepa_lab.embeddings data/cifar-10-batches-bin \
  --manifest runs/cifar10-manifest.json --representation random --seed 29 \
  --partition train --limit 64 --output runs/features-random-train.pt
.venv/bin/python -m jepa_lab.embeddings data/cifar-10-batches-bin \
  --manifest runs/cifar10-manifest.json --representation pixels \
  --partition train --limit 64 --output runs/features-pixels-train.pt
```

The context encoder is the default; `--encoder target` extracts from the EMA teacher. Both process all 64 patches without a mask or predictor, then mean-pool token embeddings. Inference disables gradients and leaves weights unchanged; the API restores the model's prior training mode. Random baseline initialization preserves the caller's CPU RNG and records its seed/architecture. The current default random architecture matches the 32-dimensional pilot. For other model configurations, pass the matching `ModelConfig` to the extraction API before comparing; a future comparison runner should enforce architecture equality.

Raw-pixel features flatten NCHW values using the same fixed `x / 127.5 - 1` scaling, producing 3,072 dimensions. Labels accompany feature artifacts for future probes but do not enter an encoder. Every representation uses the same first `--limit` indices from the selected fixed partition (default 64, cap 1,000, batch size 1–32). These bounded subsets are not guaranteed class-balanced and are not the complete experiment. Validation uses original training-file indices; test indices remain in the official test namespace. Extraction never fits normalization or a classifier on held-out examples.

Each local `.pt` artifact contains features, labels, exact indices/namespace, manifest identity, pooling/scaling/configuration, extraction settings, and a feature fingerprint. Checkpoint extraction also records the training step/settings and the SHA-256 of the exact bytes loaded. Mismatched manifests/checkpoints fail before image sampling or output replacement. Features and raw-pixel artifacts remain in ignored `runs/`; do not redistribute the images or commit artifacts. Reports contain metadata/diagnostics, not feature arrays. The real-image check extracted the same 64 training images in all three representations and another 64 validation images from the checkpoint, with finite features throughout. No probe or classification accuracy has been measured.

## Implemented model contract

1. A context encoder receives only visible patches and their positions.
2. A stop-gradient target encoder encodes the full input; target positions select prediction targets.
3. A predictor combines context embeddings with target positions to predict target representations.
4. An exponential moving average updates the target encoder from the context encoder.
5. Log loss **and** representation variance/covariance to diagnose collapse; evaluate frozen representations on a held-out downstream task.

The model uses PyTorch through an explicit optional training dependency. Baselines and ablations will share fixed splits, seeds, compute accounting, and an evaluation protocol. Improved latent prediction loss is not itself proof of useful embeddings.

For a fire-specific dataset, split by distinct fire event, time, and geography before sampling windows. A fire progression prediction head would be a separate experiment; latent prediction alone does not output fire polygons.

See [PLAN.md](PLAN.md) for the staged experiment and [DEVLOG.md](DEVLOG.md) for completed work.

## Choose a shared balanced pilot

```sh
export PYTHONPATH=src
.venv/bin/python -m jepa_lab.selection data/cifar-10-batches-bin \
  --manifest runs/cifar10-manifest.json --output runs/pilot-selection.json \
  --train-per-class 10 --validation-per-class 5
.venv/bin/python -m jepa_lab.embeddings data/cifar-10-batches-bin \
  --manifest runs/cifar10-manifest.json --selection runs/pilot-selection.json \
  --representation pixels --partition train --output runs/pilot-pixels-train.pt
```

The pilot uses deterministic within-class SHA-256 ordering inside the already locked train/validation partitions. Defaults select 100 training and 50 validation examples, balanced across all ten classes. The selection records exact indices, seed, per-class counts and a fingerprint bound to the full dataset manifest. It never includes test. Counts are bounded at 100 per class.

All feature representations can pass the same `--selection`; extraction rebuilds it against current labels and data identity before sampling. It uses the complete selected partition and rejects `--limit` or test extraction with a pilot selection. The earlier first-index wiring-check path remains available without `--selection`. Balanced pilot results will describe this small subset only; probe evaluation is still pending.

## Verify a feature artifact before evaluation

```sh
.venv/bin/python -m jepa_lab.features data/cifar-10-batches-bin runs/pilot-pixels-train.pt \
  --manifest runs/cifar10-manifest.json --selection runs/pilot-selection.json --partition train
```

The loader uses `weights_only=True` on bounded local files and verifies exact selection/manifest identity, train/validation namespace, indices, official labels, feature shape/dtype/finiteness/content hash, normalization, encoder architecture, checkpoint/random provenance, and recorded PyTorch runtime. Reports omit arrays and retain the hash of the exact artifact bytes loaded. A checksum checks consistency; it does not authenticate the producer or prove that an encoder was trained well. Use artifacts generated locally by the extraction command. Test artifacts are excluded from this pilot evaluation loader.
