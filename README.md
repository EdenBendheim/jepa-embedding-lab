# JEPA Embedding Lab

A small, reproducible representation-learning experiment: predict target embeddings from visible context, then test whether custom embeddings help a downstream task.

**Status — October 2, 2026:** patch preparation, reproducible masking, and a compact PyTorch JEPA model with EMA updates, synthetic CPU training, diagnostics, and resumable checkpoints work. Public-dataset training and downstream embedding evaluation remain pending. Synthetic smoke training does not establish useful learned representations.

## Why this experiment

I want to explore the relationship between the input representation, masking geometry, and what an embedding retains. Satellite/geospatial imagery is a candidate because it connects to my research, but the final dataset and downstream task are still open. The initial image adapter is intentionally small and uses only synthetic single-channel images.

The design reference is [I-JEPA](https://arxiv.org/abs/2301.08243) and its [official implementation](https://github.com/facebookresearch/ijepa). The current single-target-block sampler is a simplified experiment foundation, not a reproduction of the official multi-block training recipe.

## Working now

- Non-overlapping patch extraction with explicit positions and strict shape/finite-value validation.
- Seeded rectangular target blocks with context sampled outside the target.
- Separate context, target, and unused indices: the context encoder must never receive target pixels.
- Reproducible CLI mask inspection and tests covering partition integrity, preserved pixels, invalid shapes, and random-seed isolation.
- Context/target Transformer encoders and a masked-target predictor with learned patch positions. Context gathers visible raw patches before encoding; target encoding is frozen and uses no gradients.
- EMA target updates, training loss plus embedding standard-deviation/covariance diagnostics, and CPU checkpoint resume including optimizer and random-generator state.

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

Checkpoints and run metrics are ignored by Git. No weights or private research inputs are committed. Small synthetic loss changes and nonzero embedding variance are execution diagnostics, not evidence of downstream accuracy or immunity to collapse. Select a public dataset and fixed evaluation protocol before claiming representation quality.

## Implemented model contract

1. A context encoder receives only visible patches and their positions.
2. A stop-gradient target encoder encodes the full input; target positions select prediction targets.
3. A predictor combines context embeddings with target positions to predict target representations.
4. An exponential moving average updates the target encoder from the context encoder.
5. Log loss **and** representation variance/covariance to diagnose collapse; evaluate frozen representations on a held-out downstream task.

The model uses PyTorch through an explicit optional training dependency. Baselines and ablations will share fixed splits, seeds, compute accounting, and an evaluation protocol. Improved latent prediction loss is not itself proof of useful embeddings.

For a fire-specific dataset, split by distinct fire event, time, and geography before sampling windows. A fire progression prediction head would be a separate experiment; latent prediction alone does not output fire polygons.

See [PLAN.md](PLAN.md) for the staged experiment and [DEVLOG.md](DEVLOG.md) for completed work.
