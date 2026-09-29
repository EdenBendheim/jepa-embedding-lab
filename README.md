# JEPA Embedding Lab

A small, reproducible representation-learning experiment: predict target embeddings from visible context, then test whether custom embeddings help a downstream task.

**Status — September 29, 2026:** patch preparation and reproducible context/target masking work. Encoders, predictor, training, and downstream evaluation are not implemented yet. Training implementation is scheduled to start October 1. This repository does not contain a trained JEPA model.

## Why this experiment

I want to explore the relationship between the input representation, masking geometry, and what an embedding retains. Satellite/geospatial imagery is a candidate because it connects to my research, but the final dataset and downstream task are still open. The initial image adapter is intentionally small and uses only synthetic single-channel images.

The design reference is [I-JEPA](https://arxiv.org/abs/2301.08243) and its [official implementation](https://github.com/facebookresearch/ijepa). The current single-target-block sampler is a simplified experiment foundation, not a reproduction of the official multi-block training recipe.

## Working now

- Non-overlapping patch extraction with explicit positions and strict shape/finite-value validation.
- Seeded rectangular target blocks with context sampled outside the target.
- Separate context, target, and unused indices: the context encoder must never receive target pixels.
- Reproducible CLI mask inspection and tests covering partition integrity, preserved pixels, invalid shapes, and random-seed isolation.

## Run the mask demo

Python 3.11+, no runtime dependencies:

```sh
export PYTHONPATH=src
python3 -m jepa_lab.demo --seed 29
python3 -m unittest discover -s tests -v
```

`C` marks visible context, `T` marks prediction targets, and `.` marks unused patches. The command also prints indices and patch dimensions for inspection. All demo intensities are generated locally.

## Next model contract

1. A context encoder receives only visible patches and their positions.
2. A stop-gradient target encoder encodes the full input; target positions select prediction targets.
3. A predictor combines context embeddings with target positions to predict target representations.
4. An exponential moving average updates the target encoder from the context encoder.
5. Log loss **and** representation variance/covariance to diagnose collapse; evaluate frozen representations on a held-out downstream task.

The model stage will add PyTorch as an explicit dependency. Baselines and ablations share fixed splits, seeds, compute accounting, and an evaluation protocol. Improved latent prediction loss is not itself proof of useful embeddings.

For a fire-specific dataset, split by distinct fire event, time, and geography before sampling windows. A fire progression prediction head would be a separate experiment; latent prediction alone does not output fire polygons.

See [PLAN.md](PLAN.md) for the staged experiment and [DEVLOG.md](DEVLOG.md) for completed work.
