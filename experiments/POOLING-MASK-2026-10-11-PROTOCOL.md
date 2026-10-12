# Pooling and target-mask pilot — declared October 11, 2026

Committed before observing new pooling or 4x4-mask classification scores.
This is follow-up on the repeatedly used 500/200 validation pilot, not
independent confirmation. Previous mean-pooling results and classifier tuning
are already known; no official test examples/scores enter either phase.

## Fixed inputs and budgets

- CIFAR-10 manifest: `0333e4c2400a89a694e5b2f3b51ff39482e3cdf9397c5e6969fc9995ad1cec44`.
- Balanced selection: `e0c4645e619917b1f001ea3382961906349995f8df8c9e0d3307569ef73e0e78`;
  50 training and 20 validation examples per class (500/200 total).
- Existing 223-step checkpoint:
  `9943decda69020dc6e96174f5d5c6587390d207f000cbe6ca1255da731097b9e`.
- Architecture: 8x8 RGB patches (48 values), width 32, 4 heads, 2 encoder
  layers, 1 predictor layer. Frozen context encoder; no evaluation augmentation.
- AdamW classifier seeds: 29, 31, 37. Every arm gets 200 steps for each of
  eight settings: LR 0.0001/0.001/0.01/0.05 × weight decay 0.01/0.1.
- Ridge alpha grid: 0.01/0.1/1/10, deterministic and fitted once per arm.
- Normalization fitted only on the 500 training features. Settings selected
  on the same 200 validation examples using the existing tie-break rule.
- Local CPU, one thread, PyTorch 2.8.0; no downloads or paid services.

## A: pooling on fixed weights

Compare mean, max, and concatenated mean/population-std over all 64 tokens.
Mean/max have 32 features; mean_std has 64, so its classifier capacity is
different. Each arm gets the same seed-29 random encoder and matching pooling.
All arms reuse the exact pixel artifacts and fit pixel classifiers once per
classifier seed (ridge once). Verify every selected saved classifier and
paired error count without refitting. Inspect training-feature numerical and
entropy effective rank, variance, row norms and mean pair cosine; these are
descriptive, not a quality guarantee. No per-example arrays enter the public report.

## B: target geometry, with mean pooling fixed

Train one new 223-step trajectory from seed 20261003 with a 4x4 target block
and context fraction 0.75. Keep batch 4, learning rate 0.001, EMA 0.99,
architecture and sampled training-image order equal to the historical 2x2
trajectory. Verify the actual image order against its original 20+3+200-step
logs, not just an equal seed. Mean-pool the new encoder and evaluate with the
same phase-A classifier grid/seeds and exact random/pixel artifacts.

The mask changes target count 4→16 and visible context count 45→36; target
coverage and token workload are intentionally coupled. Equal optimizer steps
are **not** equal compute or equal prediction/visible-token budgets. This is
one training seed, not an isolated proof of a mask-shape effect. Record loss,
feature geometry, selected scores and paired errors versus the existing mean
arm. Do not adapt the budget/grid after seeing results.

## Reproduce and report

```sh
export PYTHONPATH=src
.venv/bin/python experiments/run_pooling_mask_20261011.py pooling
.venv/bin/python experiments/run_pooling_mask_20261011.py mask
```

The driver checks fixed input fingerprints and writes all raw reports,
features and selected states beneath ignored `runs/pooling-mask-20261011/`;
the custom checkpoint stays in ignored `checkpoints/`. Publish aggregate
metrics, scalar provenance and explicit limitations only, in separate
phase-A/phase-B results. A missing or inconsistent cached artifact/log must
fail rather than silently substitute a different run. Reserve the official
test partition, a pretrained baseline and multi-training-seed confirmation
for later fixed experiments.
