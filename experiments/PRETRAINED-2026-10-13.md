# Frozen pretrained reference — October 13, 2026

Executed the [prospectively committed protocol](PRETRAINED-2026-10-13-PROTOCOL.md)
and [pinned historical sources](pretrained-sources-20261013.json) locally on
one CPU thread. All 104 synthetic tests passed.

## Results

500 fitted training / 200 tuning-validation images, identical ordered examples
for every arm. Eight shared AdamW settings, 200 optimizer steps, classifier
seeds 29/31/37; shared ridge penalties 0.01/0.1/1/10. These rows were already
observed in earlier pilots. The official test set and October 12 confirmation
rows were not used.

| Representation | Feature dimensions | AdamW mean validation accuracy | Classifier-seed sample SD | Ridge validation accuracy |
| --- | ---: | ---: | ---: | ---: |
| JEPA (223 steps) | 32 | 26.67% | 0.76 pp | 27.00% |
| Matching random encoder | 32 | 25.17% | 0.29 pp | 24.50% |
| Raw pixels | 3072 | 28.00% | 0.50 pp | 30.00% |
| Frozen ImageNet ResNet-18 | 512 | 72.50% | 0.87 pp | 75.50% |

The 16 selected saved classifiers (12 AdamW, four ridge), their exact sources,
reported configs/scores, paired prediction errors and aggregate calculations
all reproduce without refitting. A cache-only retry with extraction, fitting
and image sampling disabled produced a byte-identical summary. A deliberately
corrupted cache copy was rejected while preserving its previous summary.
An initial manual retry without the required one-thread runtime was correctly
rejected; the one-thread retry succeeded. Losing candidate states are not
saved and were not refitted by the audit.

[Full aggregate metrics and provenance](pretrained-2026-10-13-results.json)
include per-classifier-seed results, feature/source hashes and exact recipes.
Raw images, feature arrays, model weights and classifier states remain ignored.

## What this tells me

The frozen pretrained reference scores far above the compact JEPA in this
pilot. It gives me a useful transfer-learning reference for the next training
study. The existing JEPA still does not beat the pixel baseline consistently;
these results do not establish useful learned-embedding advantage.

This comparison cannot isolate the learning objective: ResNet-18 has
supervised ImageNet-1K pretraining, 11.7 million original parameters, a different
architecture, 224-pixel inputs and 512 features. JEPA uses a compact 32-D
transformer and only 223 steps. External pretraining data overlap has not been
audited. Probe-seed SD is classifier initialization sensitivity, not encoder
seed variation, sampling uncertainty or a statistical significance test.
All scores are validation-selected, not official test accuracy.

## Reproduce

Install `.[pretrained]`; obtain the official local weight file as documented
in the protocol. No weight download occurs inside the driver. From the repo:

```sh
PYTHONPATH=src .venv/bin/python experiments/run_pretrained_20261013.py
PYTHONPATH=src .venv/bin/python experiments/run_pretrained_20261013.py --reuse-only
```

Exact source bytes and the protocol hash must match. Cache reuse requires the
recorded torch/torchvision/CPU-thread runtime. Any changed feature, classifier,
protocol or report must pass validation; failures retain earlier output.

Next: predeclare a stronger bounded JEPA training budget and repeat selected
mask settings across encoder seeds. Reserve different untuned validation rows
for any new confirmation, and keep the official test set reserved.

Interview explanation: “I added a frozen pretrained reference to my JEPA lab,
matched the downstream data and classifier budgets, and reproduced every
selected score from saved states. The baseline exposed a large quality gap,
which gives me a measurable target for improving the model.”
