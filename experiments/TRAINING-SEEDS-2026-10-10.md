# October 10 encoder-training sensitivity pilot

Completed comparison B of the [declared protocol](PILOT-2026-10-10-PROTOCOL.md): two additional models trained from scratch for 223 steps, alongside the existing 223-step model. [Machine-readable results](training-seeds-2026-10-10-results.json) contain every selected classifier, candidate scores, training diagnostics, paired errors and artifact fingerprints. No raw images, features or weights are public.

## Three training seeds, one fixed evaluation selection

500 training / 200 validation examples, identical input transform, architecture, pooling, training hyperparameters and downstream grids. Each training seed controls model initialization, sampled training images and masks. Each learned/random representation receives classifier seeds 29/31/37; the numbers below average those three classifier results, rather than treating nine classifiers as nine independently trained encoders.

| Training seed | Random-encoder seed | Learned AdamW mean | Random AdamW mean | Learned ridge | Random ridge |
| --- | --- | --- | --- | --- | --- |
| 20261003 | 29 | 26.67% | 25.17% | 27.0% | 24.5% |
| 20261010 | 31 | 26.17% | 25.17% | 24.5% | 25.0% |
| 20261011 | 37 | 28.50% | 26.67% | 25.5% | 26.5% |
| Mean across encoder runs | — | **27.11%** | **25.67%** | **25.67%** | **25.33%** |

Raw pixels remain **28.00% AdamW mean / 30.00% ridge** in every run. Pixel feature values were checked to be identical; these repeated pixel measurements are reuse, not additional independent baseline evidence.

Sample standard deviation across the three encoder-run AdamW means is **1.23 percentage points** learned and **0.87 pp** random. These are descriptive seed variation, not confidence intervals or data-sampling uncertainty. The same validation examples select classifier hyperparameters and produce the reported accuracy; only three encoder seeds and a very small training budget were tested.

One learned seed exceeds pixels under AdamW (28.5% versus 28.0%), while the learned aggregate and every learned ridge result remain below pixels. Reporting every run prevents a favorable seed/head from becoming an unsupported improvement claim. This pilot still does not demonstrate a reliable learned-representation advantage.

## Training diagnostics

| Training seed | First recorded step | First recorded loss | Loss at step 223 | First / final target mean std |
| --- | --- | --- | --- | --- |
| 20261003 | 24, resumed log | 0.277407 | 0.107335 | 0.669266 / 0.655759 |
| 20261010 | 1 | 0.491274 | 0.145408 | 0.683748 / 0.693020 |
| 20261011 | 1 | 0.551691 | 0.096625 | 0.523938 / 0.802422 |

The original model's record above is the retained 200-step resume log, so its first recorded loss is step 24 rather than step 1. New models started at step zero and each completed 223 updates. All recorded sampled indices belonged to the fixed training partition. Finite nonzero variance and falling loss are diagnostics; neither establishes useful embeddings or rules out other failure modes.

## Reproduction and checks

For seeds 20261010 and 20261011, run `jepa_lab.cifar10_train` from scratch with `--steps 223 --seed SEED --batch-size 4 --learning-rate 0.001 --momentum 0.99`, saving to `checkpoints/cifar10-seed-SEED.pt` and `runs/cifar10-training-seed-SEED.json`.

Evaluate with `jepa_lab.compare` using the shared selection, `--steps 200 --learning-rates 0.0001 0.001 0.01 0.05 --weight-decays 0.01 0.1 --probe-seeds 29 31 37 --ridge-alphas 0.01 0.1 1 10`, and random seeds 31/37. Retain features under `runs/pilot-features-seed-SEED`, fitted heads under `runs/pilot-probes-seed-SEED`, and reports at `runs/pilot-comparison-seed-SEED.json`. The original model is taken from the controlled age comparison under the same new grid.

```sh
.venv/bin/python experiments/summarize_training_seeds.py
```

This experiment-specific summarizer verifies the current dataset, locked feature sources, model/training settings, matched grids, identical pixel values, all recorded training samples, saved classifier state/scores and every paired error count. It publishes aggregate source/metric JSON without refitting or sampling images. All **36 selected classifiers** reproduced exact train/validation scores. All **62 local tests** and the synthetic training/masking smoke commands passed. Compute was bounded local CPU only; official test images were not sampled or scored.

## Next

Inspect class confusions and the mean-pooling/masking recipe, choose a suitable frozen pretrained baseline, and predeclare any model/pooling ablation before evaluating it. Longer training and a final fixed test protocol are still needed. Keep résumé claims about implementation/evaluation capability separate from a claim that these embeddings outperform baselines.

## Interview explanation

“I repeated encoder training and random initialization, kept the examples and classifier budgets fixed, and reproduced every selected score from saved weights. One seed looked better than pixels, but the overall result did not, so I reported the variation instead of picking the favorable run.”
