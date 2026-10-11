# October 10 checkpoint-age and training-seed protocol

Declared before running the new comparisons. The October 9 late-checkpoint and baseline scores are already known; this is a controlled follow-up on reused validation examples, not a fresh holdout or independent confirmation.

## Fixed data and classifier budget

- Official CIFAR-10 binary files; version-one manifest `0333e4c2400a89a694e5b2f3b51ff39482e3cdf9397c5e6969fc9995ad1cec44`.
- Reuse the October 9 balanced **500 training / 200 validation** selection, fingerprint `e0c4645e619917b1f001ea3382961906349995f8df8c9e0d3307569ef73e0e78`. Official test images remain unused for training/extraction/scoring.
- Fixed `x / 127.5 - 1` RGB transform; mean of all 64 context tokens; 32-dimensional embeddings, four heads, encoder depth two, predictor depth one, 8×8 grid of 4×4 RGB patches.
- AdamW: 200 full-batch steps, classifier seeds **29 / 31 / 37**, learning rates **0.0001 / 0.001 / 0.01 / 0.05** × weight decays **0.01 / 0.1**. These are the eight settings actually used in the corrected October 9 protocol. Same grid and budget for every representation; validation accuracy, then cross-entropy, then declared-order selection.
- Ridge: penalties **0.01 / 0.1 / 1 / 10**, validation accuracy, then MSE, then declared order. Fit once per representation. Both heads use training-only normalization.
- Retain all selected classifiers locally; verify their saved scores and paired errors without refitting. Publish aggregate scores, class errors, configuration and provenance only.

## A: controlled checkpoint age

- Early: 23 completed steps; checkpoint fingerprint `dc3c9c55ed90fc50da1f94b5c3ac2ddda1704319815be1db4c3e4038e18b73af`.
- Later: 223 completed steps; checkpoint fingerprint `9943decda69020dc6e96174f5d5c6587390d207f000cbe6ca1255da731097b9e`.
- Both record training seed **20261003**, batch four, learning rate 0.001, EMA 0.99 and the same architecture. The October 9 training log records resuming the early checkpoint for 200 steps; the feature contract itself cannot prove ancestry.
- Extract only the early checkpoint's new 500/200 features. Reuse the existing late features and exact random-seed-29/pixel artifacts from October 9. Fit shared baselines once per classifier seed and ridge once total.
- Report every age/head/seed, changes against the early checkpoint and both baselines, per-class recoveries/regressions and prediction histograms. Do not select a new training budget after seeing results.

## B: encoder-training sensitivity

- Train two additional models **from scratch**, seeds **20261010 / 20261011**, each for **223 steps**, same architecture/batch/learning rate/EMA and one CPU thread. Each seed controls initialization, training-image sampling and masks, so this explores their combined variation.
- Evaluate each on the exact same 500/200 pilot with the classifier grids above. Matched random-encoder seeds **31 / 37**, respectively, vary random initialization too. The original training seed 20261003 and random seed 29 supply the third result.
- Report each encoder's classifier-seed mean separately, and a descriptive mean/sample standard deviation across the three encoder means. Keep ridge results separate. Varying encoder and classifier seeds does not provide data-sampling uncertainty; reused validation tuning still limits interpretation.
- Report first/last training loss and target variance diagnostics, failures and class errors. No advantage over random or raw pixels is presumed, and no scores will be hidden because they are worse.

## Local bounds

Two additional 223-step CPU runs, 500/200 frozen feature extraction per model, and the declared bounded probes. No cloud compute or paid API calls. Images, features, fitted heads, checkpoints and full local reports stay under ignored `data/`, `runs/`, and `checkpoints/`. No private research data is used.
