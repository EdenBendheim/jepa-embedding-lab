# October 9 bounded pilot protocol

Declared before this pilot is run or its scores are observed. The October 7 exploratory result informed this new protocol; this is a follow-up validation experiment, not an untouched test evaluation.

- Dataset: official CIFAR-10 binary files, fixed version-one manifest `0333e4c2400a89a694e5b2f3b51ff39482e3cdf9397c5e6969fc9995ad1cec44`.
- Pilot selector: existing SHA-256 per-class ordering inside locked train/validation partitions, seed **20261009**, **50 training / 20 validation examples per class** (500/200 total). Official test is reserved.
- Source checkpoint SHA-256: `dc3c9c55ed90fc50da1f94b5c3ac2ddda1704319815be1db4c3e4038e18b73af`.
- Training: resume the existing **23-step** context/EMA checkpoint for **200 additional steps** (223 total), batch four, learning rate 0.001, EMA 0.99, original checkpoint/manifest seed, one CPU thread, same model architecture. Save to a new ignored path; retain the original checkpoint.
- Frozen representations: all 64 context patch tokens mean-pooled; one matching random encoder with initialization seed **29**; scaled NCHW raw pixels. All use identical ordered pilot examples.
- AdamW probes: **200 steps**, learning rates **0.0001 / 0.001**, weight decay **0.1 / 1.0**, classifier initialization seeds **29 / 31 / 37**. Same four-setting grid for each representation and seed. Select by validation accuracy, then cross-entropy, then declared order.
- Ridge probes: **alpha 0.01 / 0.1 / 1 / 10**, same grid for every representation. Training-only normalization; unpenalized intercept; select by validation accuracy, then MSE, then declared order. Ridge is deterministic and fitted once per representation, not averaged as if three seeds were independent runs.
- Persist all nine selected AdamW probes and three selected ridge probes locally. Recheck saved scores against the exact feature artifacts without refitting.
- Report every selected result, class confusion/support, mean/sample standard deviation across **probe** initialization seeds, the training loss/variance diagnostics, settings and artifact fingerprints. Do not change these budgets/grids after observing scores; report failures instead.

## What the result can establish

The run checks the larger balanced evaluation pipeline, baseline regularization, and sensitivity to classifier initialization. One encoder checkpoint and one random encoder are held fixed. Classifier-seed variation does **not** quantify encoder-training uncertainty, data-selection uncertainty, independent confirmation, or official test performance. A 223-step model remains a very small training pilot. No learned-embedding advantage is presumed.

Only source, configuration/provenance and aggregate results belong in public commits. Images, features, fitted classifiers and checkpoints stay ignored. No private research data, paid API or cloud compute is used.
