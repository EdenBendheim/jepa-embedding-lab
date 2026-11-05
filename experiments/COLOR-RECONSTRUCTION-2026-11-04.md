# How much do simple RGB statistics explain? — November 4, 2026

Six color/contrast statistics predict a large share of these pooled embeddings,
but **less** for the variance-penalty model than for prediction-only or random
encoders. The result weakens a simple linear-color explanation for why the
penalty increased variance while hurting downstream classification. It does
not identify the cause of that earlier accuracy drop.

The [protocol](COLOR-RECONSTRUCTION-2026-11-04-PROTOCOL.md) and
[exact sources](color-reconstruction-sources-20261104.json) were committed in
`4d265373bcfeea03ef6c7ddd34e9af9fe2fdbaf9` before inspecting these outcomes.
[Aggregate measurements](color-reconstruction-results-20261104.json) retain all
ten source results, fold counts, errors, geometry and fingerprints.

## Fixed-fold results

Five-fold reconstruction on the original 500 probe-fitting images. The six
inputs are RGB channel means and population standard deviations. Each fold
fits on 400 rows and scores 100 different rows, with training-only normalization
and fixed ridge alpha .01. Values below are descriptive means across the three
existing encoder seeds; pixels are a single shared source.

| Frozen source | Aligned reconstruction R² | Permuted-fitting control R² | Original effective rank | Residual effective rank |
|---|---:|---:|---:|---:|
| Prediction-only | 0.8220 | −0.0361 | 3.53 | 5.27 |
| Variance penalty | 0.6035 | −0.0447 | 1.88 | 2.22 |
| Random encoder | 0.9341 | −0.0378 | 2.22 | 5.99 |
| Raw pixels | 0.3126 | −0.0226 | 34.39 | 58.74 |

R² compares each row's reconstruction error with the target mean from its
fitting folds. Negative shuffled-control values mean that control performs
worse than this reference. The control permutes only fitting inputs, leaving
scoring inputs and targets aligned. It is a single fixed permutation per fold,
not a significance test. All sources use identical folds and fitting rows.

Across encoder seeds, aligned R² spans 0.801–0.857 for prediction-only,
0.585–0.613 for the penalty, and 0.926–0.943 for random encoders. Pixel R² is
lower because the same six global statistics cannot describe most spatial detail.
These are reconstruction scores in each representation's own units, not
classification accuracy or evidence that the random encoder is better.

The penalty's residual effective rank remains near 2.22. Removing a linear
color prediction therefore does not expose a broad residual representation in
this diagnostic. A nonlinear relationship with color could still matter. Five
out-of-fold heads also produce five different error mappings: their residual
geometry does not define a deployable residual encoder.

## Verification and reproduction

- The first run indexed exactly the pinned 500 fitting images, once each and
  in their original order. Every feature cache and completed checkpoint was
  verified before diagnostic image indexing; extraction and training were disabled.
- Replay reproduced the same cached bytes with image indexing, reconstruction
  fitting, model training and feature extraction disabled.
- All 358 files from the original completed objective experiment retained their
  hashes. No validation/confirmation/test image tensors were indexed by either
  invocation. Dataset byte hashes and labels are still read for provenance checks.
- The local cached-envelope SHA-256 is
  `13f389f0382568b375ca379c1e92e4346cf2fff8a1f8d66a7cbe245ba869fdae`.
  The public aggregate JSON includes its inner report digest. These checksums
  detect accidental differences; they do not authenticate intentionally edited reports.
- Local verification results are recorded in the development log. Images,
  per-row features, fitted heads, model weights and local caches remain ignored.
  This run used local CPU compute only.

From the repository root with the original local artifacts present:

```sh
python -m jepa_lab.analysis.color_cli \
  --root data/cifar-10-batches-bin \
  --directory runs/context-variance-20261031 \
  --objective-protocol experiments/context-variance-sources-20261031.json \
  --objective-sha256 0dd2e26a60885435085d59baf4070b330b689b7d875cbbbbc3b0ba2f28e49165 \
  --diagnostic-protocol experiments/color-reconstruction-sources-20261104.json \
  --diagnostic-sha256 fae3baa7df972d5cbd166b34450c56bdff42615282f5c02a3c2db4d7a1a23829 \
  --output runs/color-audit-20261104/report.json \
  --reuse-only
```

Use the declared source revision. A changed source file, feature artifact,
checkpoint or runtime cannot silently reuse this cache. Without `--reuse-only`,
a missing cache runs the declared calculation; an existing mismatched cache is
preserved and rejected.

## What this changes

The earlier [fresh-row result](CONTEXT-VARIANCE-2026-11-01.md) remains negative:
the penalty reduced AdamW accuracy from 29.20% to 22.13%. Today's fitting-only
analysis adds no new accuracy measurement and makes no quality claim. No
hyperparameters were selected from these results.

A useful next hypothesis is that global pooling discards spatial information.
A spatially retained representation needs a predeclared comparison, matching
random controls, equal source/data budgets and retained exposure history. The
current work provides the controls and replay trail for that next decision;
it does not establish that spatial pooling will improve accuracy.

Interview explanation: “I checked whether my embeddings mostly tracked six
simple image statistics. I fit the diagnostic on separate folds and included
a shuffled control. It explained 82% of prediction-only variation but only
60% after the variance penalty, so I narrowed the explanation instead of
claiming that more feature variance meant a better model.”
