# October 24: training-only pooled-feature and teacher audit

## Question and scope

The amplitude comparison did not establish useful learned embeddings. Before
choosing another intervention, inspect whether spatial token variation masks
nearly constant image means and whether the student and EMA teacher disagree.
This is a retrospective, exploratory diagnosis using **only the original 500
fitting images**, not a predeclared accuracy comparison. No new training,
classifier fitting, tuning, validation image reads or test evaluation occurred.
The known validation results motivated this diagnosis; these observations are
not independent evidence for a performance claim.

The original scale protocol, dataset fingerprints, selection and six completed
2,223-step checkpoints were rechecked. Both scales use the same fitting image
order, batch size 16 for extraction and a matching random initialization for
each training seed. Full-image context/teacher outputs include all 64 tokens;
this audit does not measure the masked predictor or a training trajectory.

## Results

Effective rank is the exponential entropy of the centered covariance spectrum.
It measures how concentrated variance is, not the number of useful semantic
features. The between-image fraction divides variance of image means by total
variance over images and spatial tokens, averaged over coordinates.

| Scale / seed | Context effective rank | Teacher effective rank | Random context effective rank | Context between-image fraction | Random fraction | Context–teacher CKA | Parameter gap / teacher norm |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0.02 / 20261003 | 3.416 | 3.430 | 3.307 | 64.04% | 50.32% | 0.9935 | 2.27% |
| 0.02 / 20261010 | 3.878 | 3.849 | 3.926 | 66.46% | 56.74% | 0.9959 | 2.21% |
| 0.02 / 20261011 | 3.763 | 3.689 | 3.039 | 71.69% | 67.72% | 0.9974 | 1.85% |
| 1.0 / 20261003 | 3.470 | 3.513 | 2.028 | 25.59% | 26.60% | 0.9955 | 2.90% |
| 1.0 / 20261010 | 3.878 | 3.803 | 2.446 | 24.80% | 19.97% | 0.9946 | 3.23% |
| 1.0 / 20261011 | 3.254 | 3.192 | 2.199 | 26.41% | 24.55% | 0.9976 | 2.69% |

All six context matrices have centered numerical rank 31 out of 32 coordinates,
while effective rank is 3.25–3.88. This rules out exactly identical pooled
vectors on these rows, but their variance is concentrated. Matching random
controls also have low effective rank (2.03–3.93). Pixel geometry has effective
rank 34.39 in 3,072 dimensions; unequal dimensions and scales prevent a direct
quality ranking from these numbers.

The student and teacher have high centered CKA (0.9935–0.9976) and paired cosine
(0.9856–0.9958) at the saved endpoint. Their parameter gap is 1.85–3.23% of the
teacher parameter norm; fixed positional buffers are identical. These describe
agreement now, not stability across updates or proof of correct historical EMA.

Scale .02 has substantially more between-image variance than scale 1.0, yet
its previous fresh classification primary contrast was negative. A favorable
geometry statistic alone is therefore not a selection criterion or evidence
of improved classification. No threshold or winning setting was chosen here.

## Equations

For equally sized token arrays z[i,t,d], define m[i,d] = mean_t z[i,t,d].
Population variance decomposes exactly as:

```
within  = mean_i,t,d (z[i,t,d] - m[i,d])²
between = mean_i,d   (m[i,d] - mean_i m[i,d])²
total   = within + between
fraction = between / total
```

For centered aligned image-feature matrices X and Y, centered linear CKA is
`sum((X Xᵀ) * (Y Yᵀ)) / (||X Xᵀ||F ||Y Yᵀ||F)`. It ignores global scale,
rotation and translation. Coordinate RMSE and paired cosine complement it.
Constant/singleton representations have undefined CKA, serialized as null.
Parameter lag is `||student - teacher||₂ / ||teacher||₂`, excluding buffers;
a zero reference norm is null. None of these measurements uses class labels.

## Reproduce and verify

```
PYTHONPATH=src .venv/bin/python -m jepa_lab.stability_audit \
  --output runs/stability-20261024/audit.json
PYTHONPATH=src .venv/bin/python -m jepa_lab.stability_audit \
  --output runs/stability-20261024/audit.json --reuse-only
```

The [aggregate report](stability-2026-10-24-results.json) is the exact local
cache, with recipe/checkpoint/data/code/runtime fingerprints and a report
checksum. It contains no image IDs, labels, raw images, feature arrays or
weights. Its file SHA-256 is
`2d6819cf1dd27a82b4bf717aa7e078167174c0b44a860df0385572237d8e28bd`.
The checksum detects accidental changes, not a forged producer.

Real cache replay with image access, training and diagnostic recomputation
patched to fail retained byte-identical output. Tests additionally cover known
CKA invariances, row-order sensitivity, spatial-only variation with collapsed
means, EMA interpolation of a known parameter gap, invalid/degenerate inputs,
training-only image selection, stale/corrupt caches and failed atomic saves.
All **185 local tests passed**, along with the mask demo and a three-step
synthetic training smoke. Raw data/model artifacts stay ignored. No downloads or paid compute were used.

## Next step and retained limits

Inspect full-image versus masked-context pooled variation on the same training
images with fixed masks before selecting one objective/normalization change.
Then predeclare that intervention and its comparison; do not optimize any new
setting on the last confirmation results. Endpoint diagnostics cannot identify
an optimal recipe, establish temporal stability, or replace held-out accuracy.

The complete exposure history remains
`runs/scale-20261020/observed-exposure.json`: 2,746 observed validation rows and
2,254 unused rows. This audit consumes no new held-out rows. Keep the official
test set reserved and preserve all earlier source evidence.

Interview explanation: “I checked whether variation across patches was hiding
identical image-level embeddings. The representations varied but had a very
concentrated covariance spectrum, while student and teacher stayed aligned at
the saved endpoint. I used training images only and kept the weak accuracy
result explicit instead of treating diagnostic improvement as model success.”
