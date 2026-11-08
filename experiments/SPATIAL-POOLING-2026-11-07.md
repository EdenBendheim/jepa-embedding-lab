# Spatial pooling results — November 7, 2026

Keeping four image regions improved the trained encoder's cross-fitted ridge
score from 24.73% to 25.40%, but matching random encoders scored 25.27% and
repeated global means scored 25.67%. **This comparison does not establish a
learned-feature advantage or a benefit specific to spatial pooling.**

The [protocol](SPATIAL-POOLING-2026-11-07-PROTOCOL.md) and
[source lock](spatial-pooling-sources-20261107.json) were committed in
`b4cb0583a37ddb46db71bae13ec894d00507e006` before observing real-image outcomes.
[Aggregate JSON](spatial-pooling-results-20261107.json) retains all 26 source
scores, confusion matrices, fold counts, 30 paired comparisons, geometry and
130 saved-head fingerprints. Images, features and fitted weights remain local.

## Results on the original fitting selection

Five stratified folds over 500 images; each classifier fits 400 rows and scores
100 different rows. The frozen encoders may have pretrained on those scoring
images. These are exploratory classifier cross-fitting results, **not held-out
representation evaluation**. Each row below uses fixed ridge alpha .1 without
search. Means and sample standard deviations describe three encoder seeds,
not uncertainty over data or an independent test.

| Frozen encoder / representation | Dimensions | Accuracy mean | Encoder-seed sample std |
|---|---:|---:|---:|
| Prediction-only, global mean | 32 | 24.73% | 2.05 pp |
| Prediction-only, four quadrants | 128 | 25.40% | 0.40 pp |
| Prediction-only, shuffled quadrants | 128 | 20.27% | 1.03 pp |
| Prediction-only, repeated global mean | 128 | 25.67% | 1.03 pp |
| Random, global mean | 32 | 25.00% | 1.59 pp |
| Random, four quadrants | 128 | 25.27% | 0.61 pp |
| Random, shuffled quadrants | 128 | 19.53% | 1.33 pp |
| Random, repeated global mean | 128 | 25.47% | 1.70 pp |
| Shared six RGB statistics | 6 | 20.80% | — |
| Shared raw pixels | 3,072 | 16.00% | — |

The fixed raw-pixel score is not comparable to earlier validation-selected
pixel scores: this run uses different classifier subsets and one fixed alpha.
No baseline-specific parameter search was performed.

Quadrants outperform the image-specific shuffled-layout control by 5.13 points
for trained encoders and 5.73 points for random encoders. Consistent layout helps
both in this setting; it does not isolate useful information learned by JEPA.
The global mean and quadrant views came from the **same** context-token pass.
Shuffle permutations are identical across sources for each image.

The trained quadrants-minus-global-mean deltas vary across seeds: +1.2, −1.2,
and +2.0 points. Against repeated means they are 0.0, −1.0, and +0.2 points.
Against matching random quadrants they are +0.8, 0.0, and −0.4 points. These
paired descriptive comparisons prevent a small mean difference from being
presented as a clear training benefit.

Repeated means match classifier coordinate count but add no independent
information. Under fixed alpha, repetition can alter the effective penalty on
shared signal. They are a diagnostic for representation/regularization effects,
not proof of equal statistical capacity. Likewise, entropy effective rank rises
with spatial pooling but does not establish a better classifier representation.

## Verification and reproduction

The first invocation indexed exactly the original 500 fitting images once,
in their pinned order. Before access it verified the completed objective
experiment, cached controls, model identities, recipe and code fingerprints.
Encoder training and historical feature extraction were disabled. The original
358 experiment files retained their hashes and no validation, confirmation or
official-test image tensors were indexed. Dataset byte hashes and labels are
still read for provenance validation.

Each of 130 fold heads is bound to the exact disjoint classifier fitting and
scoring subsets, full feature digest, class labels, source identity, runtime and
fixed configuration. A second invocation disabled image indexing, encoding,
encoder training and both ridge-fitting paths. It loaded all heads, reproduced
the full aggregate report and preserved all 132 new artifact hashes (features,
report and 130 heads). A missing or altered head cannot silently refit during
score verification. Content hashes detect accidental differences rather than
authenticate against intentional forgery.

The inner report digest is
`b368152a78d9bec7e2606c4d54334453c1752935deb4f8aaa01c15192455cebe`.
Local check results are recorded in the development log. With the original
ignored data and artifacts available at the declared source revision:

```sh
python -m jepa_lab.analysis.spatial_run \
  --root data/cifar-10-batches-bin \
  --directory runs/context-variance-20261031 \
  --objective-protocol experiments/context-variance-sources-20261031.json \
  --objective-sha256 0dd2e26a60885435085d59baf4070b330b689b7d875cbbbbc3b0ba2f28e49165 \
  --diagnostic-protocol experiments/spatial-pooling-sources-20261107.json \
  --diagnostic-sha256 50c5cb3f48332f784d50db397299d46cb5da78c5505613738c7578ffad32de16 \
  --features runs/spatial-audit-20261107/features.pt \
  --output runs/spatial-audit-20261107/report.json \
  --reuse-only --verify-scores
```

Head files live beside the feature cache in `features.heads/`. Without
`--reuse-only`, a missing feature cache is extracted and missing fold heads are
fitted; existing mismatched evidence is rejected. `--verify-scores` reproduces
cached outcomes from saved heads; ordinary cache reads only validate the report
and its bound source artifacts. A changed code fingerprint requires a new
declaration and output path, or checking out the original revision.

## Next decision

A pooling-only confirmation is not justified by a trained-versus-random gap in
these results. Keep the 1,754 unused validation rows reserved. The next model
work should test a training change under a new prospective protocol, with
synthetic gradient/resume checks first and matching random controls throughout.
A bounded multi-target-block prediction variant is a candidate to implement;
it is not selected here as a proven improvement. Preserve the current defaults,
prior sources and exposure history, and do not tune against old confirmation
scores.

Interview explanation: “I compared global and spatial embeddings with random,
shuffled-layout and repeated-mean controls. Layout helped, but random encoders
benefited too. I saved every fold head and reproduced all scores without
refitting, then kept the negative result instead of claiming a model gain.”
