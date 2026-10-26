# October 25: fixed-mask context sensitivity on fitting images

## Question and boundary

The October 24 audit found concentrated full-image pooled variance. Check whether
removing patches or changing their attention context materially changes that
picture before choosing another training intervention. This is an exploratory
endpoint diagnosis on the **original 500 fitting images**, motivated by already
known validation results. No new training, classifier fitting, hyperparameter
selection, validation image reads or official test evaluation occurs.

The six pinned 2,223-step amplitude checkpoints and their exact-initialization
random controls are reused. Every image receives four shared fixed masks with
seeds 29, 31, 37 and 41, a 2x2 target block and context fraction .75 of non-target
positions: 45 visible patches. Batch size is 16, one CPU thread, PyTorch 2.8.0.
Mask recipe hash: `8388079233811515154a1a55de673b422146feefadae99d10e7bc38325511dd2`.
The masks are deterministic diagnostic views, not a representative sample of all
training masks or real-world corruptions.

## Three views of each image

1. **Full**: encode all 64 raw patches, then average output tokens.
2. **Subset**: average the 45 visible positions from those same full-attention
   outputs. This isolates pooling changes, but those tokens already attended to
   hidden patches. It is deliberately a diagnostic control and must never be
   used as a leak-free training context.
3. **Masked**: gather the 45 visible raw patches before the encoder, then average
   their outputs. Comparing subset with masked exposes the additional effect of
   recomputing attention without hidden patches. The predictor is not evaluated.

## Aggregate results

CKA columns average four aligned, per-mask centered linear CKA values. Rank is
covariance-entropy effective rank, not numerical rank or semantic information.
Mask fraction measures within-image variance across these four views divided by
total variance across images and views; it is not prediction error.

| Scale / training seed | Trained mask fraction | Random mask fraction | Full rank | Mean-of-masked-views rank | Full/subset CKA | Full/masked CKA | Subset/masked CKA |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0.02 / 20261003 | 1.42% | 1.69% | 3.416 | 3.390 | 0.9976 | 0.9846 | 0.9925 |
| 0.02 / 20261010 | 1.32% | 1.93% | 3.878 | 3.855 | 0.9980 | 0.9819 | 0.9902 |
| 0.02 / 20261011 | 1.28% | 1.81% | 3.763 | 3.711 | 0.9989 | 0.9840 | 0.9898 |
| 1.0 / 20261003 | 1.98% | 2.70% | 3.470 | 3.454 | 0.9934 | 0.9851 | 0.9971 |
| 1.0 / 20261010 | 1.92% | 3.99% | 3.878 | 3.838 | 0.9942 | 0.9833 | 0.9960 |
| 1.0 / 20261011 | 1.79% | 3.24% | 3.254 | 3.203 | 0.9956 | 0.9843 | 0.9949 |

Within-image mask variation is 1.28–1.98% of the trained representations' total
variance under these views, versus 1.69–3.99% for random controls. Full/masked
CKA is 0.9819–0.9851, while averaging masked views leaves effective rank near
3.20–3.85. These observations do not suggest that full-image pooling alone
explains the concentrated representation on these rows. They do not establish
robustness, causal explanations for the earlier accuracy results, or an advantage
over random features. Four shared masks can miss important sensitivities.

For scale .02, mean subset/masked coordinate RMSE is 0.0897–0.1001, versus
0.0427–0.0475 for full/subset. For scale 1.0 those ranges are 0.0502–0.0536
and 0.0583–0.0637. Pooling and attention changes have different coordinate
contributions, but RMSE is scale-dependent; it cannot rank representation
quality or be added as an independent causal decomposition.

## Variance calculation

For masked pooled embeddings z[i,v,d], let m[i,d] = mean_v z[i,v,d]. With the
same number of views for every image, population variance decomposes as:

```
within  = mean_i,v,d (z[i,v,d] - m[i,d])²
between = mean_i,d (m[i,d] - mean_i m[i,d])²
total   = within + between
mask_fraction = within / total
```

The ratio is null if total variance is zero. Per-view CKA/RMSE/cosine and geometry
are retained in the [aggregate report](masked-context-2026-10-25-results.json),
alongside source, checkpoint, code and mask identities. There are no raw images,
row indices, labels, embeddings or weights in that published report.

## Reproduce and limits

```sh
PYTHONPATH=src .venv/bin/python -m jepa_lab.stability_audit \
  --masked-context --output runs/masked-context-20261025/audit.json
PYTHONPATH=src .venv/bin/python -m jepa_lab.stability_audit \
  --masked-context --output runs/masked-context-20261025/audit.json --reuse-only
```

Real cache replay with raw-image access, training and diagnostic recomputation
patched to fail reproduced the existing report without changing its bytes.
Synthetic checks cover fixed masks, the variance identity, caller state/RNG,
hidden-patch boundaries, exact training-only image access and incompatible cache
modes. All **189 local tests passed**, plus the mask demo and three-step synthetic
training smoke. Remote CI is recorded separately in the daily run record. No paid compute or new downloads were used.

Next predeclare one objective/normalization intervention at matched capacity,
image order and update budget, including random controls. Do not choose or tune
it using the last confirmation scores. All 2,746 observed validation rows remain
in the retained ledger, with 2,254 unused; this diagnosis consumes none. Keep the
official test reserved. A new objective still needs a frozen downstream comparison.

Interview explanation: “I separated what happens when I average fewer patches
from what happens when those patches can no longer attend to the rest of the
image. The low-rank picture persisted on the same training images, so I avoided
claiming that mask stability meant better features.”
