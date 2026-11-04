# What the variance penalty changed

The variance penalty amplified between-image spread about 10.6 times across
four fixed masks. The full-image context embeddings' effective rank fell from
3.53 to 1.88, so the spread was concentrated in fewer directions. This pattern
coincides with the earlier 7.07-point drop in frozen AdamW accuracy. The
diagnostics describe endpoints on the fitting selection and leave the cause of
the accuracy drop open. [Previous frozen comparison](CONTEXT-VARIANCE-2026-11-01.md)

## Fitting-image audit

The audit used the same 500 original probe fitting images, in their recorded
order, for three prediction-only checkpoints, their three matched
variance-penalty checkpoints, and one seeded random model per pair. Each trained
checkpoint had completed 2,223 updates. All model settings, checkpoint hashes,
and actual training image orders passed the existing locked-run replay before
the audit indexed images. Dataset hash and label checks also ran to verify the
original split. [Locked experiment protocol](CONTEXT-VARIANCE-2026-10-31-PROTOCOL.md)

Each image was encoded under the same four masks from seeds 29, 31, 37, and 41.
Masks shared the original 2×2 target and 45 visible patches. The crossed analysis
uses the visible context means. It separates variation in image means, the shift
shared by all images under a mask, and the residual interaction between an image
and a mask. Effective rank is computed separately on full-image context
embeddings, using the entropy of the normalized covariance eigenvalues.

| Arm, mean of three seeds | Between-image variance | Shared mask variance | Interaction variance | Full-image effective rank |
| --- | ---: | ---: | ---: | ---: |
| Prediction only | 0.17581 | 0.001150 | 0.002240 | 3.534 |
| Variance penalty | 1.86715 | 0.000792 | 0.069269 | 1.877 |
| Random initialization | 0.06119 | 0.001230 | 0.000810 | 2.224 |

The average image share of variance was 98.1% for prediction only and 96.4% for
the variance penalty. The shared mask component fell in absolute terms, while
the interaction grew from 1.26% to 3.57% of variance. This decomposition applies
to the four views used here. It measures neither robustness across all masks
nor the semantic information in the image effect.

## Prediction controls

For each model and mask, the audit measured Smooth L1 on the aligned predictions
and teacher targets. It then reassigned predictions to other images with cyclic
shifts of 1, 250, and 499 rows, retaining the same target slots. The template
averages the other 499 teacher targets at each slot. These are descriptive
controls on the fitting sample. Each model has its own teacher, which limits
cross-model comparisons of absolute loss.

| Arm, mean of three seeds and four masks | Aligned error | Reassigned minus aligned error | Leave-one-out template error |
| --- | ---: | ---: | ---: |
| Prediction only | 0.11172 | +0.19312 | 0.21409 |
| Variance penalty | 0.07460 | +0.63486 | 0.46360 |
| Random initialization | 0.53240 | −0.00058 | 0.05697 |

Aligned predictions beat the reassignments on all 24 trained seed–mask pairs.
The gap was close to zero for random initialization. The aligned advantage
shows sensitivity to image context on this sample. The test leaves open whether
the predictor uses low-level color, texture, or more useful structure.

## Reproduction and limits

The audit command prints aggregates, checkpoint identities, and a hash of the
ordered fitting selection. Run it from the repository with the completed local
artifacts.

```sh
PYTHONPATH=src .venv/bin/python -m jepa_lab.analysis.objective_representations \
  --root data/cifar-10-batches-bin --directory runs/context-variance-20261031 \
  --protocol experiments/context-variance-sources-20261031.json \
  --expected-sha256 0dd2e26a60885435085d59baf4070b330b689b7d875cbbbbc3b0ba2f28e49165
```

The diagnostic implementation is pinned at commit
`be02b3a731151437ba5f8970a5c5e14b648d92fd`. Two guarded runs read exactly the
500 fitting rows each, preserved 358 original artifact hashes, and returned
byte-identical JSON. Its SHA-256 is
`de4bc041e56d4d69115c77e7669b2408dce985523c6087d5c2a4cd4d7f98c381`.
The guard restricted image indexing to the training selection and disabled the
trainer. The original validation exposure ledger remains the source for any
future reservation.

The next fitting-only diagnostic is to compare embedding variation with simple
RGB statistics, using disjoint subsets of the existing fitting selection. This
can inform a prospective intervention on concentrated variance before spending
another set of untouched validation rows.
