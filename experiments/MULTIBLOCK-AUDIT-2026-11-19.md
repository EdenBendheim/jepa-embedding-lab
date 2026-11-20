# What the longer grouping encoders retain

Six global RGB statistics predicted about **78% of the longer joint encoder’s
feature variation** on the fixed fitting panel. Its out-of-fold classifications
changed little when whole 4×4 patches were shuffled. Adding its RGB residuals
raised training-panel fold accuracy from 20.80% to 24.73%. Adding the matching
random encoder’s residuals reached 24.60%.

The [protocol](MULTIBLOCK-AUDIT-2026-11-19-PROTOCOL.md) was committed in
`beb2341` before extraction. The analysis reopens the November 18 checkpoints
and uses their original 500 CIFAR-10 fitting images. The calculations below
average seeds 29, 31 and 37. All encoders see the same rows, labels and folds.

| Encoder | RGB reconstruction R² | Permuted fit R² | Effective rank | Residual effective rank |
| --- | ---: | ---: | ---: | ---: |
| Separate, 658 updates | .785 | −.007 | 3.06 | 3.81 |
| Joint, 658 updates | .781 | −.004 | 3.01 | 3.56 |
| Joint, 960 updates | .782 | −.003 | 3.41 | 3.99 |
| Matching random | .811 | .007 | 3.20 | 5.96 |

The longer joint run increased effective rank on these rows. Color
reconstruction remained at nearly the same level. The near-zero permuted
scores show that image-to-embedding alignment matters for reconstructing the
features. The two 658-update trained arms had original-view CKA of .993–.997
within their matched seeds. The longer joint endpoint had CKA of .966–.969
against its own 658-update checkpoint.

| Encoder | Original view | Shuffled view, same heads | RGB residual only | RGB plus residual |
| --- | ---: | ---: | ---: | ---: |
| Separate, 658 updates | 23.47% | 23.33% | 17.67% | 24.13% |
| Joint, 658 updates | 22.33% | 22.20% | 17.60% | 23.87% |
| Joint, 960 updates | 23.60% | 23.40% | 18.80% | 24.73% |
| Matching random | 24.40% | 24.67% | 17.33% | 24.60% |

RGB alone reached 20.80%. The extended joint residuals improved that by 3.93
points and the random residuals by 3.80. The 0.13-point difference between the
combined scores corresponds to two additional correct predictions across the
three evaluations of the same 500 images. The residual projection and ridge
classifier both fit inside each scoring fold’s complement. A ridge residual can
retain color information outside the fitted linear component.

Shuffling preserved each image’s whole-patch RGB values. Across all sources
and seeds, centered CKA between corresponding original and shuffled features
was above .9995. The result describes one perturbation to a global mean
representation. The shuffle preserves structure inside each patch and creates
a distribution shift, so this test cannot measure sensitivity to every kind of
spatial information.

Between 20 and 46 of the 500 fitting images also occurred in a trained
encoder’s journal, depending on seed and endpoint. These results describe
training-partition diagnostics and leave the earlier fresh validation contrast
unexplained. The official test remains reserved.

The first run indexed exactly the 500 fitting rows once. Cache replay
recomputed the fold fits and reproduced the aggregate byte for byte with image
indexing, encoder forward passes and encoder training disabled. The previous
scoring artifacts and exposure ledger retained their hashes. The 39 focused
tests, protocol preflight, mask demo, training smoke and aggregate checks
passed. [Aggregate results](multiblock-audit-results-20261119.json)

For interviews, “The longer training run improved validation accuracy, so I
tested what the embeddings actually retained. Color explained most of their
variation. Residual embeddings added some class signal, but random features
added almost as much, and patch shuffling barely moved the scores.”
