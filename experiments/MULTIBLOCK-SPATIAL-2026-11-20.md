# What four spatial regions add on the fitting panel

Ordered quadrant features improved the longer joint encoder’s average fold
accuracy from **23.60% to 25.13%**. Matching random quadrants reached 25.20%.
Six RGB statistics per quadrant reached **27.20%**, the highest mean among
these readouts. The improvement from quadrant pooling shows that region-aware
readouts can help on this panel. The controls leave a learned spatial advantage
unestablished.

| Source | Global mean | Ordered quadrants | Patch shuffle | Quadrant shuffle |
| --- | ---: | ---: | ---: | ---: |
| Separate, 658 updates | 23.47% | 25.87% | 22.53% | 22.93% |
| Joint, 658 updates | 22.33% | 24.33% | 22.20% | 21.93% |
| Joint, 960 updates | 23.60% | 25.13% | 23.00% | 22.80% |
| Matching random | 24.40% | 25.20% | 21.13% | 21.33% |
| RGB moments | 20.80% | 27.20% | 19.40% | 19.20% |

The shuffle columns use the same heads as the ordered quadrant column.
Each head fits only on original rows in the corresponding fold complement.
Patch shuffling places intact 4×4 patches in new positions. Feature shuffling
keeps the four regional vectors and permutes their order. Both are distribution
shifts for the fitted head.

The longer joint quadrant readout recovered 46, 44 and 44 predictions while
regressing 39, 37 and 35 against its own global readout, for seeds 29, 31 and
37. Against matching random quadrants, its seed differences were −1.0, +1.2
and −0.4 percentage points. Regional RGB exceeded every trained regional
score on these folds. The longer joint quadrant head lost 2.13 points after
patch shuffling and 2.33 points after feature shuffling. Random and regional
RGB heads lost more under both perturbations. These drops measure sensitivity
to the declared shuffles.

The [protocol](MULTIBLOCK-SPATIAL-2026-11-20-PROTOCOL.md) was committed in
`3fa9a5765d4063459796833110bf8c9cb69a691e` before extraction. It reused the same
12 frozen sources, 500 images, five folds and patch permutations as the
November 19 audit. The new global readout reproduces all 24 original and
shuffled encoder accuracy counts exactly. Repeating the global vector four
times with the matched L2 penalty produces identical original-view predictions
for every source. Model weights, feature matrices and row IDs stay local.

The first pass indexed the 500 fitting images exactly once. Cache replay
reproduced the aggregate byte for byte with image indexing, encoder calls and
training disabled. Earlier scoring artifacts and the exposure ledger kept
their hashes. All 44 focused tests, protocol preflight, mask and training
smokes and aggregate checks passed. [Aggregate results](multiblock-spatial-results-20261120.json)

Some fitting images overlap the encoder-training journals, and seeds share
the same panel. The scores describe this exploratory fitting sample. The
validation result from November 18 remains the last independent sample, and
the official test set stays reserved.

For interviews, “I kept the encoder fixed and changed only how I read its
tokens. Quadrants improved fitting-fold accuracy, but random quadrants
matched the longer encoder and regional color did better. I used matched
shuffles to verify that the new heads were sensitive to layout.”
