# Multi-block CIFAR-10 pilot — November 11, 2026

The multi-block trainer now supports fixed CIFAR-10 training rows and exact
checkpoint resume. Its frozen encoder loader checks the checkpoint bytes,
dataset, runtime, source fingerprints, model settings, and deterministic mask
and image journals before loading a model for inference.

## Matched run

The [protocol](MULTIBLOCK-CIFAR-2026-11-11-PROTOCOL.md) was committed before
training in `2308b5b857a52262555e1720879add0c4c9b26fe`. The two arms each ran
32 updates with batch size four, one local CPU thread, and matching initial
tensors. Their checkpoints record the same 128 distinct training images in
the same order. Every image has 36 context patches and 16 target patches.

| Target geometry | Predictor passes | Sequence tokens per pass | Total predictor tokens per image | Attention pairs per image, head, and layer |
|---|---:|---:|---:|---:|
| One 4×4 block | 1 | 52 | 52 | 2,704 |
| Four 2×2 blocks | 4 | 40 | 160 | 6,400 |

Both arms share one visible-context pass and one full-image teacher pass.
The four-block arm repeats visible context in four predictor sequences.
Attention-pair counts describe sequence geometry. Measuring latency, memory
and FLOPs would require separate instrumentation. This intervention changes
target layout and predictor grouping together.

## Frozen feature comparison

The context encoders mean-pool all 64 tokens for the same 500 balanced fitting
rows. Five stratified folds train and score a ridge classifier with fixed
regularization .1. Both trained encoders and the same-initialization random
encoder produce 32 features. The RGB control uses three channel means and
three population standard deviations. One scoring image also appears in the
encoder-training journals for both arms.

| Representation | Correct / 500 | Cross-fitted accuracy |
|---|---:|---:|
| One 4×4 block | 114 | 22.8% |
| Four 2×2 blocks | 110 | 22.0% |
| Matching random encoder | 122 | 24.4% |
| RGB statistics | 104 | 20.8% |

Four blocks recovered 12 errors and introduced 16 relative to one block
(−0.8 percentage points). The random encoder had the highest accuracy in this
single-seed pilot. The scores describe the fixed fitting rows and this small
training budget. Broader claims require independent confirmation with
multiple seeds and an explicit compute budget.
[Aggregate scores, paired errors and provenance](multiblock-cifar-results-20261111.json)

## Reproduction

The first invocation indexed 256 encoder-training records across the two arms
and 500 scoring records. An instrumented second invocation reused both
checkpoints, indexed the same 500 scoring records, and reproduced every
aggregate field. The second invocation refits the fixed classifier folds.
Both invocations indexed zero validation and official test images. File
verification hashes all official binaries and reads their labels.

A separate real-image smoke run resumed two steps plus two steps and matched
every field of a continuous four-step checkpoint, including optimizer, RNG,
weights and journal. The previous objective study's 358 files retained their
hashes throughout the pilot. Checkpoints and sample journals remain local.
The [protocol](MULTIBLOCK-CIFAR-2026-11-11-PROTOCOL.md) has the preflight and
execution commands.

Interview explanation — “I tested whether several smaller targets learn
better embeddings than one larger target. I held initialization, image order
and visible and target token counts fixed. I counted the extra predictor tokens
and used the same frozen-feature classifier folds for both arms and a random control.
The random encoder still won this small pilot.”
