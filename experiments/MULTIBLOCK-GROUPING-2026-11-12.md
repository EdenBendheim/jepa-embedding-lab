# Joint versus separate target prediction

On the fixed 500 CIFAR-10 training rows, joint prediction averaged 23.47%
accuracy and separate prediction averaged 23.80%. The matching random
encoders averaged 24.40%. Joint prediction uses a shorter total sequence
budget per image, with a mean accuracy difference of −0.33 percentage points
in this pilot.

The [protocol](MULTIBLOCK-GROUPING-2026-11-12-PROTOCOL.md) was committed before
training in `88d5edee30831da043fcdf219f319f27c3f4e212`. Unlike the November 11
comparison, the two arms share all four target tiles and every visible patch.
They use identical initialization, image order, and 32 optimizer updates per
seed. Only the grouping of predictor inputs changes. Joint prediction sends
all target mask tokens through one pass so they can attend across blocks.
Both modes share one context encoding and one stopped teacher pass.

| Seed | Separate | Joint | Random | RGB statistics | Joint minus separate | Scoring rows seen by encoders |
|---:|---:|---:|---:|---:|---:|---:|
| 29 | 22.0% | 22.0% | 24.4% | 20.8% | 0.0 pp | 1 |
| 31 | 25.2% | 24.8% | 24.6% | 20.8% | −0.4 pp | 2 |
| 37 | 24.2% | 23.6% | 24.2% | 20.8% | −0.6 pp | 0 |
| Mean | 23.80% | 23.47% | 24.40% | 20.80% | −0.33 pp | |

Every representation uses the same five stratified ridge folds on the same
500 training rows. Joint prediction recovered 8, 8, and 9 cases and regressed
8, 10, and 12 against separate prediction. Each seed uses the same scoring
selection, so the mean describes encoder variation on that selection. The
short training budget and overlap with encoder training limit the result.
[Per-seed aggregates and provenance](multiblock-grouping-results-20261112.json)

Separate prediction processes four 40-token sequences per image. Joint
prediction processes one 52-token sequence. That reduces predictor tokens
from 160 to 52 and attention pairs per head and layer from 6,400 to 2,704.
These counts describe sequence work. End-to-end latency, memory, and FLOPs
remain unmeasured.

The run indexed 768 training records across six encoders and the same 500
scoring records per seed. It verified the actual image and mask journals.
An instrumented second run reproduced every aggregate field with encoder
training disabled. It re-encoded the 500 scoring images per seed and refit the
fixed folds. Both passes indexed zero validation and official test images.
All 362 files from the preceding objective and layout studies retained their
hashes. Weights, row journals, and detailed fold results remain local.

Use the [protocol commands](MULTIBLOCK-GROUPING-2026-11-12-PROTOCOL.md) to
reproduce the comparison. Synthetic checks verify joint output against a
single forward over the ordered target union, hidden-pixel isolation,
stop-gradient teacher outputs, exact resume, and rejection of a changed
grouping on resume.

Interview explanation — “I fixed the images, masks and initial weights so I
could measure the effect of grouping the target tokens in one predictor pass.
That cut the token count by about two thirds, with a small accuracy drop in
this pilot. A longer run on independent scoring rows would test the tradeoff.”
