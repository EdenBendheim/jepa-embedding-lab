# Experiment plan

Started September 29, 2026; model implementation completed its first working synthetic training/checkpoint milestone October 2. CIFAR-10 RGB images and frozen-embedding classification were selected October 3 as the first public-data experiment. Geospatial experiments remain a later direction.

## Foundation

- [x] Synthetic single-channel image patchification with shape and finite-value checks.
- [x] Seeded target-block/context masking; target/context separation and index tests.
- [x] CLI inspection of patch and mask contracts.
- [x] Select CIFAR-10 and frozen-embedding ten-class classification as the first public experiment.
- [x] Write fixed train/validation/test manifests while preserving the official test partition.
- [x] Add RGB input adaptation with tested channel/patch order and a fixed transform.
- [x] Run a bounded public-data training smoke check using training images only.
- [x] Extend resumable training/checkpoints to public data, preserving manifest identity and sample RNG state.
- [x] Run a bounded public-data pilot and resume it while preserving training state.

## Compact model — begin October 1

- [x] Add PyTorch context encoder, stop-gradient target encoder, and predictor.
- [x] Implement EMA updates and checkpoint/config persistence.
- [x] Verify gradient boundaries, mask gather shapes, and EMA behavior.
- [x] Run a small smoke experiment; log embedding variance and training loss.
- [x] Establish a frozen pretrained baseline appropriate to the selected modality.

## Custom embeddings and useful comparisons

- [x] Extract frozen full-image context/target embeddings with a documented pooling rule.
- [x] Add seeded random-encoder and raw-pixel feature representations on identical fixed examples.
- [x] Verify feature artifacts against selected examples/labels, numerical integrity, architecture, and weight/runtime provenance before evaluation.
- [x] Add a shared class-balanced pilot manifest and exact-subset feature extraction without test selection.
- [x] Persist features/labels with exact partition indices, manifest/checkpoint identity, and configuration.
- [x] Compare three embedding widths at the same update/data/evaluation budget.
- [x] Compare learned versus fixed spatial positions at width 32 under matched initialization, budgets and fresh rows.
- [x] Replay and retain observed selections in a source-file-pinned flat exposure ledger.
- [x] Add resume-bound positional amplitude and matched scale-comparison APIs.
- [x] Predeclare a fixed-position amplitude contrast with retained exposure history.
- [x] Train and evaluate that contrast with matched journals and frozen fresh-row scoring.
- [x] Hold the data split, downstream task, update budget, and evaluation code fixed.
- [x] Add frozen cosine-neighbor selection, saved indices and untuned confirmation.
- [x] Persist selected probe weights/normalization with source identity and reproduced scores.
- [x] Add a matched regularized ridge head and a larger predeclared pilot.
- [x] Fit bounded linear probes with training-only normalization and validation-only hyperparameter selection.
- [x] Run a matched real-image checkpoint/random/pixel comparison and report its exploratory pilot limitations.
- [x] Repeat classifier initialization on matched frozen features and label the limited uncertainty scope.
- [x] Repeat encoder-training/random-encoder seeds in a bounded pilot and distinguish descriptive seed variation from test/data uncertainty.
- [x] Compare fixed-checkpoint mean/max/mean_std with matching random pooling and shared pixel classifiers.
- [x] Inspect verified feature geometry and reproduce selected classifiers/paired class errors.
- [x] Implement resume-bound custom masks and run a declared 4x4-target pilot matched to actual historical image order.
- [x] Reserve balanced confirmation rows excluding all declared earlier tuning examples.
- [x] Score frozen selected AdamW/ridge classifiers on fresh examples without refitting or retuning.
- [x] Publish configurations, provenance, plots, a results table, and a small row-and-label embedding explorer.

## Next session

The October 30 run continued into October 31 with the objective-specific
feature/tuning/confirmation workflow. Shared controls, complete candidate-state
replay, selected-state locks and pre-access observation receipts work in
synthetic integration tests, including interrupted recovery and no-fit/no-image
replay. The [complete prospective protocol](experiments/CONTEXT-VARIANCE-2026-10-31-PROTOCOL.md)
retains the earlier 500-row unscored reservation and original history. All six matched
budgets now completed 2,223 updates each with equal actual image order. Locked frozen
confirmation is now complete: regularized/prediction-only AdamW 22.13%/29.20%,
primary −7.07 points. Ridge and kNN also worsen; all 280 candidate states, 40
selected heads and 10 indices replay without refitting or raw-image reads.
Next inspect fitting-image representations, then predeclare a new hypothesis.
Use `runs/context-variance-20261031/observed-exposure.json` for any later
reservation: 3,246 observed rows, 1,754 unused. Preserve all earlier source files. Report every control,
paired errors and negative results; keep the official test reserved.

The [October 18 position comparison](experiments/POSITION-2026-10-18.md) is
complete: three new fixed-position runs and three historical learned-position
runs, 2,223 updates each with matched non-position initialization and actual
image order. On 500 fresh rows, trained fixed AdamW 29.04% versus learned-position
27.47%, but fixed random 28.93%. A general trained-feature advantage remains
unproven; pixel kNN is still higher. Reproduced 60 linear artifacts and 13 indices,
plus paired fresh recoveries/regressions, with no-refit cache replay.

The [October 19 exposure audit](experiments/EXPOSURE-2026-10-19.md) resolves
the historical depth blocker. Six pinned source files retain all 2,246 declared
observed rows, with 2,754 remaining. Source replay, append-only evidence extension,
version-three selection, validation-only features and no-refit scoring work. The
500-row reservation smoke is unscored; a future experiment must predeclare its
contrast and use complete then-current history. Fingerprints cannot detect
omitted experiments, so keep original files and register all new observations.

The [October 20 amplitude protocol](experiments/SCALE-2026-10-20-PROTOCOL.md)
was executed [October 22](experiments/SCALE-2026-10-22.md): three new .02 runs
versus pinned 1.0 controls, equal capacity and exact actual image order. Fresh
AdamW means are 25.53% / 26.51%, a negative −0.98-point primary contrast.
Random controls remain close; useful learned embeddings are not established.
The driver locks and audits all states, records conservative exposure before
fresh access, and supports no-fit/no-image cache replay.

The [October 24 training-only audit](experiments/STABILITY-2026-10-24.md)
compares original fitting-image context/teacher geometry and matching random
controls. Context effective rank is 3.25–3.88 and endpoint teacher CKA exceeds
0.993, without proving quality or temporal stability. No held-out rows were read.
The [October 25 fixed-mask audit](experiments/MASKED-CONTEXT-2026-10-25.md)
completed the full/subset/masked comparison on these same training rows. Low
effective rank persists; four-mask sensitivity cannot establish useful features
or robustness. October 27 added an opt-in variance hinge on visible-context
image means, with synthetic/CIFAR training, exact resume/budget identity and
validated frozen-feature provenance. Prediction-only defaults remain unchanged.
The [prospective objective recipe](experiments/CONTEXT-VARIANCE-2026-10-27.md)
specifies the contrast; the matched training driver and source pins now exist.
Its downstream scoring path is complete; the intervention worsened fresh
accuracy. No quality gain is established.
Do not tune on the latest confirmation scores.
For any future reservation, retain `runs/context-variance-20261031/observed-exposure.json`
and its predecessor `runs/scale-20261020/observed-exposure.json` plus all original
source files. The new bundle retains 3,246 observed rows with 1,754 unused; the
500-row objective reservation is now observed. Fingerprints cannot detect omitted
experiments; retain all original evidence. Keep the official test reserved.
No raw images, features, weights or row-level evidence belong in public commits.
