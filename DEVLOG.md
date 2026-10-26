# Development log

## October 25, 2026

- Added fixed-mask context diagnostics comparing full-image pooling, visible-subset
  pooling of full-attention tokens, and raw-visible-patch re-encoding. The subset
  control deliberately retains full-image attention and cannot serve as a leak-free
  training context. Reports separate mask variation within an image from variation
  between image means. Three synthetic tests pass for deterministic views, the
  variance identity, hidden-patch isolation, state/RNG preservation and invalid inputs.
  No real-data or quality result is claimed in this first implementation step.

## October 24, 2026

- Ran the aggregate audit on the original 500 fitting images for all six
  completed scale/seed checkpoints and matching random controls. Context
  effective ranks are 3.25–3.88 (numerical rank 31); student/teacher endpoint
  CKA is 0.9935–0.9976. These diagnose concentrated variance and current
  agreement, not useful features or stability over time. The earlier weak
  classification results remain unchanged. Published aggregate-only evidence
  and interpretation in experiments/STABILITY-2026-10-24.md.
- All 185 local tests passed, plus the mask demo and three-step synthetic
  training smoke. Real cache replay with image reads, training and diagnostic
  recomputation disabled was byte-identical. No new held-out rows, images,
  features, weights or private data were published; official test is reserved.
  The next step checks full-image versus masked-context variation on training
  rows before predeclaring one objective/normalization intervention.
- Interview explanation: “I separated variation within an image from variation
  between images, checked student/teacher agreement, and reported concentrated
  representations without treating diagnostic statistics as accuracy gains.”

- Added a CLI and atomic aggregate-report cache bound to source/data/checkpoint,
  code and runtime identity. Two new integration checks prove image-free,
  calculation-free replay, stale-code/corrupt-report refusal, protected evidence
  and cleanup after failed atomic replacement. All four audit integration
  tests pass; CLI help works. The checksum is not producer authentication.

- Connected stability diagnostics to the completed amplitude experiment using
  only its original fitting images. The audit compares context/teacher and
  matching initialization controls, spatial/pooled geometry and parameter gaps;
  pixel geometry is shared. Two synthetic integration tests verify exact
  training-only image reads, unchanged checkpoint bytes and caller RNG, no
  retraining, and rejection of changed weights before image access.

- Added named student/teacher parameter-gap diagnostics with separate buffer
  equality checks. Two new tests check the expected EMA interpolation of a
  known gap, unchanged state/RNG, missing or incompatible parameters and a
  zero-norm reference. Six stability tests pass. An endpoint gap cannot measure
  temporal stability or prove the training EMA updates were correct.

- Added the full-image token variance decomposition: within-image variation
  versus between-image pooled variation, with existing pooled geometry metrics.
  Two new tests demonstrate that large spatial variance can hide identical image
  means, verify the population variance identity and check degenerate/invalid
  inputs. All four stability tests pass; no quality threshold is invented.

- Added aligned pooled-feature RMSE, cosine and centered linear CKA diagnostics.
  Two synthetic tests verify row-order sensitivity, translation/rotation/scale
  invariance of CKA, undefined degenerate ratios, invalid inputs and unchanged
  caller tensors/gradients/RNG. Similarity does not establish useful features
  or stability across training time. No held-out images or scores were used.

## October 22, 2026

- Executed frozen amplitude evaluation on the predeclared 500 new rows,
  excluding all 2,246 earlier observations. Trained .02 / 1.0 AdamW means:
  25.53% / 26.51% (primary −0.98 points); matched random: 25.38% / 26.60%.
  Ridge favors 1.0; .02 improves kNN against 1.0 but remains below pixels.
  No broad trained-feature advantage or winning amplitude is established.
- Audited 60 linear artifacts (52 unique states), 13 neighbor indices and all
  candidate results. Real replay with training, raw-image reads, extraction,
  linear fitting and neighbor tuning disabled reproduced a byte-identical
  summary. All 175 local tests, mask demo, three-step synthetic training smoke
  and whitespace checks passed. CI is checked separately after the push.
- Published aggregate metrics, paired class errors, recipe/checkpoint/order
  hashes and training windows. Raw images, row selections, training journals,
  feature/weight/index artifacts and source bundles remain ignored. The new
  source-pinned ledger conservatively retains 2,746 observed validation rows
  and 2,254 remaining; original history files are unchanged, test is reserved.
- Interview explanation: “I isolated positional amplitude at equal capacity
  and matched image order, locked classifiers before fresh evaluation, and
  reproduced the result without fitting. The primary metric got worse and
  random controls stayed close, so I reported the negative result.”

- Completed all three predeclared .02 training runs at 2,223 updates, with
  one CPU thread and no paid compute. Replayed the pinned scale-one runs:
  each paired journal has exactly the same actual image order, 8,892 draws
  and 8,061–8,112 unique training images. Both arms have 29,472 trainable
  parameters. Added an aggregate-only no-image/no-training audit command;
  its synthetic test passed. Published checkpoint/order hashes and 100-step
  diagnostic windows. Finite losses/gradients are not evidence of quality.

- Added frozen scale evaluation: shared pixel heads, exact-initialization random
  controls, full saved-candidate/state audits, locked selections, paired
  small-minus-one comparisons and source-replayed exposure extension. Before
  fresh extraction, persist a conservative observation receipt; interrupted
  attempts keep those rows consumed. Three synthetic workflow tests pass for
  lock/receipt ordering, no-fit/no-image byte-identical replay, interrupted
  recovery and tampered-feature/receipt refusal with prior output preserved.

- Extracted the complete neighbor selection-cache workflow for reuse in scale
  evaluation and the existing position driver. It now checks the selection
  context/rule, every candidate score, adjacent artifact path and every partial
  state tensor. Two focused tests and all three position-driver regressions
  pass; cache replay cannot retune or silently substitute an alias index.

- Added a source-pinned amplitude training driver with bounded checkpointed
  chunks, exact .02/1.0 capacity and actual image-order checks, and refusal to
  overwrite historical evidence. Two synthetic workflow tests pass: completed
  caches replay without training/images; altered journals, source bytes, missing
  reuse caches and unsafe output locations fail before starting new training.
  Real scaled training and downstream evaluation remain pending.

## October 20, 2026

- Final verification: all 167 local tests passed; the seeded mask demo and
  three-step scaled synthetic training smoke passed with finite loss. Real
  preparation replay retained the pinned reservation and equal parameter counts.
  No real-image training or fresh-row classification scores were produced.

- Predeclared fixed-position .02 versus 1.0 at equal width/capacity, retained
  original source/code/checkpoint pins and added a no-image reservation preflight.
  The real check recreated the same 500 unscored rows while retaining all 2,246
  observed rows; both arms have 29,472 trainable parameters. The synthetic
  preflight check verifies idempotence, protected evidence and stale-source /
  conflicting-output failure preservation. Real scaled training/evaluation
  remain pending; no new accuracy result or official test use.
- Interview explanation: “I separated positional amplitude from encoding type,
  preserved exact initialization and reproducible resume, and required matched
  baselines and fresh held-out rows before judging whether the change helps.”

- Added positional-amplitude comparison/audit APIs with shared pixel heads,
  matched random controls and fixed non-scale configuration. Synthetic checks
  reproduce 15 saved classifier states without refitting, preserve sources/RNG,
  and refuse confounded mode/depth/budget/seed/row/pixel/checkpoint changes before
  fitting. This is comparison infrastructure, not a measured accuracy result.

- Wired positional scale into training/budget CLIs and random feature
  extraction. Checkpoint extraction retains the recorded scale. Synthetic CLI
  checks compare continuous and resumed full states, reject a changed scale
  before image sampling/output replacement, and preserve feature provenance.

- Added a nonnegative positional-signal scale shared by the encoder, teacher
  and predictor. Scaling occurs at use time, keeping initialized weights and
  RNG identical across scale arms. Default math and serialized historical
  configuration identities are preserved. Zero is an explicit no-position
  ablation. Focused synthetic checks cover both position modes, exact encoder
  math, hidden-pixel/teacher gradient boundaries, and invalid scales. No new
  real-image model scores are claimed.

## October 19, 2026

- Final local verification: all 160 tests passed (including ten exposure tests),
  the seeded mask demo and three-step synthetic training smoke passed, and the
  real source-bound CLI recheck reproduced the 2,246/2,754 observed/remaining
  counts. Official test remained reserved; no new representation accuracy claim.

- Added pinned-bundle extension retaining every original file/hash and source
  union, with idempotent duplicate registration and protected prior snapshots.
  Ten exposure checks cover this and the earlier source/reservation/workflow
  behavior. A first temporary-path assertion needed canonical path comparison
  on macOS; production evidence already used resolved paths.
- Replayed six real source selections: 1,746 observed rows through October 15,
  2,246 through October 18, and 2,754 remaining (275–277/class). An independent
  source traversal matches the flat union. The old collector's depth refusal
  was reproduced without weakening its guard. A 500-row reservation smoke
  excludes all history and remains unscored; no image sampling or classifier
  fitting occurred. Published aggregate hashes/counts only; bundles stay local.
- Interview explanation: “I retained the evidence of every recorded validation
  selection and replayed it before reserving new rows, so later experiments
  cannot silently forget earlier observations.” Model quality is unchanged.

- Integrated pinned-ledger reservation with the existing selector, feature
  loader and frozen scoring pipeline. Two workflow checks passed: pinned-source
  and expected-ledger failures, protected input files, atomic disk-error rollback,
  no-image selection, no fresh-row fitting, and unchanged AdamW/ridge states/RNG
  through source-verified scoring. No new real model/feature scores were measured.

- Added version-three balanced reservations over the flat ledger. Observed later
  selections retain source roots/hash and replay without recursively embedding
  history. Six sequential synthetic rounds preserve all ancestor rows, fitting
  rows and disjointness beyond the old depth bound; forged sources/roots/rows and
  legacy nesting are rejected. Historical v1/v2 fingerprints remain unchanged.

- Added a source-file-pinned build/recheck CLI, atomic bundle writes, read-only
  inspection and aggregate per-class capacity. Four exposure tests pass,
  including changed-byte/missing-file refusal, dataset identity, protected source
  paths and preservation of previous output. Bundles retain local evidence paths
  under ignored runs; no original selection is rewritten.

- Added flat validation exposure ledgers that replay original pilot/confirmation
  protocols, retain ancestor sources once, and derive the union from their exact
  selected rows. Two synthetic checks passed for deduplication, source/RNG
  preservation, aggregate capacity, and rehashed tampering/missing/cyclic sources.
  No images, fitting or new quality result. Later reservations and CLI are next.

## October 18, 2026

- Executed the declared position comparison: three new fixed-position runs at
  2,223 updates, shared actual image order with three reused learned-position
  models, exact-initialization random controls and 500 fresh confirmation rows
  excluding 1,746 observed rows. Trained fixed/learned-position/random-fixed
  AdamW means 29.04%/27.47%/28.93%. Broad learned advantage remains unproven;
  pixels still lead kNN. Reproduced 60 linear artifacts (52 unique) and 13
  neighbor indices, including candidate scores and paired fresh errors.
- Real cache-only replay with training/extraction/fitting/raw images disabled
  is byte-identical. Its first default-path attempt exposed an evaluation path
  pointing at the previous date; corrected it and added a default-path workflow
  check. No model/selection/metric changed. Full local suite 149 passed before
  that correction; all three driver checks passed afterward. Mask/demo smoke
  checks passed. Public result JSON is aggregate-only; real artifacts stay local.
- Current reservation ancestry now covers 2,246 observed rows and refuses
  another nested round. Next work needs a flattened, source-verified exposure
  ledger; do not omit earlier observations to get around the bound.
- Interview explanation: “I controlled initialization, sampled data and training
  budget while changing positional encoding, then replayed saved classifiers
  on new rows. The fixed-position model improved one comparison, but its random
  control stayed close, so I kept the limitation in the results.”


- Completed position evaluation with shared baselines, saved-state/candidate
  audits, a pre-confirmation classifier lock and paired fresh errors. Synthetic
  end-to-end check verifies lock-before-extraction, cache replay with all
  fitting/training/image access disabled, and corrupt-cache output preservation.
  No real scores were inspected before this workflow was committed.

- Prospectively declared the width-32 learned-versus-fixed position experiment:
  three seeds, 2,223 steps, exact non-position initialization/image order, shared
  old tuning recipe and 500 fresh rows excluding 1,746 observed examples.
  Added pinned training/cache driver; synthetic cache/order/tamper check passed.
  This declaration precedes new training and scoring; no result is claimed here.

- Added paired frozen-head confirmation and a saved-feature CLI. Two synthetic
  tests verify known recoveries/regressions, unchanged states/source bytes/RNG,
  no image reads or fitting, and rejection of changed/reordered fresh sources
  before restoring a classifier. This measures paired errors, not seed uncertainty.

- Added fixed-architecture position comparisons and saved-state auditing,
  sharing the existing width comparison fitter without changing its schema.
  Two synthetic tests verify shared pixel fits, exact state/paired/aggregate
  replay and pre-fit refusal of changed width/depth/budget/seed/examples.
  Fixed buffers change capacity; this is not yet a measured quality result.

- Exposed position choice through both public-data training entry points and
  random-feature extraction. Two additional CLI tests verify continuous/resumed
  equality, changed-mode rejection before sampling/checkpoint replacement,
  extraction metadata and refusal to override a recorded checkpoint choice.

- Integrated optional fixed spatial buffers into context/target/predictor while
  preserving default initialization, RNG draws and historical serialized keys.
  Canonical model-config serialization is shared by training and extraction;
  synthetic artifact fixtures use that wire contract. Two model tests verify
  matched non-position weights, no hidden-pixel gradients, fixed/EMA buffers,
  exact resume and mode-change rejection. Default/capacity checks also pass.
  All nine historical width-training caches reproduced with image sampling
  disabled, preserving their old config/capacity/journal identities.

- Added a deterministic row/column sinusoidal encoding with a documented
  formula and bounded dimensions. Two tests verify origin/axis/frequency
  ordering, independent tensors, finite float32 output and unchanged RNG.
  This foundation alone changes no existing model or quality result.

## October 16, 2026

- Linux CI exposed a too-strict bitwise comparison between 64+1-row and
  65-row matrix products in the new retrieval test. Replaced only that score
  assertion with float32 rtol=1e-6/atol=1e-7; exact row-identity, stable ties,
  source immutability and RNG checks remain. This required a fifth repair
  commit beyond the drawn four; it changes no model/index artifacts.

- Added matched neighborhood comparisons with common-depth overlap and exact
  shared/distinct row IDs, plus representation switching in the offline viewer.
  Split/query/label mismatches and edited comparison metrics are rejected.
  Two synthetic comparison tests passed. Restored real learned/random/pixel
  indices on 20 observed tuning rows: top-one overlaps 25% / 15% / 15%.
  No fitting, retuning, new confirmation exposure or official-test evaluation.
  Safari representation and query changes update tables and comparison text.
  All real reports remain ignored/local; this makes no new accuracy claim.
  Final local checks: all 137 tests passed, six explorer tests passed again
  after the final renderer change, mask demo and three-step synthetic training
  smoke passed. Git diff whitespace check passed.
  Interview explanation: “I made a source-verified viewer to inspect what learned
  embeddings retrieve and where they differ from random features and pixels,
  while keeping the evaluation split and fitted neighbor indices fixed.”


- Added an offline HTML explorer with query selection, ranked neighbor table,
  limitations and expandable provenance. Strict report checks, inert escaped
  JSON, text-only DOM insertion and a hashed-script CSP prevent report content
  from becoming executable markup. Two renderer tests cover malformed rows,
  injection text, CSP integrity and preservation of earlier output on failure.
  Opened the synthetic report in Safari: populated controls/table and visual
  layout verified with the offline CSP active.

- Added a source-verified neighborhood report/CLI that restores saved indices
  without refitting and maps neighbors to official fitting-row identities. Two
  synthetic workflow tests cover rank/order, fingerprints, mutation rejection,
  confirmation refusal and operation with raw-image reads disabled.

- Exposed ranked frozen-neighbor positions and cosine similarities, sharing the
  existing classifier normalization/tie rules. Four neighbor tests verify exact
  retrieval across a batch boundary, zero/tied vectors, unchanged sources/RNG
  and unchanged classification behavior.

## October 15, 2026

- Final local checks: 130 tests passed, followed by two updated width-workflow
  tests after adding no-refit replay of every neighbor candidate. Mask demo and
  three-step synthetic smoke passed. The cached real summary remains byte
  identical with all training/extraction/fitting/raw-image access disabled.
  Interview explanation: “I controlled initialization and data while changing
  model width, then checked both optimized heads and frozen neighborhoods on new
  examples. Wider was not consistently better, so I kept the mixed result.”

- Executed the committed width protocol: three widths x three seeds at 2,223
  steps, reusing the three verified width-32 models and training six wider ones.
  Exact image order matches. Reproduced 84 linear artifacts (76 unique states)
  and 19 neighbor indices; scored 500 fresh rows excluding 1,246 declared earlier
  rows. Learned AdamW means 28.16%/27.80%/25.47%; wider did not consistently help.
  The 32-D AdamW head exceeds random/pixels here; ridge/kNN still favor pixels.
  Published aggregate-only results and source-pinned workflow. Real cache-only
  retry with training, image reads, extraction and fitting disabled is byte
  identical. No general learned advantage, width winner or official test claim.

- Added no-refit neighbor confirmation using shared source/ancestry checks with
  linear probes. Synthetic checks preserve saved index/statistics/source bytes
  and RNG, reject altered/overlapping fresh rows before loading state, and verify
  unchanged normalization/count choice. Existing confirmation checks still pass.

- Added controlled-width fitting and saved-state auditing with exact
  initialization-matched random encoders and shared pixel classifiers. Two
  synthetic checks verify shared fit counts, reproduced selected states/paired
  errors and rejection of changed budgets, architecture, seeds, pixels or rows.
  This verifies comparison mechanics, not an observed quality improvement.

- Declared the fixed 32/64/96-width experiment before new training/scores:
  three seeds each, 2,223 steps, same mask/optimization/image order, exact
  initialization-matched random controls, shared probe grids and kNN counts.
  Reserved 500 fresh rows excluding all 1,246 observed rows. Added a recoverable
  driver and capacity accounting; synthetic width-order/cache check passed.
  The three already-complete width-32 runs are reused at their exact budget.

- Added shared width/head/encoder-depth/predictor-depth options to both public
  training CLIs. A real synthetic-fixture CLI run at width 64 / eight heads
  resumes exactly like four continuous steps and rejects a changed width before
  image sampling or checkpoint replacement. Historical default config unchanged.

- Added exact context/EMA-target/predictor parameter accounting and a labeled
  persistent FP32 memory lower estimate. Local RGB training is bounded before
  model allocation (width <=128, <=8 heads, encoder <=4 and predictor <=2 blocks);
  Boolean dimensions are rejected. Model/capacity tests passed, with RNG
  preservation and exact counts across widths 32/64/96. No compute equality claimed.

- Added saved-feature neighbor selection/restore CLI with fixed split/source
  verification, explicit no-retuning restore mode and protected earlier states.
  End-to-end synthetic check passed: no raw-image reads, unchanged restore,
  corrupted held-out features rejected and prior outputs preserved. Help works.

- Added tuning-only neighbor-count selection and atomic source-bound saved
  indices. Five neighbor checks passed, including exact restore without fitting,
  changed sources/state and confirmation-as-tuning rejection. Saved vectors,
  labels and normalization remain local model artifacts; public outputs contain
  aggregate scores/provenance only.

- Added a bounded frozen cosine kNN classifier with training-only normalization,
  deterministic row/vote ties, zero-vector behavior and unchanged inputs/RNG.
  Three synthetic checks passed. This tests embedding neighborhoods without an
  optimized linear head; it is not yet evidence of improved representations.

## October 14, 2026

- Added complete-journal fixed-window diagnostics and an optional Matplotlib
  renderer bound to exact published journal bytes/recipes. Two synthetic tests
  cover partial-tail windows, unchanged inputs, missing steps, changed sample
  order, non-finite/negative values and invalid bounds. Generated and visually
  inspected the six-run plot/aggregate points. Ribbons are seed ranges, not
  confidence intervals. Changing loss/token spread does not establish useful
  embeddings or exclude pooled-feature collapse. Matplotlib 3.11.2 used locally.

- Completed six 2,223-step runs with matched image order, compared the same
  500/200 pilot and scored 72 fixed selected states on 500 new confirmation
  examples excluding 746 observed rows. Fresh learned 2x2 / 4x4 / random / pixel
  AdamW means: 24.24% / 23.11% / 24.11% / 25.73%; pixel ridge 30.20%. All states,
  old scores, selection rules and paired errors reproduce. Real cache-only retry
  disabled training, sampling, extraction and fitting. 114 tests passed. No
  learned advantage, official test or paid compute is claimed. Published metrics
  and provenance only; actual journals/weights/features/classifiers remain local.

- Committed the prospective longer-budget protocol and driver: six 2,223-step
  runs, two masks across three encoder seeds, unchanged optimization recipe,
  matched actual image order, and 500 new confirmation rows excluding all 746
  previously observed rows. Synthetic repeated-mask/cache/preflight checks
  passed. New real-data training and scores follow this declaration.

- Added bounded multi-chunk training budgets with complete-journal/image-order
  verification, partial recovery and a no-training cache mode. Three synthetic
  checks reproduce a continuous trajectory, reject changed recipes/budgets or
  altered sampling order, and reject incomplete/legacy caches. CLI help works.

- Added periodic atomic checkpoints with an accumulated diagnostic journal.
  Simulated interruption and resume reproduce uninterrupted weights, optimizer,
  RNG and full history exactly. Malformed journals/intervals fail before image
  sampling; legacy checkpoints preserve the explicit missing prefix. Eight
  focused public-training tests passed.

- Added bounded confirmation ancestry: later rounds exclude previously scored
  confirmation rows and every earlier tuning ancestor, preserve the original
  fitted training rows, and reject corrupted or over-deep ancestry. The existing
  version-two format and historical pilot identities remain compatible.
  Focused confirmation checks passed.

## October 13, 2026

- Executed the committed frozen pretrained protocol on the exact existing
  500/200 pilot. JEPA/random/pixel/pretrained AdamW means: 26.67%/25.17%/28.00%/
  72.50%; ridge: 27.00%/24.50%/30.00%/75.50%. Reproduced all 16 selected
  classifiers and paired errors. A cache-only retry with extraction/refitting/
  sampling disabled produced byte-identical output; corrupt-cache copy failed
  without replacing its summary. Full local suite: 104 tests. Published
  aggregate metrics/provenance only; no images/weights/features/probes.
  One initial manual cache check omitted the required one-thread setting and
  was rejected; the correct runtime succeeded. No official test or paid compute.
  Interview explanation: “I added a pretrained transfer reference, matched the
  downstream budgets and reproduced every selected state. The quality gap gives
  me a measurable target for improving JEPA, without claiming an objective-only
  comparison or an embedding advantage.”

- Added no-refit transfer audit and the committed experiment driver. Selected
  states, reported candidate grids/rules, paired errors and seed aggregates
  must reproduce before summaries are written. Synthetic audit/cache checks
  passed, including reuse with fitting/extraction disabled and a corrupt cache
  preserving earlier output. Historical input bytes are pinned prospectively.
  The driver’s CLI help works; real-data execution follows this commit.

- Added the saved-feature transfer CLI with fixed-data verification, atomic
  reports and bounded classifier grids. Its synthetic end-to-end check passes:
  no raw-image reads, unchanged source bytes, all four arms fitted, and corrupt
  held-out labels rejected before fitting or replacing an earlier report.
  CLI help works.

- Added four-arm transfer comparison with exact ordered examples, identical probe
  budgets/seeds, ridge fitted once per arm, saved-state reproduction and paired
  errors. Two synthetic tests pass; they verify eight fits for two seeds and
  rejection of altered sources/confirmation rows before fitting. Architecture,
  supervision, resolution and width differences are explicitly reported.

- Added bounded offline selected-example extraction and its CLI. Three synthetic
  checks verify exact indices/labels, train-file namespace only, rejected test
  access/bad manifests/bounds before sampling, and prior output surviving failure.
  CLI help works. No raw images, weights or features are committed.

- Bound pretrained feature artifacts to exact named weights, torchvision runtime,
  full transform metadata, 512-D output and extraction bounds. Seven feature
  tests pass, including existing pixel/confirmation paths and rejection of
  altered pretrained provenance, dimensions and mutable metadata copies.

- Added offline loading of the exact named ResNet-18 weights, frozen 512-D
  extraction, finite-state checks, per-submodule mode restoration and caller RNG
  isolation. Eight focused tests pass; one synthetic-image smoke using the
  official local weight file produced finite 1x512 output. No CIFAR scores yet.

- Added optional pinned torchvision preprocessing: deterministic 32x32 uint8
  RGB to the named ImageNet 256-resize/224-crop recipe. Four synthetic tests
  pass for channel scaling, batch isolation, input bounds and unchanged inputs/RNG.
  CI installs the CPU torchvision build; no weights are downloaded in tests.

- Declared the frozen ImageNet ResNet-18 transfer baseline before extraction or
  scores. Fixed the already-observed 500/200 pilot, shared probe grid/seeds,
  artifact verification and explicit architecture/pretraining confounds.
  Protocol inspected against official torchvision 0.23 documentation.

## October 12, 2026

- Executed the committed 24-state confirmation protocol on 500 new balanced
  validation rows, excluding all 246 previously evaluated rows and preserving
  the original 500 fitted training rows. Learned mean_std / matching-random /
  pixels AdamW means: 25.67% / 32.73% / 28.20%; default mean and 4x4 mean both
  24.47%. All original states/scores reproduce; no normalization, classifier
  or settings were refitted/selected. A reuse-only retry with image extraction,
  sampling and fitting disabled produced a byte-identical report. A deliberately
  corrupted cached feature was rejected while preserving prior output.
  Published aggregate counts/configuration/provenance only. Full suite: 85 tests.
  Interview explanation: "I froze the classifiers, kept fresh examples out of
  tuning, and found that the early model improvements did not hold up against
  matching baselines. That tells me which claims are supported before I invest
  in stronger training." One encoder seed per mask and the shared validation
  partition remain limitations; no official test or paid compute was used.

- Added the no-refit confirmation CLI with dataset and selection verification,
  exact original/fresh artifact loading and atomic aggregate reports. A synthetic
  end-to-end check proves no image sampling/refitting or source mutation and
  preserves an earlier report when its input protocol is altered. CLI help
  checked. The October 12 fixed confirmation protocol is declared before scores.

- Added fixed saved-classifier confirmation scoring for AdamW and ridge.
  Original scores/state must reproduce before fresh rows are evaluated; exact
  feature provenance and no overlap with prior tuning/training rows are required.
  Two synthetic tests verify unchanged normalization/state/source bytes/RNG and
  rejection of altered representations, dimensions, indices and protocol.
  Content fingerprints also reject features/labels mutated after verification.

- Added confirmation selection CLI and validation-only feature extraction.
  Verified artifacts retain their confirmation status; matched fitting and
  saved-classifier persistence reject these rows to prevent retuning.
  Two synthetic workflow tests passed, including unchanged old output after
  invalid CLI input and legacy pilot compatibility checked by the full suite.

- Added reproducible confirmation selection: preserve the fitted pilot's exact
  training rows, exclude every validation row used by declared earlier pilots,
  and select balanced fresh rows only inside the fixed validation partition.
  Embedded source documents are revalidated, including after rehashed edits;
  nested/duplicate exclusions and insufficient class coverage are rejected.
  Three synthetic tests passed; no new real-data score has been observed.

## October 11, 2026

- Added fixed mean/max/mean-plus-population-std pooling to the frozen image API.
  Three synthetic tests check analytic moments, batch isolation, constant
  tokens, invalid inputs, no gradients/mutations, and preserved encoder state.
  The doubled mean_std feature width is explicit; no quality result is claimed.

- Integrated pooling into feature artifacts, verification and the matched
  comparison CLI. Old mean artifacts remain valid. Two synthetic pipeline
  tests cover all three contracts, mislabelled dimensions, matched random
  features, and wrong-directory reuse rejected before fitting.

- Added a controlled fixed-checkpoint pooling comparison API with matched
  random pooling and shared pixel classifiers. Three synthetic checks reject
  changed checkpoint/seed/examples, reproduce saved scores and paired errors,
  and prove one pixel fit per classifier seed (ridge once overall).

- Added the verified saved-feature pooling CLI, with duplicate-arm rejection,
  atomic reports and optional selected-classifier persistence. An end-to-end
  synthetic workflow confirms zero image reads, unchanged source artifacts,
  matched dimensions, and edited labels rejected before any fitting. CLI help checked.

- Added provenance-verified feature geometry inspection: numerical/effective
  rank, spatial-feature variance, row norms and mean pair cosine. Three
  synthetic checks cover analytic spectra, constant/zero rows, invariance,
  RNG/gradient isolation and a saved-artifact workflow with no image reads.
  Diagnostics are descriptive and never presented as classification quality.

- Added configurable target rectangles/context fractions to resumable RGB
  training, with explicit custom-mask checkpoint identity and patch counts.
  Three synthetic checks cover custom split-versus-continuous runs, changed
  masks rejected before sampling/replacement, malformed mask metadata and
  legacy default-checkpoint compatibility. Boolean geometry is rejected.

- Declared the fixed 500/200 pooling and 223-step target-mask experiment before
  observing its scores. Added a bounded two-phase driver that verifies source
  fingerprints, selected classifiers/paired errors, and actual historical
  image-order equality. Mean_std capacity and mask-token/compute confounding
  are explicit. Driver compilation/help checked before execution.

- Executed phase A on the declared checkpoint/500+200 examples. Learned
  mean/max/mean_std AdamW means: 26.67%/26.67%/28.67%; matching random:
  25.17%/27.33%/33.83%; pixels 28.00%. Ridge learned: 27%/24%/29.5%, random:
  24.5%/24.5%/30%, pixels 30%. Reproduced all 28 saved states and paired
  errors. Published aggregate metrics/provenance/geometry only. Full suite:
  77 tests. Spatial variance helped but does not demonstrate training benefit;
  reused validation and doubled mean_std classifier capacity are explicit.

- Executed phase B's 223-step 4x4-target run with identical initialization
  seed/optimizer/actual image order to the original 2x2 logs. AdamW learned
  validation mean 27.50% versus default 26.67%, random 25.17%, pixels 28.00%;
  ridge 27.5% versus default 27%, random 24.5%, pixels 30%. All 12 selected
  states and paired errors reproduce. Published aggregate results only;
  token counts/compute, one encoder seed and reused validation are explicit.
  Bound the cached training log to exact checkpoint bytes; a simulated
  mismatch preserved old output, and a retry with training disabled produced
  a byte-identical summary. Interview explanation: "I held initialization and
  image order fixed while changing mask coverage, reproduced every selected
  state, and reported a small pilot gain with its compute/data limitations."

## October 7, 2026

- Added a matched comparison CLI that extracts/reuses six balanced feature artifacts, copies checkpoint architecture for the random baseline, validates all data/provenance/configuration before fitting, shares probe grid/budget/seed, records all candidates/class outcomes/source hashes, and writes reports atomically. Published protocol, results and failure analysis in experiments/PILOT-2026-10-07.md.
- Final verification: all 43 tests passed locally. New end-to-end tests cover fixture extraction/verification/comparison, reuse without image reads, fairness failures before fitting, and report-write rollback. Real local CPU pilot: 100 training/50 validation examples, 23-step checkpoint, eight shared 200-step probe candidates; selected validation accuracy checkpoint 24% (12/50), random 30% (15/50), pixels 28% (14/50). No test features or scores, paid API/cloud, private/research data, or public weight/image artifacts.
- An initial LR 0.01/0.05, decay 0.01 grid exposed pixel overfitting (100% train, 20% validation, cross-entropy 102.4547); expanded the same grid for all representations. Scores are exploratory validation selection, not independent confirmation. The checkpoint has not demonstrated an advantage. Manifest is unchanged; selection hash cddc893ced41a1a7421977f78f929eec98e04f0bfa0ce39e29da5cf4902ab85b.
- Interview explanation: "I separated frozen representation evaluation from encoder training, matched the examples and architecture across baselines, and found that the early checkpoint still trails a random encoder on the exploratory pilot."
- Still pending: larger fixed pilots, stronger regularized baselines, multiple seeds, sustained JEPA training, pretrained comparison, probe-state persistence, and final official test evaluation.

- Implemented detached, bounded full-batch AdamW linear probes with training-only mean/std, near-constant-column handling, repeatable seeded initialization, weight-only decay, and preserved caller RNG. Added validation-only setting selection with fixed seed/budget, accuracy/cross-entropy, per-class counts and confusion matrices.
- Verification for this step: four new synthetic tests passed: ten-class separable features classify correctly, repeated fits match, changing validation values leaves fitted normalization/weights unchanged, source gradients/values are untouched, and malformed inputs or unequal candidate budgets fail.

- Added a reusable feature-artifact verifier and metadata-only inspection CLI. It checks exact balanced selection/namespace/labels, finite frozen float32 rows, recorded content hashes, normalization/pooling/architecture, checkpoint/random provenance, and runtime; test artifacts are excluded. File hashes bind reports to the actual loaded bytes without claiming producer authenticity.
- Verification for this step: three new tests passed for round-trip rows/file identity, array-copy isolation, wrong labels/indices/splits, edited features, incompatible metadata/runtime, and corrupted file rejection.

- Added a manifest-bound balanced pilot selector: deterministic per-class ordering within fixed train/validation partitions, exact indices, per-class counts and a fingerprint. Integrated full-selected-subset extraction; combining a pilot with test or a truncating limit is rejected before sampling.
- Verification for this step: three new synthetic tests passed for balance/disjointness/reproducibility/RNG isolation, malformed or edited selections, and exact extraction/no-test boundaries. Created the real local 100-train/50-validation selection after rebuilding the dataset manifest; image/model artifacts remain ignored.
- Interview explanation: "I fixed the same balanced examples for every representation, so a pilot comparison cannot silently change class mix or cross a data split."

## October 5, 2026

- Implemented frozen context/target full-image embeddings: fixed RGB patches, all 64 positions, mean pooling, inference without gradients, and restoration of caller model mode. Added seeded random-encoder and normalized NCHW raw-pixel feature baselines.
- Added bounded extraction from fixed train/validation/test namespaces and atomic local feature artifacts containing labels, exact indices, manifest identity, configuration/scaling/pooling, extraction settings, and feature hashes. Checkpoint provenance hashes the exact bytes loaded; invalid data/checkpoint identity is rejected before sampling. Random model construction preserves caller CPU RNG.
- Verification: 30 unit tests passed, covering pooling against direct full-position encoding, unchanged weights/gradients, restored modes on failure, seeded baselines/RNG isolation, exact example/label/namespace identity, repeatable artifacts, mismatched data/checkpoints, invalid bounds, and non-finite encoders.
- Real CIFAR-10 check: extracted the same 64 training examples using the 23-step checkpoint (32 dimensions), a matching random encoder (32), and raw pixels (3,072), plus 64 validation examples from the checkpoint via the CLI. All features were finite. Manifest hash remains `0333e4c2400a89a694e5b2f3b51ff39482e3cdf9397c5e6969fc9995ad1cec44`. Artifacts/metrics remain local and ignored; no paid compute or research data was used.
- Interview explanation: "I separated representation learning from evaluation and made checkpoint, random-encoder, and pixel features traceable to the same examples, so the next probe compares representations fairly."
- Still pending: linear probes, balanced pilot/full-split extraction, validation-only model selection, sustained training, pretrained comparisons, and any classification-quality result. The first-index feature checks are execution checks, not class-balanced benchmarks or measured representation quality.

## October 4, 2026

- Added bounded, resumable CIFAR-10 CPU training with manifest/model/settings/runtime identity checks. Checkpoints preserve the model, optimizer, global step, sampling RNG, and PyTorch RNG; deterministic mask seeds continue from the restored step. Labels are never used in the JEPA objective and only the fixed training partition is sampled.
- Added atomic checkpoint replacement and clear additional-step semantics, gradient norms, and loss/embedding diagnostics. Failed saves preserve the previous checkpoint. Saving is at invocation boundaries; interrupted unsaved steps are not recovered.
- Verification: 23 unit tests passed. Split-versus-continuous training matched diagnostics, all model/optimizer tensors, and both RNG states exactly in the same CPU environment. Mismatched splits, settings, model, runtime, missing state, and edited manifests were rejected before image sampling; a simulated disk error preserved the old checkpoint and cleaned its temporary file.
- Ran a real-image CPU pilot for 20 steps (batch four), then resumed the checkpoint for three additional steps. All losses/gradient norms were finite. Manifest hash remains `0333e4c2400a89a694e5b2f3b51ff39482e3cdf9397c5e6969fc9995ad1cec44`. Data, model artifacts, and metrics stay in ignored local directories; no paid API/cloud compute was used.
- Interview explanation: "I made real-data JEPA runs reproducibly resumable and tied each checkpoint to the exact data split, so an experiment cannot silently resume against different examples or training settings."
- Still pending: frozen full-image embeddings, probes, sustained training, and raw-pixel/random-encoder/pretrained comparisons. The pilot verifies training and recovery, not downstream accuracy or useful representations.

## October 3, 2026

- Selected official CIFAR-10 RGB images and frozen-embedding ten-class classification as the first public experiment. Downloaded the binary archive locally and verified its published MD5 before extracting the six expected files into ignored `data/`.
- Added binary record loading without pickle, content fingerprints, channel-preserving RGB patch preparation, and stable stratified splits: 45,000 train, 5,000 validation, 10,000 official test. The manifest hash is `0333e4c2400a89a694e5b2f3b51ff39482e3cdf9397c5e6969fc9995ad1cec44`.
- Added a bounded public-data smoke runner that revalidates the data/split manifest, samples only training images, and uses the existing JEPA loss, frozen EMA target, and diagnostics. Class labels are not used in its training objective.
- Verification: 18 unit tests passed, covering RGB channel/patch order and model gradients, corrupt binary records, stable/disjoint class-balanced splits, protocol fingerprints, and reproducible training-only smoke sampling. Ran three real-image CPU steps (batch four) with finite loss/gradients. A changed manifest is rejected before sampling. Data, images, weights, and run metrics remain ignored; the public protocol records metadata only.
- Interview explanation: "I connected the model to a real RGB dataset and locked down provenance and evaluation splits before fitting a probe, so later embedding comparisons use the same examples without leaking validation or test images into training."
- Still pending: sustained/resumable public-data training, frozen embedding extraction, linear probes, pretrained/raw-pixel/random-encoder comparison results, and any useful-embedding quality claim. Three smoke steps establish wiring only.

## October 2, 2026

- Implemented compact Transformer context/target encoders, positional embeddings, and a masked-target predictor. The teacher is frozen and evaluated without gradients; an EMA update moves it toward the context encoder.
- Context encoding gathers raw visible patches before attention. Per-example masks are checked for bounds, uniqueness, and target/context overlap.
- Added bounded synthetic CPU training, loss and embedding standard-deviation/covariance diagnostics, and resumable model/optimizer/configuration/RNG checkpoints. Added a pinned optional PyTorch training dependency and CPU training checks to CI.
- Verification: 11 tests passed, including gradient boundaries, prediction invariance to hidden-pixel changes, exact EMA interpolation/endpoints, mask validation, seeded training, and split-versus-continuous checkpoint resume. Ran 20 synthetic steps and resumed for three more steps successfully; these are execution checks, not a public-data quality result.
- Interview explanation: "I built masked embedding prediction with a frozen EMA teacher, verified hidden pixels cannot reach the context encoder, and made the training state reproducibly resumable."
- Still pending: dataset/task selection, data splits, real-data training, pretrained comparison, downstream probes, and useful-embedding results. Checkpoints and metrics remain in ignored local directories.

## September 29, 2026

- Started the image experiment foundation with non-overlapping patch extraction and position metadata.
- Added deterministic target-block masks, visible context sampling, and a synthetic CLI inspection demo.
- Added tests for target/context separation, complete grid coverage, input validation, pixel preservation, and seed isolation.
- Model training and custom learned embeddings remain future work, beginning October 1.
- Verification: six unit tests passed; the seeded CLI demo generated 64 patches, 45 context patches, and four target patches. Added GitHub Actions checks for Python 3.11 and 3.13.

## October 9, 2026

- Added atomic fitted-linear-probe persistence, tied to the exact verified training/validation feature artifacts. Loading validates state/configuration/runtime/fingerprints and reproduces recorded scores; it never refits or evaluates test. Round-trip checks cover restored scores/weights, caller RNG, tensor copy isolation, altered inputs/state/schema/runtime, corrupt files, and failed-save rollback. Runner integration remains next.

- Integrated selected-probe persistence into the matched comparison API/CLI. Reports record content-addressed classifier files and source/state hashes. Existing identical states are verified/reused; earlier classifiers survive subsequent runs. Synthetic checks reproduce each saved score and verify an identical rerun retains the same bytes/paths.

- Added a saved-probe scoring API/CLI that revalidates data, selection, feature artifacts and classifier identity, then reproduces scores without refitting or sampling raw images. A complete synthetic pipeline test checks identical recorded validation outcomes, no image reads/refitting, and rejection of edited labels before scoring.

- Added a training-only normalized multiclass ridge baseline with unpenalized intercept, bounded alpha selection, and smaller primal/dual float64 solves. Fitted ridge classifiers can use the same atomic artifact/recheck path. Synthetic tests check shrinkage, caller RNG/gradient isolation, validation independence, primal/dual equivalence, selection metrics, invalid settings, and artifact score round trips.

- Added matched multi-seed classifier comparison and optional ridge selection to the CLI. All seeds share fixed features and candidate budgets; ridge is deterministic and fitted once per representation. Reports retain every seed, mean/sample standard deviation, ridge metrics and saved states. Synthetic tests verify fairness, score restoration, caller RNG isolation, aggregate calculations and invalid-grid rejection before fitting. Declared the fixed 500/200, 223-step October 9 protocol before observing results.

- Executed the predeclared 500-training/200-validation pilot after committing its protocol. Resumed 23 to 223 JEPA steps; training-only sampling and finite diagnostics verified. Three AdamW classifier seeds measured means 24.83% checkpoint, 24.00% random, 28.00% pixels; matched ridge scored 27.0%, 24.5%, 30.0%. All twelve selected saved classifiers reproduced their scores; standalone no-refit CLI check passed. Published aggregate metrics/configuration/fingerprints, with no images/features/weights. All 54 local tests pass. Pixel baselines remain ahead; classifier-seed variation is not encoder-training uncertainty, and no official test score or learned-quality advantage is claimed.
- Interview explanation: "I compared learned, random and pixel features on the same examples, tested both AdamW and ridge heads, and made every selected classifier's score reproducible from its saved state. The pixel baseline still wins this pilot, guiding the next model/training experiment."

## October 10 — paired error diagnostics

Added bounded, descriptive paired classification counts: recovered/regressed predictions, per-class changes, confusion deltas and prediction histograms. Synthetic checks cover opposing changes, absent classes, identity, and invalid vectors; 2 focused tests passed. These do not imply statistical significance.

Integrated paired errors into matched comparisons for both classifier families using their selected predictions. Extracted one-representation fitting for later shared-baseline evaluation. Six comparison tests passed, including a deliberately permuted baseline and an exact three-fit assertion.

Added a saved-probe pair audit CLI with exact source alignment and score/state reproduction. A synthetic test covers both AdamW/ridge and rejects mismatched sources before scoring. A local real-data audit of the October 9 seed-29 checkpoint versus pixels reproduced 30 recoveries and 36 regressions on 200 validation examples (−3 percentage points), without refitting. Raw artifacts remain ignored. CLI help checked.

Added a controlled checkpoint-age input contract: 2–4 ordered ages, distinct checkpoint fingerprints, identical training/model/pooling settings and ordered examples, and exact shared random/pixel artifacts. A synthetic test rejects equal ages, duplicate weights, changed seeds/pooling/examples, and altered baseline fingerprints or values. Matching metadata does not itself prove checkpoint ancestry.

Implemented checkpoint-age evaluation with shared fitted baselines, repeated classifier seeds, deterministic ridge and paired errors against the earliest age and both baselines. Three progression tests passed; exact call counts prove shared-baseline reuse, saved classifiers persist, RNG is preserved, and a synthetic later-age regression is detected. Reserved baseline names are rejected.

Added the saved-feature progression CLI, dataset/selection verification and a synthetic end-to-end check proving zero image reads and unchanged source bytes. Declared the October 10 age comparison and two additional 223-step training seeds before running new scores; known October 9 validation results and reused tuning are disclosed. Full suite: 62 tests passed; CLI help checked.

Executed controlled 23/223-step comparison on 500/200 examples with the prospectively declared eight-setting grid. AdamW means 25.67%/26.67%, random 25.17%, pixels 28%; ridge 26%/27%, random 24.5%, pixels 30%. Sixteen saved classifiers and all paired errors reproduced. Published source/metric aggregates only. Corrected an erroneous historical grid reference in the new protocol: October 9 used four settings; the new eight settings were fixed before this run and did not change. Independent encoder-seed comparison B remains next.

Completed declared encoder-seed comparison: trained seeds 20261010/20261011 from scratch for 223 steps each, evaluated matching random seeds 31/37 with identical 500/200 examples and eight-setting classifier grids, and combined with the original training seed under the same grid. Learned/random/pixel AdamW means: 27.11%/25.67%/28.00%; ridge: 25.67%/25.33%/30%. Published all seed outcomes, diagnostics, paired errors and provenance. The experiment-specific summarizer reproduced 36 saved classifiers without refitting and verified pixel equality/training-partition-only sampling. Fixed its initial assumption that per-run reports contain a seed field by deriving seeds from the parent report. Full suite: 62 tests; synthetic 3-step training and masking smoke checks passed. No official test evaluation or paid compute.
