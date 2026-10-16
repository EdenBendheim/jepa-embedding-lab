# Development log

## October 15, 2026

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
