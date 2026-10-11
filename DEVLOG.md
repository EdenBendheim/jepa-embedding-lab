# Development log

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
