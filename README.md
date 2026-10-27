# JEPA Embedding Lab

A small, reproducible representation-learning experiment: predict target embeddings from visible context, then test whether custom embeddings help a downstream task.

**Status — October 25, 2026:** the [predeclared amplitude comparison](experiments/SCALE-2026-10-22.md)
is complete. Three new .02 runs and pinned 1.0 controls share actual image order
and 29,472 trainable parameters. On 500 fresh validation rows, trained .02 / 1.0
AdamW accuracy is **25.53% / 26.51%** (−0.98 points). Matching random controls
score 25.38% / 26.60%; a broad learned-feature advantage remains unproven.
Frozen heads, full candidate audits and conservative exposure receipts now
replay the evaluation. The ledger retains 2,746 observed rows; official test
stays reserved. These are descriptive validation results, not test accuracy.
A [training-only diagnostic audit](experiments/STABILITY-2026-10-24.md) now
finds concentrated pooled variance and high student/teacher endpoint agreement,
with matching random controls. It adds no new accuracy or temporal-stability claim.
The [fixed-mask fitting-image audit](experiments/MASKED-CONTEXT-2026-10-25.md)
finds that full-image versus visible-only context still has concentrated pooled
variance. Four-mask sensitivity is a diagnostic, not a robustness or accuracy result.

## Why this experiment

The [October 12 confirmation protocol](experiments/CONFIRMATION-2026-10-12-PROTOCOL.md)
was committed before observing its scores. It excludes all 246 unique validation
rows from the two earlier pilot selections, keeps the fitted training rows and
24 selected classifiers fixed, and reserves the official test set. Fresh data
continues to favor the matching random/pixel baselines over the learned features.

The [October 11 pooling/mask protocol](experiments/POOLING-MASK-2026-10-11-PROTOCOL.md)
was committed before new scores; both fixed-weight pooling and the 223-step
4x4-target trajectory matched to historical image order are now completed.
The driver reproduces every saved classifier and paired error before producing
aggregate-only summaries; features, logs and weights remain ignored.

I want to explore the relationship between the input representation, masking geometry, and what an embedding retains. The first public experiment uses [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) and frozen-embedding image classification. Its small RGB images make bounded local iteration practical. Satellite/geospatial imagery remains a later direction because it connects to my research; no unreleased research data is used here.

The design reference is [I-JEPA](https://arxiv.org/abs/2301.08243) and its [official implementation](https://github.com/facebookresearch/ijepa). The current single-target-block sampler is a simplified experiment foundation, not a reproduction of the official multi-block training recipe.

## Working now

An opt-in `objective.context_variance` helper penalizes low **between-image**
variance after visible-token mean pooling. `JEPA.forward(..., return_context=True)`
returns those existing context tokens as a third output. Spatial variation alone
does not satisfy the penalty; mask variation can still contribute to it. The
historical prediction-only objective remains the default. This is a training
intervention foundation, not a measured quality improvement.

- Non-overlapping patch extraction with explicit positions and strict shape/finite-value validation.
- Seeded rectangular target blocks with context sampled outside the target.
- Separate context, target, and unused indices: the context encoder must never receive target pixels.
- Reproducible CLI mask inspection and tests covering partition integrity, preserved pixels, invalid shapes, and random-seed isolation.
- Context/target Transformer encoders and a masked-target predictor with learned patch positions. Context gathers visible raw patches before encoding; target encoding is frozen and uses no gradients.
- EMA target updates, training loss plus embedding standard-deviation/covariance diagnostics, and CPU checkpoint resume including optimizer and random-generator state.
- Official CIFAR-10 binary ingestion without pickle, a fixed RGB patch adapter, dataset fingerprints, and disjoint stratified train/validation indices while retaining the official test partition.
- A bounded RGB training smoke command that validates its manifest, samples training images only, and records loss/variance diagnostics without claiming downstream accuracy.
- A resumable public-data CPU trainer with atomic checkpoints, matching manifest/model/settings/runtime checks, optimizer and sampling/PyTorch RNG restoration, and split-versus-continuous reproducibility checks.
- Frozen context/target full-image embeddings with mean/max/mean_std pooling, matched random-encoder and raw-pixel baselines, and atomic feature artifacts bound to exact examples, pooling dimensions, data identity and checkpoint bytes.
- Saved-feature pooling comparisons with shared pixel classifiers, reproduced selected states and paired class errors; descriptive numerical/effective rank and cosine inspection.
- Configurable target-mask training with resume-bound geometry, patch counts, and legacy default-checkpoint compatibility; a declared real-data mask pilot with verified image-order equality.
- Disjoint balanced confirmation selection, validation-only extraction and no-refit saved-classifier scoring with preserved original normalization, feature-content checks and explicit retuning boundaries.

## Track validation exposure

Source-replayed validation exposure ledgers deduplicate nested pilot/confirmation
history into a flat dependency graph. `exposure.make_exposure_ledger` verifies
each source against its original selection protocol and derives the observed
row union; `exposure_summary` reports remaining per-class capacity without rows.
`exposure.make_ledger_confirmation` reserves version-three confirmation rows
against the complete replayed union. Supplied history must be complete: content
hashes cannot detect an experiment the caller leaves out.

Build local pinned history with `python -m jepa_lab.exposure ROOT --manifest
MANIFEST --source SELECTION_JSON FILE_SHA256 --output runs/exposure.json` (one
line; repeat `--source` for more files). Inspect/replay it using `--bundle
runs/exposure.json` instead of `--source`/`--output`. Original files must remain
available with exactly the pinned bytes. The bundle stores local source paths;
keep it under ignored `runs/`. Stdout contains aggregate capacity only. Invalid
sources cannot replace previous output or overwrite their own evidence.

Version-three selections embed a flat ledger plus the original fitting pilot.
Register a later observed selection with `make_exposure_ledger`; its original
ledger roots and source fingerprint are retained, without copying a deeper tree.
Validation replays each dependency in topological order, including the original
reservation recipe. Historical version-one/two documents keep their old hashes
and depth guard. Six synthetic later rounds retain all earlier rows and exact
fitting examples; malformed source dependencies fail closed. The ledger is
bounded to 128 unique sources, and exhausted per-class validation capacity is
an error.

Reserve later rows using `python -m jepa_lab.selection ROOT --manifest MANIFEST
--confirmation-of ORIGINAL_PILOT --exposure-bundle runs/exposure.json
--expected-ledger-sha256 PREDECLARED_HASH --validation-per-class 50 --seed SEED
--output runs/later-confirmation.json` (one line). The bundle's original files
are rechecked before selection; changed pins fail without replacing old output.
Do not mix `--exclude-pilot` with ledger mode. Version-three feature extraction
remains validation-only and cannot feed probe selection. Existing saved-classifier
and paired/kNN confirmation APIs accept it and keep normalization/head state
frozen. Self-contained selection replay checks declared evidence; use the pinned
bundle when rechecking original source file bytes is also required.

Extend history with `python -m jepa_lab.exposure ROOT --manifest MANIFEST
--source NEW_SELECTION FILE_SHA256 --extend-bundle runs/exposure.json
--expected-ledger-sha256 PREVIOUS_HASH --output runs/exposure-next.json` (one
line). Every original source file is retained and rechecked. Previous bundles
cannot be overwritten; keep the prior snapshot. Re-registering identical source
bytes is idempotent. Treat an unscored reservation as reserved, not observed;
register it after it has actually been used for evaluation or tuning.

[October 19 audit](experiments/EXPOSURE-2026-10-19.md): six pinned original source
files account for 2,246 observed validation rows, leaving 2,754 (275–277 per
class). Flat replay agrees with the independent historical union. A 500-row
reservation check excludes them all and remains unscored. This repairs the
history-management blocker without changing model results or official test use.

## Run the mask demo

Python 3.11+, no runtime dependencies:

```sh
export PYTHONPATH=src
python3 -m jepa_lab.demo --seed 29
python3 -m unittest discover -s tests -p test_foundation.py -v
```

`C` marks visible context, `T` marks prediction targets, and `.` marks unused patches. The command also prints indices and patch dimensions for inspection. All demo intensities are generated locally.

## Run the model and synthetic training check

The training extra pins PyTorch 2.8.0; the lightweight mask demo remains dependency-free. On Linux, install the CPU PyTorch wheel from its official index first if CPU-only training is intended.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[training]'
export PYTHONPATH=src
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m jepa_lab.train --steps 20 --checkpoint checkpoints/smoke.pt --output runs/smoke.json
.venv/bin/python -m jepa_lab.train --steps 3 --resume checkpoints/smoke.pt
```

The default model uses an 8x8 patch grid, four values per patch, width 32, two encoder blocks, and one predictor block. The CLI runs on CPU with one thread and locally generated spatial patterns. Every sample receives a deterministic mask. Checkpoints include model/configuration, optimizer, step count, and data/PyTorch RNG state; resume requires matching seed, batch size, learning rate, and momentum. `--steps` means additional steps. Tests compare a resumed run with an uninterrupted run in the same environment. Reproducibility across hardware or library versions is not guaranteed.

Checkpoints and run metrics are ignored by Git. No weights or private research inputs are committed. Small synthetic loss changes and nonzero embedding variance are execution diagnostics, not evidence of downstream accuracy or immunity to collapse. The public dataset/protocol below is selected; quality claims still require downstream evaluation.

## Prepare the first public dataset

Download the **binary version** from the [official CIFAR-10 page](https://www.cs.toronto.edu/~kriz/cifar.html). Its published archive MD5 is `c32a1d4ab5d03f1284b67883e8d87530`; verify the download and unpack it beneath ignored `data/`. Images are not redistributed in this repository. The loader requires the six official `.bin` files, validates record sizes/labels, and hashes their contents. No network request occurs inside the loader.

```sh
export PYTHONPATH=src
.venv/bin/python -m jepa_lab.cifar10 data/cifar-10-batches-bin \
  --manifest runs/cifar10-manifest.json
.venv/bin/python -m jepa_lab.cifar10_smoke data/cifar-10-batches-bin \
  --manifest runs/cifar10-manifest.json --steps 3 \
  --output runs/cifar10-adapter-smoke.json
```

The version-one protocol uses 45,000 official training images for JEPA/probe fitting, 5,000 for validation (500 per class), and all 10,000 official test images for final evaluation. SHA-256 ordering of original training indices with seed `20261003` makes the split independent of global RNG state or Python's shuffle implementation. The manifest preserves official split names, exact indices, batch hashes, transform configuration, and its own fingerprint. Training and test indices have separate namespaces; numeric overlap alone does not mean shared examples. The smoke command rebuilds the manifest from local files and rejects stale or edited splits.

The adapter accepts uint8 NCHW RGB images and applies the fixed `x / 127.5 - 1` transform. A 32x32 image becomes an 8x8 grid of 4x4 RGB patches (`patch_dim=48`); patches follow row-major grid order with channel-major pixels. Class labels are used for stratification and future probe evaluation, not the JEPA training objective. The three-step smoke run uses only training images, one CPU thread, and no paid API. Its command is capped at 50 steps / batch size 32; it is not the resumable long-run trainer.

See [the experiment protocol](experiments/CIFAR10.md) and [version-one provenance](experiments/cifar10-v1.json). The [October 7 pilot](experiments/PILOT-2026-10-07.md) reports validation-selected classification scores; official test accuracy remains unmeasured.

## Resume a public-data training pilot

```sh
export PYTHONPATH=src
.venv/bin/python -m jepa_lab.cifar10_train data/cifar-10-batches-bin \
  --manifest runs/cifar10-manifest.json --steps 20 \
  --checkpoint checkpoints/cifar10-pilot.pt --output runs/cifar10-pilot.json
.venv/bin/python -m jepa_lab.cifar10_train data/cifar-10-batches-bin \
  --manifest runs/cifar10-manifest.json --steps 3 \
  --resume checkpoints/cifar10-pilot.pt --checkpoint checkpoints/cifar10-pilot.pt \
  --output runs/cifar10-pilot-resumed.json
```

`--steps` is the number of **additional** optimizer steps, capped at 1,000 per invocation with batch size 1–32. Defaults are batch four, learning rate 0.001, EMA momentum 0.99, the manifest seed, and one CPU thread. This is a bounded local trainer, with no download, API request, paid compute, or automatic prolonged run. Reports record the start/completed step counts, exact sampled training indices, loss, gradient norm, and embedding variance/covariance for that invocation.

Use `--checkpoint-every 50` with `--checkpoint` to atomically save model,
optimizer, RNG and accumulated training diagnostics every 50 global steps.
An interrupted run resumes from the last durable step. The interval does not
alter the optimization recipe. Old checkpoints explicitly retain a missing
history prefix; no earlier diagnostics are invented.

Checkpoint version one binds state to the verified manifest fingerprint, model configuration, seed, batch size, learning rate, momentum, PyTorch version, and CPU thread count. Resume rejects mismatches before sampling. Model, optimizer, sample RNG, and PyTorch RNG are restored; mask seeds advance from the saved global step. Tests require a resumed run to match uninterrupted diagnostics, weights, optimizer, and RNG state in the same environment. This does not guarantee identical results across platforms or library versions; incompatible recorded runtimes are rejected.

Checkpoints load with `weights_only=True` on CPU and are written via temporary file plus atomic replacement. A failed save leaves the previous checkpoint intact. Without a periodic interval, saving occurs at the end of a successful invocation; an interruption loses only steps after the last durable save. Data/weights/run metrics remain ignored. The 20+3-step real-image pilot had finite losses/gradients but is too small to establish useful representations. Frozen features, classifier/encoder-seed checks and declared pooling/mask pilots now work. Fresh confirmation and frozen pretrained comparison are now measured below; the completed longer-training and width pilots are reported below.

## Finish a declared local training budget

`python -m jepa_lab.training_budget ROOT --manifest MANIFEST --checkpoint
checkpoints/budget.pt --total-steps 2223 --output runs/budget.json` runs bounded
chunks of at most 1,000 steps, with periodic recovery checkpoints. Retries reuse
completed steps, verify the complete journal and deterministic image order, and
reject changed recipes or a checkpoint beyond the target. `--reuse-only`
requires a completed cache and never trains. Total budget is capped at 10,000
steps; the CLI uses one CPU thread. Width/depth options support later controlled
architecture comparisons. No API or cloud compute is used.

## Extract frozen image features and baseline features

The real-data trainer also accepts `--target-rows`, `--target-cols` and
`--context-fraction`. Custom geometry is validated before image sampling,
recorded in `settings.mask`, and must match on resume. Version-one checkpoints
without that field mean exactly the historical 2x2 target / 0.75 context
fraction; default runs preserve that identity. Reports include visible, target
and unused patch counts. Context fractions apply to non-target patches, so
changing block size changes both prediction and visible-token budgets.

```sh
export PYTHONPATH=src
.venv/bin/python -m jepa_lab.embeddings data/cifar-10-batches-bin \
  --manifest runs/cifar10-manifest.json --checkpoint checkpoints/cifar10-pilot.pt \
  --partition train --limit 64 --output runs/features-checkpoint-train.pt
.venv/bin/python -m jepa_lab.embeddings data/cifar-10-batches-bin \
  --manifest runs/cifar10-manifest.json --representation random --seed 29 \
  --partition train --limit 64 --output runs/features-random-train.pt
.venv/bin/python -m jepa_lab.embeddings data/cifar-10-batches-bin \
  --manifest runs/cifar10-manifest.json --representation pixels \
  --partition train --limit 64 --output runs/features-pixels-train.pt
```

The context encoder is the default; `--encoder target` extracts from the EMA teacher. Both process all 64 patches without a mask or predictor, then apply the selected pooling rule (mean by default). Inference disables gradients and leaves weights unchanged; the API restores the model's prior training mode. Random baseline initialization preserves the caller's CPU RNG and records its seed/architecture. The current default random architecture matches the 32-dimensional pilot. For other model configurations, pass the matching `ModelConfig` to the extraction API before comparing; the matched comparison runner enforces architecture equality.

Raw-pixel features flatten NCHW values using the same fixed `x / 127.5 - 1` scaling, producing 3,072 dimensions. Labels accompany feature artifacts for classification probes but do not enter an encoder. Every representation uses the same first `--limit` indices from the selected fixed partition (default 64, cap 1,000, batch size 1–32). These bounded subsets are not guaranteed class-balanced and are not the complete experiment. Validation uses original training-file indices; test indices remain in the official test namespace. Extraction never fits normalization or a classifier on held-out examples.

Each local `.pt` artifact contains features, labels, exact indices/namespace, manifest identity, pooling/scaling/configuration, extraction settings, and a feature fingerprint. Checkpoint extraction also records the training step/settings and the SHA-256 of the exact bytes loaded. Mismatched manifests/checkpoints fail before image sampling or output replacement. Features and raw-pixel artifacts remain in ignored `runs/`; do not redistribute the images or commit artifacts. Reports contain metadata/diagnostics, not feature arrays. The real-image check extracted the same 64 training images in all three representations and another 64 validation images from the checkpoint, with finite features throughout. The balanced pilot comparison below fits probes and reports exploratory validation scores.

## Compare full-image pooling

The frozen `image_embeddings(..., pooling=...)` API supports `mean`, `max`, and
`mean_std` across all 64 spatial tokens. `mean_std` concatenates the mean and
population standard deviation, doubling the classifier's input dimension;
any comparison must disclose that capacity difference. Pooling fits no state,
uses no labels, and preserves encoder weights/mode. Both `embeddings` and
`compare` accept `--pooling mean|max|mean_std` (default `mean`). Artifact
verification binds the pooling description to the feature dimension; matched
random encoders use the same rule. Pixel features keep their fixed flattening.
Reusing a directory with a different requested encoder/pooling fails before fitting.

`pooling_compare.compare_pooling` compares 2–3 verified feature arms with
`mean` first. It fixes checkpoint bytes/age/training settings, ordered examples,
encoder choice, architecture and random seed. Each random baseline uses its
arm's pooling, while the exact pixel artifacts/classifiers are shared. It
records per-arm dimensions, all selected states and paired prediction errors.

The saved-feature CLI verifies every arm before fitting and never samples raw
images. Prepare checkpoint/random files in each arm directory with the matching
`embeddings --pooling` setting, using one checkpoint and random seed throughout.
Reuse one directory containing the exact selected pixel train/validation files:

```sh
.venv/bin/python -m jepa_lab.pooling_compare data/cifar-10-batches-bin \
  --manifest runs/cifar10-manifest.json --selection runs/pilot-selection-20261009.json \
  --pixel-dir runs/pilot-features-20261009 \
  --arm mean runs/pooling/mean --arm max runs/pooling/max \
  --probe-seeds 29 31 37 --ridge-alphas 0.01 0.1 1 10 \
  --probe-dir runs/pooling/probes --output runs/pooling/comparison.json
```

Add `--arm mean_std runs/pooling/mean_std` to include spatial variance. Default
classifier grids/budgets match the existing comparison CLI. This is a
validation-selected ablation; it does not evaluate the official test set.

## Inspect frozen-feature geometry

```sh
.venv/bin/python -m jepa_lab.feature_geometry data/cifar-10-batches-bin \
  runs/pilot-features-20261009/checkpoint-train.pt \
  --manifest runs/cifar10-manifest.json --selection runs/pilot-selection-20261009.json \
  --partition train --output runs/feature-geometry.json
```

This verifies the source artifact and reports row norms, zero-norm rows,
constant columns, population standard deviations, centered numerical rank,
covariance-spectrum entropy effective rank, and mean cosine between nonzero
rows. Zero-variance sets have effective rank zero; fewer than two nonzero rows
have no pair cosine (`null`). Eigenvalues below a float64 relative numerical
tolerance are excluded. No transform is fitted, no images are sampled, and no
arrays are included in the report. These are descriptive diagnostics, not
evidence of useful representations or a guarantee against collapse.

## Implemented model contract

1. A context encoder receives only visible patches and their positions.
2. A stop-gradient target encoder encodes the full input; target positions select prediction targets.
3. A predictor combines context embeddings with target positions to predict target representations.
4. An exponential moving average updates the target encoder from the context encoder.
5. Log loss **and** representation variance/covariance to diagnose collapse; evaluate frozen representations on a held-out downstream task.

The model uses PyTorch through an explicit optional training dependency. Baselines and ablations will share fixed splits, seeds, compute accounting, and an evaluation protocol. Improved latent prediction loss is not itself proof of useful embeddings.

For a fire-specific dataset, split by distinct fire event, time, and geography before sampling windows. A fire progression prediction head would be a separate experiment; latent prediction alone does not output fire polygons.

See [PLAN.md](PLAN.md) for the staged experiment and [DEVLOG.md](DEVLOG.md) for completed work.

## Choose a shared balanced pilot

```sh
export PYTHONPATH=src
.venv/bin/python -m jepa_lab.selection data/cifar-10-batches-bin \
  --manifest runs/cifar10-manifest.json --output runs/pilot-selection.json \
  --train-per-class 10 --validation-per-class 5
.venv/bin/python -m jepa_lab.embeddings data/cifar-10-batches-bin \
  --manifest runs/cifar10-manifest.json --selection runs/pilot-selection.json \
  --representation pixels --partition train --output runs/pilot-pixels-train.pt
```

The pilot uses deterministic within-class SHA-256 ordering inside the already locked train/validation partitions. Defaults select 100 training and 50 validation examples, balanced across all ten classes. The selection records exact indices, seed, per-class counts and a fingerprint bound to the full dataset manifest. It never includes test. Counts are bounded at 100 per class.

All feature representations can pass the same `--selection`; extraction rebuilds it against current labels and data identity before sampling. It uses the complete selected partition and rejects `--limit` or test extraction with a pilot selection. The earlier first-index wiring-check path remains available without `--selection`. Pilot results describe this small subset only; the comparison below evaluates it without using test features.

## Verify a feature artifact before evaluation

```sh
.venv/bin/python -m jepa_lab.features data/cifar-10-batches-bin runs/pilot-pixels-train.pt \
  --manifest runs/cifar10-manifest.json --selection runs/pilot-selection.json --partition train
```

The loader uses `weights_only=True` on bounded local files and verifies exact selection/manifest identity, train/validation namespace, indices, official labels, feature shape/dtype/finiteness/content hash, normalization, encoder architecture, checkpoint/random provenance, and recorded PyTorch runtime. Reports omit arrays and retain the hash of the exact artifact bytes loaded. A checksum checks consistency; it does not authenticate the producer or prove that an encoder was trained well. Use artifacts generated locally by the extraction command. Test artifacts are excluded from this pilot evaluation loader.

## Fit and select a frozen linear probe

`fit_probe(train_features, train_labels, ProbeConfig(...))` fits a ten-class, full-batch AdamW linear classifier on detached CPU feature copies. It learns population mean/std from training rows only; dimensions with std below `1e-6` use scale one. Weight decay applies to classifier weights, not its bias. Source encoder features and caller RNG remain unchanged. Runs allow 1–1,000 steps on at most 1,000x4,096 features.

`select_probe(train_features, train_labels, validation_features, validation_labels, candidates)` considers up to eight distinct learning-rate/weight-decay settings with the same seed and step budget. Selection uses highest validation accuracy, then lowest cross-entropy, then declared order. Reports include counts, per-class outcomes, a confusion matrix, and normalization/optimizer rules. No test input enters fitting or selection. Same-environment reproducibility is tested; the classifier/encoder-seed pilots below report descriptive variation. This probe alone measures tuning performance; the separate confirmation experiments assess fixed classifiers on fresh rows. Official test uncertainty remains unmeasured.

## Run a matched pilot comparison

```sh
export PYTHONPATH=src
.venv/bin/python -m jepa_lab.compare data/cifar-10-batches-bin \
  --manifest runs/cifar10-manifest.json --selection runs/pilot-selection.json \
  --checkpoint checkpoints/cifar10-pilot.pt --feature-dir runs/pilot-features \
  --steps 200 --output runs/pilot-comparison.json
```

With `--checkpoint`, the runner extracts six local artifacts using the same selection and a random architecture copied from the checkpoint. Omit it to verify/reuse existing `checkpoint-train.pt`, `checkpoint-validation.pt`, `random-train.pt`, `random-validation.pt`, `pixels-train.pt`, and `pixels-validation.pt`. All six artifacts and cross-representation example/label/configuration checks pass before any probe fitting.

The default grid has eight candidates: learning rates `0.0001, 0.001, 0.01, 0.05` crossed with weight decay `0.01, 0.1`; each uses 200 full-batch steps and seed 29. Every representation receives the same grid/budget and selects its settings on validation. Reports retain source-artifact hashes, the grid and all candidate outcomes, selected settings, class counts/confusion, runtime and timing. Atomic report replacement preserves an older report on write failure. Features/weights remain ignored; reports omit arrays. Pass `--probe-dir runs/pilot-probes` to save each selected classifier and normalization. Reports retain the state/source/file hashes and paths; content-addressed filenames preserve earlier reports' classifiers. Repeated identical states are verified and reused. Without this option, weights remain transient. A report-write failure can leave unreferenced local probe files, but does not overwrite earlier classifiers.

See [the October 7 results and failure analysis](experiments/PILOT-2026-10-07.md). The grid was expanded after an initial validation diagnostic exposed pixel overfitting; this is exploratory model selection with one seed and 50 validation examples. There is no independent confirmation or official test score. The early checkpoint has not demonstrated an advantage over the random baseline.

## Reserve fresh confirmation examples

The selector can preserve the fitted pilot's training examples while excluding
validation examples from that pilot and additional historical selections:

```sh
.venv/bin/python -m jepa_lab.selection data/cifar-10-batches-bin \
  --manifest runs/cifar10-manifest.json --confirmation-of runs/pilot-selection-20261009.json \
  --exclude-pilot runs/pilot-selection-20261007.json --validation-per-class 50 \
  --seed 20261012 --output runs/confirmation-selection-20261012.json
```

Pass the resulting `--selection` to `embeddings` with `--partition validation`.
It rejects training/test extraction; the feature loader marks these rows as
confirmation data, and fitting/selection/persistence runners reject them.
Version-one pilot files and classifiers remain valid.

`confirmation.score_confirmation(probe_path, train, tuning_validation, fresh,
selection)` verifies exact source/representation identity and disjoint fresh
rows, restores the selected classifier against its original artifacts, then
scores the new rows with its unchanged training-only normalization and weights.
It supports AdamW and ridge, fits/selects nothing and emits aggregate metrics.
The no-refit CLI below works; official test remains reserved. Once used for
new tuning, these examples no longer provide an untuned confirmation check.

```sh
.venv/bin/python -m jepa_lab.confirmation data/cifar-10-batches-bin PROBE_PATH \
  --manifest runs/cifar10-manifest.json --reference-selection runs/pilot-selection-20261009.json \
  --confirmation-selection runs/confirmation-selection-20261012.json \
  --train-features TRAIN_FEATURES --tuning-features OLD_VALIDATION_FEATURES \
  --confirmation-features FRESH_VALIDATION_FEATURES --output runs/confirmation-score.json
```

Use original feature paths and the selected classifier path from the historical
comparison report. Dataset/selection/features/state are revalidated before
scoring; invalid inputs leave an existing output report intact. The workflow
neither samples images nor fits a classifier. Fresh feature extraction is a
separate, validation-only step.

## Persist a fitted linear probe

`jepa_lab.probe_artifacts.save_probe(path, probe, train, validation)` saves a fitted classifier and its training-only normalization atomically. `load_probe(path, train, validation)` restores it using CPU `weights_only=True`, checks the exact source-artifact/manifest/selection identities, tensor dimensions/dtypes/finiteness, positive normalization scales, configuration and runtime, and reproduces its saved train/validation scores. A failed replacement preserves the previous file. Checksums detect inconsistency, not producer authenticity; use locally generated artifacts. Probe files stay in ignored `runs/` or `checkpoints/`. Test evaluation remains reserved. The matched comparison CLI accepts `--probe-dir` to persist selected probes after all six feature artifacts pass verification and fairness checks.

## Recheck a saved classifier without refitting

Use the `selected_probe_artifact.path` recorded by the comparison report as the probe argument:

```sh
.venv/bin/python -m jepa_lab.probe_score data/cifar-10-batches-bin \
  runs/pilot-probes/checkpoint-STATE_HASH.pt \
  --manifest runs/cifar10-manifest.json --selection runs/pilot-selection.json \
  --train-features runs/pilot-features/checkpoint-train.pt \
  --validation-features runs/pilot-features/checkpoint-validation.pt \
  --output runs/rechecked-probe.json
```

Replace `STATE_HASH` with the actual report path. The command verifies current dataset identity, both feature artifacts, and the saved classifier before reproducing train/validation scores. It neither refits nor samples raw images nor evaluates the official test partition. This is a reproducibility check on the original validation result, not new evaluation evidence.

## Regularized ridge baseline

`fit_ridge(features, labels, RidgeConfig(alpha))` fits ten one-hot targets using training-only normalization and a closed-form float64 ridge solve. The objective is mean squared error plus `alpha * ||W||²`; the intercept is unpenalized. It uses a smaller row-space dual solve for high-dimensional pixel features and a column-space primal solve for narrow embeddings. Returned frozen coefficients are float32 and use the same bounded inputs/score contract as AdamW probes.

`select_ridge(train, train_labels, validation, validation_labels, alphas)` selects up to eight distinct positive penalties on validation accuracy, then MSE, then declared order. It never fits statistics/coefficients on validation. Scores are linear least-squares outputs, not calibrated probabilities. Saved-probe persistence also supports ridge state. Matched ridge results are reported in the pilot, confirmation and pretrained experiments.

## Repeat probe initialization and compare a ridge baseline

The comparison CLI accepts mutually exclusive `--probe-seed 29` or `--probe-seeds 29 31 37`. It verifies the same frozen features once, gives every representation the identical grid/step budget for each classifier seed, and retains all per-seed selections plus mean/sample standard deviation. `--ridge-alphas 0.01 0.1 1 10` adds a matched deterministic ridge comparison, fitted once per representation and stored separately. `--probe-dir` also persists selected ridge classifiers. Invalid/duplicate seeds or penalties fail before fitting.

This measures classifier-initialization sensitivity only. It does not vary encoder training, random encoder initialization, pilot examples or data splits, and its validation-selected scores are not test results or independent confirmation. See the [predeclared October 9 larger pilot](experiments/PILOT-2026-10-09-PROTOCOL.md).

Comparison reports also include `paired_validation`: checkpoint predictions versus random and pixels, separately for AdamW and ridge. Recoveries, regressions, per-class changes and confusion deltas use the same verified ordered validation examples and the already-selected classifiers; no extra fits are performed. Counts are descriptive and validation-selected.

## Audit paired saved classifiers

`python -m jepa_lab.probe_compare ROOT --manifest MANIFEST --selection SELECTION --baseline TRAIN_FEATURES VALIDATION_FEATURES PROBE --candidate TRAIN_FEATURES VALIDATION_FEATURES PROBE --output runs/paired-audit.json` verifies both classifiers and exact ordered sources, then reports paired errors without refitting or sampling images. Feature dimensions may differ between representations. Each saved classifier retains its original training-only normalization. The audit can compare checkpoint ages or a learned encoder against a baseline; it provides no new evaluation evidence.

## Controlled checkpoint-age comparison

`compare_progression(arms, candidates, seeds, probe_dir=..., ridge_alphas=...)` compares 2–4 increasing checkpoint ages with identical architecture/training settings and exact shared baseline artifacts. Every arm is checked before fitting. Random/pixel classifiers are fitted once per classifier seed, and ridge once total; each checkpoint receives the same grid and budget. Reports retain every selected classifier, paired age/baseline errors, and classifier-seed summaries. Matching metadata does not establish checkpoint ancestry, and this is a validation-selected comparison from one encoder training seed.

The saved-feature CLI leaves source artifacts untouched and does not sample images:

```sh
.venv/bin/python -m jepa_lab.progression data/cifar-10-batches-bin \
  --manifest runs/cifar10-manifest.json --selection runs/pilot-selection-20261009.json \
  --baseline-dir runs/pilot-features-20261009 \
  --arm early runs/pilot-features-early-20261010 \
  --arm later runs/pilot-features-20261009 \
  --probe-seeds 29 31 37 --ridge-alphas 0.01 0.1 1 10 \
  --probe-dir runs/progression-probes-20261010 --output runs/progression-20261010.json
```

Each arm directory must contain verified `checkpoint-train.pt` and `checkpoint-validation.pt`; the shared directory supplies `random-*.pt` and `pixels-*.pt`. Use the existing embedding extractor to prepare missing features with the same selection. The [October 10 protocol](experiments/PILOT-2026-10-10-PROTOCOL.md) fixes the age comparison and two additional encoder-training seeds before the new runs.

## Frozen pretrained transfer reference

Confirmation exclusions can also include an earlier confirmation selection.
Its observed rows and all earlier tuning ancestors are excluded, so a second
confirmation round cannot quietly reuse the first round's examples. Ancestry
is bounded to three earlier rounds; original fitting rows remain fixed.

Install the optional `.[pretrained]` extra (torchvision 0.23.0). The extractor
requires the local official `resnet18-f37072fd.pth` weight file, verifies its
exact SHA-256, loads with `weights_only=True`, freezes the encoder and removes
the classification head. It never downloads weights inside extraction.

The named weight recipe uses deterministic tensor resize to 256, center crop
224 and ImageNet channel normalization on CPU uint8 RGB inputs. Frozen global
average pooling produces 512 features. Modes, buffers, gradients and caller
RNG remain unchanged. The artifact verifier binds weights, transform, runtime,
feature dimensions and exact selected examples.

```sh
.venv/bin/python -m jepa_lab.pretrained_extract data/cifar-10-batches-bin \
  --manifest runs/cifar10-manifest.json --selection runs/pilot-selection-20261009.json \
  --weights checkpoints/resnet18-f37072fd.pth --partition train \
  --output runs/pretrained/pretrained-train.pt
```

Repeat for validation. This is bounded to 16 images per batch and 1,000 per
partition, uses no official test images, and fits no classifier. To compare
existing features with identical downstream grids/seeds:

```sh
.venv/bin/python -m jepa_lab.transfer_compare data/cifar-10-batches-bin \
  --manifest runs/cifar10-manifest.json --selection runs/pilot-selection-20261009.json \
  --feature-dir runs/pilot-features-20261009 --pretrained-dir runs/pretrained \
  --probe-dir runs/transfer-probes --output runs/transfer-comparison.json
```

The [October 13 protocol](experiments/PRETRAINED-2026-10-13-PROTOCOL.md) fixes
the historical sources, eight-setting/200-step AdamW grid, seeds 29/31/37 and
ridge penalties. Execute it with `PYTHONPATH=src .venv/bin/python
experiments/run_pretrained_20261013.py`; `--reuse-only` reproduces saved states,
paired errors and aggregates without extracting or refitting. Losing candidate
states were not saved. Failed verification preserves an earlier summary.

[Results](experiments/PRETRAINED-2026-10-13.md) use already-observed validation
rows. ImageNet supervision, model size, 224-pixel input and 512-D features
differ from the compact JEPA. Scores are not test accuracy or proof of useful
JEPA representations. All images, weights, features and probes remain ignored.

## Longer mask training and fresh confirmation

Execute `PYTHONPATH=src .venv/bin/python experiments/run_budget_20261014.py`
for the declared six local training runs. Add `--evaluate` to extract matched
features, select classifiers on the original tuning rows, lock all selected
states, then score fresh confirmation. `--evaluate --reuse-only` verifies
completed journals, saved states, paired errors and aggregates without training,
image reads, extraction or refitting. Failed checks preserve the prior summary.
The protocol fixes all budgets/seeds and exact source identities; local
checkpoints/features/probes/journals remain ignored.

## Inspect training dynamics

Install `.[plots]`, then run `PYTHONPATH=src .venv/bin/python
experiments/plot_budget_20261014.py` with the local ignored journals present.
Their exact bytes and complete budget/order must match the published source
hashes before rendering. The [fixed-window plot](experiments/budget-2026-10-14-dynamics.svg)
and [aggregate points](experiments/budget-2026-10-14-dynamics.json) contain no
per-image records. Ribbons are the range across three seeds, not confidence
intervals. Latent loss and token spread are diagnostics; they do not establish
useful embeddings or rule out pooled-feature collapse.

## Frozen cosine-neighbor baseline

`neighbors.fit_neighbors(features, labels, NeighborConfig(k))` stores a detached
training-only standardized, L2-normalized index. Queries use the same statistics
and uniform cosine-neighbor votes. Equal similarities retain fitting-row order;
vote ties choose the smallest class ID. Zero vectors remain zero. Bounded to
1,000 rows / 4,096 dimensions and 100 neighbors. No classifier optimization or
encoder gradients are involved. Neighborhood quality still needs evaluation.

`python -m jepa_lab.neighbor_cli ROOT --manifest MANIFEST --selection PILOT
--train-features TRAIN --validation-features TUNING --k 1 5 15
--save-index runs/neighbors.pt --output runs/neighbors.json` selects on tuning
accuracy, breaking ties by declared order. Use `--restore-index` instead of
`--save-index`/`--k` to reproduce a fixed index without fitting or raw-image reads.
Earlier index files are protected from replacement. Neighbor vectors and labels
are local artifacts and must remain beneath ignored paths.

Both `cifar10_train` and `training_budget` accept `--embedding-dim`, `--heads`,
`--encoder-depth` and `--predictor-depth`. The recorded architecture must match
on resume. Local RGB training limits width to 8–128, heads to eight, encoder
blocks to four and predictor blocks to two. `model_size.model_size(config)`
reports exact parameter groups and a persistent FP32 state-memory estimate
that excludes activations and allocation overhead. Equal steps are not equal compute.

`neighbor_confirmation.score_neighbor_confirmation(path, train, tuning, fresh,
selection)` restores the already-selected index and verifies disjoint reserved
rows using the same source/ancestry checks as linear probes. It never changes
normalization, neighbor count or fitting vectors. These are validation
confirmation scores; the official test partition remains reserved.

## Controlled width and neighborhood experiment

Run `PYTHONPATH=src .venv/bin/python experiments/run_width_20261015.py` for
the committed fixed-budget width runs. Existing verified width-32 checkpoints
are reused; six wider models train from scratch. Add `--evaluate` for shared
pixel probes, exact-initialization random controls, neighbor-count selection
and frozen confirmation. `--evaluate --reuse-only` forbids missing-state fitting
or extraction and rechecks saved states, reported neighbor candidates and
aggregate scores. Failed checks preserve the previous summary.

The [protocol](experiments/WIDTH-2026-10-15-PROTOCOL.md) fixes exact source hashes,
rows, budgets and candidate grids before scores. Only aggregate outputs are
public; raw data, training journals, feature arrays, neighbor vectors and
classifier/model artifacts remain ignored. Width changes compute and capacity.

### Inspect frozen neighborhoods locally

`python -m jepa_lab.explorer ROOT --manifest MANIFEST.json --selection PILOT.json
--train-features TRAIN.pt --validation-features VALIDATION.pt --index INDEX.pt
--output runs/neighborhoods.json` (write as one shell command).

This restores an existing source-bound kNN index without fitting or choosing k.
It reports official row IDs, labels, ranked cosines and source fingerprints for
up to 100 already observed tuning rows (`--query-index ID ...`, default first 20).
It exports no images or feature vectors and refuses confirmation selections.
Inspection is descriptive, not fresh evaluation; the official test stays reserved.
Keep reports under ignored `runs/`.

Render a report as an offline interactive viewer:
`python -m jepa_lab.explorer_html runs/neighborhoods.json --output runs/neighborhoods.html`.
Open the HTML locally, choose a validation row and inspect its ranked training
neighbors, labels, cosine scores and source fingerprints. It needs no server,
network access or extra packages. It deliberately contains no image thumbnails.
Treat labels as a diagnostic aid, not proof of representation quality.

Compare two to four representations on the same ordered queries:

```sh
python -m jepa_lab.explorer_compare data/cifar-10-batches-bin \
  --manifest runs/cifar10-manifest.json --selection runs/pilot-selection-20261009.json \
  --arm learned LEARNED_TRAIN.pt LEARNED_VALIDATION.pt LEARNED_INDEX.pt \
  --arm random RANDOM_TRAIN.pt RANDOM_VALIDATION.pt RANDOM_INDEX.pt \
  --arm pixels PIXELS_TRAIN.pt PIXELS_VALIDATION.pt PIXELS_INDEX.pt \
  --output runs/explorer/comparison.json --html runs/explorer/comparison.html
```

Replace the uppercase artifact placeholders with existing verified files. The
viewer switches representations and queries; comparison shows shared and distinct
training neighbors. All arms must match split identity, row order and labels.
Overlap uses a common depth (default the smallest saved k; override with
`--depth K` within that bound), while each table retains its own saved k.
Neither overlap nor cosine magnitude across representations establishes quality.
No classifier is fitted or retuned. Renderer recomputes comparison metrics before
writing HTML and rejects inconsistent reports.

October 16 local check: restored the October 15 width-32/seed-20261003 learned,
exact-initialization random and pixel indices on the first 20 already observed
tuning rows. Their common depth was one. Learned/random top-one overlap was
25%; learned/pixel and random/pixel were each 15%. These are descriptive counts
on a small ordered subset, not accuracy or independent evidence of improvement.
The report and HTML remain ignored under `runs/explorer-20261016/`; no images,
features or weights are published. Position/input-feature experiments and
reserved official-test evaluation remain future work.

### Fixed spatial-position foundation

`ModelConfig(position_scale=...)` multiplies positional signals in the context,
EMA target and predictor at use time. It accepts finite nonnegative values;
zero removes the positional signal. The default `1.0` preserves historical
math and serialized identities. Scaling does not change initialized weights,
RNG consumption or parameter count. This enables an amplitude ablation; it
does not establish a better representation.

`scale_compare.compare_scales` accepts two or three increasing arms named
`scale0.02`, `scale1.0`, etc. It requires equal non-scale architecture, training
settings, steps, examples and matching random seeds, with shared pixel artifacts.
Saved AdamW/ridge heads are checked by `audit_scales` without refitting.
Changing position mode and amplitude together is refused as a confounded
comparison. Metadata checks do not replace actual training-journal checks.

The [October 20 protocol](experiments/SCALE-2026-10-20-PROTOCOL.md) declares
fixed-position multipliers .02 versus 1.0, matched seeds/budgets and frozen
confirmation. Run `PYTHONPATH=src .venv/bin/python
experiments/prepare_scale_20261020.py` on one line to recheck pinned sources and
recreate its unscored reservation. It reuses the October 19 unscored check;
the reservation was initially unscored. [October 22 execution](experiments/SCALE-2026-10-22.md)
completed training and frozen evaluation, so these rows are now observed. Use
the extended local exposure bundle for any later reservation.

`positions.sincos_2d(rows, cols, width)` defines a deterministic row-major
sin/cos encoding with separate row/column channels and frequencies
`10000**(-j/(width/4))`. It consumes no RNG and returns independent float32
tensors. Width must be divisible by four. This is an experimental alternative,
not an asserted reproduction of another model's positional recipe.

`ModelConfig(position_encoding="sincos2d")` uses fixed buffers in context,
EMA target and predictor. The default remains learned positions, serialized
with the historical default keys by `model.config_dict`. The alternative is
recorded explicitly in training checkpoints and feature provenance; resume
rejects a changed mode. Both branches consume the same positional initializer
RNG draws before fixed values replace parameters, preserving all other initial
weights for matched seeds. Fixed buffers reduce trainable capacity; model-size
reports count their persistent memory separately. Equal steps still do not mean
equal parameter counts or compute.

Public-data training and budget CLIs accept `--position-encoding learned|sincos2d`.
Both also accept `--position-scale FLOAT` (default 1). Nondefault scales are
bound to checkpoint identity and must match on resume. Random feature extraction
accepts `--position-scale` to match its trained control; checkpoint extraction
reads the saved scale and rejects a command-line override, including an explicit
default. Example: add `--position-encoding sincos2d --position-scale 0.02` to
training and matching random extraction. This command is an execution option,
not evidence that 0.02 performs better.
Random feature extraction accepts the same explicit choice. Checkpoint extraction
uses the saved mode and refuses an override. A changed mode cannot resume an
existing training run. Existing commands keep their learned-position default.

`position_compare.compare_positions` compares learned/fixed position arms at
identical width/depth, update/mask/pooling settings, examples, classifier grids
and seeds, with each arm's exact-initialization random encoder. Pixel classifiers
are shared. It persists selected heads and `audit_positions` reproduces saved
states, paired errors and aggregate scores without refitting. The shared fitting
implementation preserves the historical width-report schema. These checks do not
establish matched image order; experiment drivers must verify training journals.

`confirmation_compare.compare_confirmed_probes` restores two selected classifiers
against their original sources and compares predictions on identical verified
fresh rows. It reports paired recoveries/regressions/class changes without
refitting, changing normalization or retuning. CLI: `python -m
jepa_lab.confirmation_compare ROOT --manifest MANIFEST --reference-selection PILOT
--confirmation-selection FRESH_SELECTION --baseline TRAIN TUNING FRESH PROBE
--candidate TRAIN TUNING FRESH PROBE --output runs/paired-fresh.json`. This is
paired descriptive confirmation, not a new training-seed uncertainty estimate.

The [October 18 prospective position protocol](experiments/POSITION-2026-10-18-PROTOCOL.md)
fixes width, seeds, budgets, old tuning rows, matched initial weights and new
confirmation exclusions before any fixed-position scores. Training:
`PYTHONPATH=src .venv/bin/python experiments/run_position_20261018.py`.
It reuses three pinned learned-position checkpoints, trains three fixed-position
runs and verifies actual image order. `--reuse-only` cannot train missing states.

Add `--evaluate` to the position driver to verify/extract features, fit shared
pixel and matched learned/random heads on original tuning rows, save/audit and
lock every selected classifier/index, then extract fresh features for frozen
confirmation. It also reports paired trained-encoder recoveries/regressions for
each matched encoder/classifier seed. `--evaluate --reuse-only` reproduces the
complete report with training, image reads, extraction and fitting disabled.
Changed caches fail before replacing an earlier summary.


[October 18 results](experiments/POSITION-2026-10-18.md): fixed positions improved
the predeclared trained-encoder contrast by 1.58 points under AdamW, but the
matching random fixed-position control was only 0.11 points behind. Pixel kNN
remained higher. Cache replay reproduced 60 linear artifacts (52 unique) and
13 indices without fitting or image reads. These are descriptive fresh-validation
results, not official test evidence or a general JEPA advantage.

## Execute the declared amplitude training

`PYTHONPATH=src .venv/bin/python -m jepa_lab.scale_run` verifies the October 20
source pins and reservation, trains the .02 arm in bounded local chunks, and
reuses the pinned scale-one checkpoints. Each seed must have identical actual
image-order journals and equal trainable capacity across scales. `--reuse-only`
requires completed caches and performs no training. Add `--evaluate` to select and audit heads on the old tuning rows, lock them,
register the reservation as conservatively consumed, and score fresh rows
without refitting. `--evaluate --reuse-only` requires all caches and replays
without image reads, extraction, training or fitting. An interrupted evaluation
retains `observed-exposure.json` and `observation-receipt.json`; use that complete
source-pinned history when reserving later rows. Keep these files local.

`python -m jepa_lab.scale_audit --output runs/scale-training-audit.json`
replays the completed journals without image reads or training and exports
100-step aggregate diagnostic windows, checkpoint/order hashes and capacity.
[October 22 training audit](experiments/scale-2026-10-22-training.json) records
three new 2,223-update .02 runs and three pinned scale-one controls, with equal
29,472 trainable parameters and matched actual image order. Token spread and
loss are diagnostics only; downstream quality requires frozen evaluation.

[October 22 amplitude results](experiments/SCALE-2026-10-22.md) report the
negative primary contrast and all random/pixel, ridge and neighbor controls.
The .02 signal did not improve AdamW; the real no-fit replay and check outcomes
are recorded in the development log.

## Inspect training-only student and teacher diagnostics

```sh
PYTHONPATH=src .venv/bin/python -m jepa_lab.stability_audit \
  --output runs/stability-20261024/audit.json
PYTHONPATH=src .venv/bin/python -m jepa_lab.stability_audit \
  --output runs/stability-20261024/audit.json --reuse-only
```

This inspects completed amplitude checkpoints on the original 500 fitting
images. It compares full-image context/EMA-target tokens, matching random
encoders and shared pixel geometry. Population variance is decomposed into
within-image spatial variation and between-image pooled variation. Aligned
pooled RMSE, cosine and centered linear CKA complement named parameter gaps.
These are exploratory endpoint measurements, not temporal stability or accuracy.
A constant pooled representation has undefined CKA; undefined ratios are JSON null.

No models/classifiers are fitted, no validation images are read and official
test stays reserved. Cached replay checks original source pins, dataset hashes,
complete training journals, checkpoint bytes, code, runtime and report checksum
without reading images or recalculating diagnostics. The saved report describes
the original computation; stdout separately states whether this invocation read
images. Changed caches fail rather than being replaced. Checksums detect
accidental changes, not a forged report. Keep the cache outside the scale-run
state/data/source directories. Only aggregate reports belong in public commits.

The `masked_context.masked_context_report` API compares full-image means with
fixed visible-subset pooling and visible-only re-encoding. It uses 2–8 fixed
mask seeds shared across images and reports aggregate view sensitivity. Subset
pooling retains hidden-image attention and is only a diagnostic control; the
re-encoded arm gathers raw visible patches before attention. It fits nothing
and preserves model modes, weights and RNG. These statistics do not establish
robustness or downstream accuracy.

Add `--masked-context` to `jepa_lab.stability_audit` with a **new output path**
(for example `runs/masked-context-20261025/audit.json`) to inspect the original
fitting images at every completed scale/seed checkpoint and matching random
initialization. Four fixed seeds (29, 31, 37, 41) share each run's declared
target geometry/context fraction. The cache identity distinguishes this mode;
`--masked-context --reuse-only` must read no images or recompute diagnostics.
Original full-image caches remain separate and are never overwritten.
