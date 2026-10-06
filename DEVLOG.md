# Development log

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
