# Longer training and repeated masks — October 14, 2026

Declared before training the new models or observing their scores. Earlier
223-step models did not demonstrate a reliable advantage over random/pixel
features, and the observed October 12 confirmation rows cannot serve as fresh
confirmation again.

## Fixed training and evaluation

- Six models from scratch: target blocks 2x2 and 4x4, each with seeds 20261003,
  20261010 and 20261011. Pair these with random-encoder seeds 29, 31 and 37.
- Each gets 2,223 optimizer steps: nearly ten times the earlier 223-step budget.
  Batch four, AdamW lr 0.001, default weight decay 0.01, EMA 0.99, gradient clip
  1.0, 0.75 context fraction. Width 32, two encoder blocks, one predictor block.
  All three seeds run both masks on exactly the same ordered image draws.
- One CPU thread, chunks at most 1,000 steps, recovery save every 50 steps.
  Complete journals retain actual indices; all draws must belong to the fixed
  training partition. Checkpoints and journals remain local and ignored.
- Full-image context embeddings, mean pooling. Same historical balanced
  500 training / 200 tuning rows for every representation. Three classifier
  seeds 29/31/37, 200 steps, the existing eight-setting lr/decay grid and ridge
  alphas 0.01/0.1/1/10. Validation alone selects each classifier. No model or
  mask winner is selected by new confirmation results.
- Reserve 500 balanced new validation examples with seed 20261014, excluding
  the 246 old tuning rows and all 500 October 12 confirmation rows. Preserve
  the original 500 fitted training rows. This selection is fingerprinted in
  `budget-sources-20261014.json` before feature extraction.
- Once all classifiers are selected, score every saved state on that reserved
  subset without changing normalization, refitting or retuning. Reproduce old
  scores and paired errors first. Keep the official 10,000 test rows reserved.
- Report each encoder seed separately, average classifier seeds within each
  encoder, then average the three encoder means. Seed SD is descriptive;
  shared data and classifier seeds are not independent datasets.

## Limits

Equal steps/image order do not mean equal compute or visible/predicted token
counts between masks. Training is still small: 8,892 image draws per model,
with repetition, from 45,000 training images. No augmentation, larger model,
schedule or official I-JEPA multi-block recipe is added. This is an isolated
budget/mask experiment, not a reproduction or state-of-the-art claim. The new
confirmation examples become observed after this run and must not be reused
as untuned confirmation for later choices. Public outputs contain only metrics,
configuration and fingerprints; no images/features/weights/probe artifacts.
