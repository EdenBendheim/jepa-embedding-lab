# Context-variance intervention — prospective recipe, October 27

The October 24/25 fitting-image diagnostics found concentrated pooled variance.
This motivates testing one objective intervention. It does not establish that
raising variance improves downstream representations. Existing targets already
have layer normalization; adding it again would not be a new intervention.

## Fixed contrast

- Historical Smooth L1 prediction objective versus the same objective plus
  **0.1** times a visible-context image-mean variance hinge with **floor 1.0**.
- For context tokens h[b,t,d], pool z[b,d] = mean_t h[b,t,d]. The penalty is
  mean_d max(0, 1 - sqrt(population_variance_b(z) + 0.0001)). Teacher targets,
  target stop-gradient and EMA behavior remain unchanged. No label enters it.
- Width 32, four heads, encoder/predictor depth 2/1; fixed sin/cos position scale
  1.0; 2x2 target, .75 visible-complement fraction; batch four; AdamW LR .001,
  EMA .99; seeds 20261003/20261010/20261011; 2,223 updates per arm. Start both
  arms from scratch at each seed and verify identical initial tensors, parameter
  counts, masks and actual image-order journals. Equal updates are not equal
  compute: the enabled arm also computes pooled variance.
- Freeze/extract full-image context means on the same historical 500/200
  fitting/tuning examples. Use one shared exact-initialization random control
  per encoder seed and shared pixel features, since architecture is unchanged.
- Reuse the existing selection budget: AdamW LR .0001/.001/.01/.05 x decay
  .01/.1, 200 steps, classifier seeds 29/31/37; ridge .01/.1/1/10; cosine kNN
  1/5/15. This reuses known tuning data, not an independent validation sample.
- Primary descriptive contrast: regularized minus historical-objective AdamW
  accuracy, averaged over matched encoder/classifier seeds on identical fresh
  rows. Report random/pixel controls, ridge, neighbors and paired errors,
  including negative results. Shared rows/seeds are not independent samples.

## Evidence boundary before running

Training APIs and synthetic-fixture resume/extraction checks work. October 29
added a matched **training** coordinator and no-image aggregate audit. Its
two-arm synthetic tests observe identical actual initial tensors/masks/image
order and recover an interrupted arm. A separate prospective source manifest,
`context-variance-sources-20261029.json`, freezes the complete implementation,
original dataset/splits, exposure bundle and this recipe before any real-image
objective training. The 500-row reservation is unscored, not observed.

**Downstream feature/probe/kNN comparison, selected-state locking, conservative
pre-access observation receipt and fresh scoring for the objective axis are
still pending. Do not use the scale-evaluation driver as if it implements this
contrast, or access reserved images before those gates are implemented.**

Before any real-image experiment, commit the driver and exact code/data/source
pins. Rebuild complete then-current exposure from original files, retaining
`runs/scale-20261020/observed-exposure.json` and every later observation. The
current history has 2,746 observed validation rows and 2,254 remaining; hashes
cannot detect an omitted experiment. Then reserve 50 unused rows/class with
seed 20261027, provided capacity remains. If new observations make this infeasible,
record an amended protocol before accessing any replacement rows or scores.

Lock selected heads/indices and record conservative exposure before fresh-row
access. No fresh fitting, normalization changes, tuning or picking favorable
seeds. Keep official test reserved, use bounded local CPU chunks and publish
aggregates only. Images, checkpoints, features, predictions and row-level
journals remain local and ignored. No paid compute or API calls are needed.

## Limitations to inspect

The batch-four statistic is noisy. Different visible masks can contribute to
between-image variation, and exact constant representations have zero variance
gradient. Increased variance or lower total loss is not proof of useful content,
robustness or higher accuracy. Do not choose a new weight/floor from confirmation
scores; another intervention would require a new prospective protocol. The
training receipt reconstructs seeded initial tensors and the deterministic
mask schedule from pinned code. Actual image order is replayed from durable
journals; the mask schedule is not a separately observed journal. Neither hash
authenticates a producer or detects undisclosed experiments. Both total-loss
definitions differ, so the audit reports prediction/penalty terms separately.
