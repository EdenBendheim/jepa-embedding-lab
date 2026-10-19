# Position encoding comparison — prospective October 18, 2026

Declared before new fixed-position training or confirmation scoring.
[Exact recipes/source hashes](position-sources-20261018.json).

- Compare learned positional parameters against the defined row/column sin/cos
  buffers in context, target and predictor. Width 32, four heads, two encoder
  blocks and one predictor block throughout. Fixed values have amplitude one
  while learned parameters initialize at std .02; this contrast changes both
  positional form and trainable capacity, not just an isolated mathematical term.
- Three encoder seeds: 20261003/20261010/20261011; 2,223 updates, batch four,
  AdamW LR .001, EMA .99, same 2x2 target/.75 visible-complement mask recipe.
  Reuse the three exact completed learned-position checkpoints from October 14.
  Train three fixed-position runs from scratch. Non-position initial weights
  match within each seed, and actual sampled image order must match in journals.
- Frozen full-image context mean pooling, exact same 500/200 fitting/tuning rows.
  Each arm has its exact-initialization random encoder; pixels shared. AdamW
  grid LR .0001/.001/.01/.05 x decay .01/.1, 200 steps, seeds 29/31/37; ridge
  .01/.1/1/10; cosine kNN counts 1/5/15. Select only on old tuning rows.
- Reserve 500 new balanced validation rows, seed 20261018, excluding all 1,746
  observed validation examples in the October 15 confirmation ancestry.
  Selection hash: `dae8b93be16dd715569d0df47f590a875e3ba66bc0681c9c80cda5848054954a`.
  Restore/audit and lock selected heads/indices before fresh feature extraction.
  No fitting, parameter selection or normalization changes on fresh examples.
- Primary descriptive contrast: fixed versus learned position *trained* encoders
  under AdamW, matched seed averages and paired recoveries/regressions. Report
  all random/pixel/ridge/kNN outcomes too. Three seeds are not a reliable broad
  uncertainty estimate; fresh rows are shared across seeds. Do not select a
  winner/hyperparameters from these scores or claim official test performance.
- Local one-thread CPU only. Keep real images, features, weights, row-level
  journals and saved classifiers ignored. Publish aggregate metrics/configuration
  and hashes. Do not incur paid API or cloud charges. Finite loss is not quality.

The training driver verifies pinned inputs, saved journals, matched actual sample
order and complete budget identity. A cached retry cannot silently train. The
next evaluation step must restore states and verify its full cache before
replacing a summary.
