# Complete objective evaluation protocol — October 31, 2026

This completes the execution boundary of the [October 27 recipe](CONTEXT-VARIANCE-2026-10-27.md).
The [October 29 source manifest](context-variance-sources-20261029.json) and its
500-row unscored reservation remain unchanged. The new implementation has a
separate [75-file source manifest](context-variance-sources-20261031.json), SHA-256
`0dd2e26a60885435085d59baf4070b330b689b7d875cbbbbc3b0ba2f28e49165`.
It pins all implementation modules, dependency metadata, original source files,
the predecessor manifest/record and original reservation. No new objective
training outcome or fresh score has been inspected before this lock.

Both arms start from scratch: prediction-only versus the same prediction loss
plus 0.1 times the visible-context image-mean variance hinge, floor 1.0.
Width 32, four heads, depth 2/1, fixed sin/cos positions at scale 1, 2x2 targets,
.75 context fraction, batch four, AdamW .001 and EMA .99 remain fixed.
Seeds 20261003/20261010/20261011 each receive 2,223 updates per arm. Training
journals must match actual image order; initialization/mask schedules are
reconstructed from pinned code. Equal updates and capacity are not equal compute.

Use the original 500 fitting / 200 tuning rows. One shared random initialization
per encoder seed and one pixel source avoid duplicate control fitting. Select
AdamW on LR .0001/.001/.01/.05 × decay .01/.1, 200 steps, classifier seeds
29/31/37; ridge .01/.1/1/10; cosine kNN 1/5/15. Save/reproduce **every linear
candidate state**, not just reported losing scores. Ties use the existing rule:
accuracy, then cross-entropy/MSE, then declared order. kNN uses accuracy then
declared order. No new classifier grid, penalty or seed selection is allowed.

The primary descriptive result is context-variance minus prediction-only AdamW
accuracy, averaged over the three matched encoder seeds and three classifier
seeds. Report both trained arms, shared random/pixel controls, ridge, neighbors,
individual encoder means and paired class errors, including negative results.
The same examples and seeds are shared; sample SD is descriptive, not an
independent confidence interval or official test estimate.

The retained reservation has selection SHA-256
`89e0432f0845c22e0a139a41c07fa5b380c3936fe57d2b9409d937ea489ae049`:
50 unused rows/class, seed 20261027, excluding all 2,746 declared observed rows.
It is the **same 500 rows**, not an additional reservation. Before fresh access,
lock selected linear/kNN states, verify every candidate artifact, extend the
source-replayed exposure ledger and atomically record the matching receipt.
An interrupted attempt consumes the reservation conservatively. No fresh fitting,
normalization changes, classifier tuning or favorable-seed selection. Official
test stays reserved; missing/corrupt caches cannot replace earlier evidence.

```sh
PYTHONPATH=src .venv/bin/python -m jepa_lab.objective_run \
  --protocol experiments/context-variance-sources-20261031.json \
  --expected-sha256 0dd2e26a60885435085d59baf4070b330b689b7d875cbbbbc3b0ba2f28e49165 \
  --directory runs/context-variance-20261031 --mode prepare --reuse-only
```

Modes `train`, `select`, `evaluate` and `audit` perform the corresponding explicit
steps. `select` stops before observation; `evaluate` consumes before accessing
fresh images. Completed `--reuse-only` evaluation must reproduce the summary
without fitting, training or image indexing. Checkpoint and source locks are
integrity checks, not authenticated producer evidence or proof of complete
experiment disclosure. Preserve all original files. Publish aggregates only;
raw images, selections, features, weights and per-row journals remain ignored.
Use bounded local CPU chunks; no paid API/cloud compute is required.
