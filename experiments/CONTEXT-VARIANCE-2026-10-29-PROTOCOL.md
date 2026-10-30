# Context-variance training lock — October 29, 2026

This implements the training boundary of the [October 27 prospective
recipe](CONTEXT-VARIANCE-2026-10-27.md). It adds no classification result.

The [source manifest](context-variance-sources-20261029.json) has SHA-256
`b66b82fe7a8e3e031a6a61417995a382c9a9f1866e0f04319eb73fb40d5a40a8`.
Its 67 file pins cover every experiment module, project dependency metadata,
the recipe, original split/fitting selection, and the retained exposure sources.
The manifest itself is pinned by that external digest, avoiding a self-hash.

The fixed contrast is prediction-only versus the same objective plus 0.1 times
the context image-mean variance hinge, floor 1.0. Both use width 32, four heads,
encoder/predictor depth 2/1, fixed sin/cos positions at scale 1, 2x2 targets,
.75 visible-complement fraction, batch four, AdamW LR .001 and EMA .99.
Seeds 20261003/20261010/20261011 each start both arms from scratch for 2,223
updates. Each arm has 29,472 trainable parameters; equal updates are not equal
compute. Completed checkpoints cannot be adopted without the matching receipt.

## Preparation actually performed

The complete original-source ledger replays **2,746 observed** validation rows,
leaving **2,254 unused**. No new selection document outside that ledger was
found in the local selection/reservation inventory. The ledger fingerprint is
`16a928e1e139f9ff27297b58a7e336acc01c5703cc4d0e7b2178dda5314347ef`.
Hashes still cannot detect undisclosed experiments or authenticate authorship.

Reserved 50 rows/class using seed 20261027, excluding the observed union.
The selection fingerprint is
`89e0432f0845c22e0a139a41c07fa5b380c3936fe57d2b9409d937ea489ae049`.
All **500 reserved rows remain unscored**; they have not been registered as new
observations. The official test remains reserved. Preparation and its repeated
cache check ran with image indexing disabled; they read file hashes and labels,
not image tensors. Local row selections remain ignored under `runs/`.

Recheck preparation from the repository root:

```sh
PYTHONPATH=src .venv/bin/python -m jepa_lab.objective_run \
  --protocol experiments/context-variance-sources-20261029.json \
  --expected-sha256 b66b82fe7a8e3e031a6a61417995a382c9a9f1866e0f04319eb73fb40d5a40a8 \
  --directory runs/context-variance-20261029 --mode prepare --reuse-only
```

## What is verified and what is next

Synthetic fixtures verify actual equal initial tensors/masks, training-only
image order, complete-cache replay without training/image reads, interrupted
recovery, protected evidence and loss-component audits. The training receipt
reconstructs the initial state and mask schedule from pinned code; only image
order is independently retained in the training journal. It is an integrity
check, not evidence of semantic quality or an authenticated training ancestry.

No real-image objective budget, fitted classifier, fresh-image extraction or
accuracy evaluation was executed for this protocol. The objective-specific
feature/probe/neighbor comparison and selected-state/observation receipt gates
must be implemented before reserved-image access. The earlier scale driver
does not implement this objective contrast. New evaluation code requires a
new source manifest retaining this original training/data/history evidence;
do not silently repin this file or choose another penalty using fresh scores.

Keep all raw images, checkpoint/feature artifacts, per-row journals and source
bundles local. Publish aggregates only; no paid compute or API was used.
