# Multi-target-block foundation — November 8, 2026

An opt-in synthetic trainer can now predict several separate target regions
from one shared visible context. It provides a working implementation to test
under a future controlled real-data protocol. It does not establish better
representations or downstream accuracy. [Aggregate smoke evidence](multiblock-synthetic-20261108.json)

## Mask and objective contract

`jepa_lab.variants.multiblock_masks.multi_block_mask` samples 1–4 distinct
rectangular tiles. Grid dimensions must be divisible by the tile dimensions;
selection is uniform over subsets of that fixed tiling. There is no rejection
loop or arbitrary overlapping rectangle placement. The default 8×8 grid uses
two 2×2-patch target blocks. One context is sampled outside the **union of all
target blocks**, with its size equal to the floor of the chosen fraction of
remaining patches, at least one. Context, targets and unused patches partition
the grid. The local Python RNG leaves the caller's RNG unchanged.

This is a defined-here tiled distribution, not the official I-JEPA multi-block
sampler. The historical single-block sampler and model defaults remain fixed.
The variant lives in a separate package, preserving the pinned top-level
sources of earlier completed experiments.

`multi_block_forward` performs one context-encoder pass and one frozen,
full-image teacher pass. Each target block has its own predictor pass using
that same visible context, so its predictor does not attend to the other
blocks' mask tokens. Output shape is `[images, blocks, target_tokens, width]`.
The loss is the mean Smooth L1 over images, equally sized blocks, tokens and
channels. Teacher outputs remain stopped; context and predictor receive gradients.
EMA updates occur once after the optimizer update.

The extra predictor passes change computation. Matching optimizer steps does
not establish equal compute. Target/context token counts and predictor-pass
counts are recorded in every step's diagnostics.

## Resume and local bounds

The CLI allows 1–100 additional updates, batch size 1–8, at most 64 patches,
embedding width 8–64 and at most two encoder/predictor layers, on one CPU thread.
Trajectories are bounded to 1,000 total updates. Checkpoints retain the exact
model and optimizer state, synthetic-data RNG, PyTorch RNG, completed history,
mask-schedule digests, complete settings, runtime and recursive Python-source
fingerprints. A different block recipe, runtime, code or optimizer configuration
cannot silently resume. The caller's PyTorch RNG is restored on success or failure.

An existing destination requires explicit resume. Checkpoint replacement uses
the existing atomic-save helper; a failed save preserves the previous artifact.
Metrics output cannot replace the resume or destination checkpoint. These are
local experimental checkpoints, not an interchange format for historical CIFAR
feature extraction. Use the same code revision and CPU environment for replay.

## Verification

Eleven focused tests pass, covering seeded complete partitions, impossible
recipes, overlap/nonrectangle rejection, exactly one context/teacher pass,
independent-block equivalence, target stop-gradient, hidden-patch invariance,
zero gradients on all hidden patches, predictor gradients, mismatched recipes,
invalid optimizer settings, failed-save preservation, CLI resume and output
collisions. Split-versus-continuous tests compare all model, optimizer and RNG
tensors and the full history, rather than loss alone.

The local CLI ran ten updates and resumed for three. A separate continuous
13-update run matched **every checkpoint field and history row exactly**. With
two target blocks the context has 42 patches and the target union has eight;
loss changed from 0.5787 on the first synthetic batch to 0.2140 on the thirteenth.
Those different-batch losses verify finite execution, not representation quality.
A separate three-update, four-block run completed with finite losses, 36 context
patches and 16 target patches. All 358 original objective-experiment artifact
hashes remained unchanged. No real-data training or new held-out evaluation
occurred. All compute was local; weights and optimizer states remain ignored.
Full regression results are recorded in the development log.

```sh
python -m jepa_lab.variants.multiblock_train \
  --steps 10 --batch-size 4 --blocks 2 \
  --checkpoint runs/my-multiblock/model.pt \
  --output runs/my-multiblock/first.json

python -m jepa_lab.variants.multiblock_train \
  --steps 3 --batch-size 4 --blocks 2 \
  --resume runs/my-multiblock/model.pt \
  --checkpoint runs/my-multiblock/model.pt \
  --output runs/my-multiblock/resumed.json
```

Use a fresh path for a new trajectory. `--steps` always means additional updates.
For API calls, pass the same custom `ModelConfig` when resuming a custom model.
Source changes require a new trajectory or checking out the checkpoint's revision.

## Remaining work

Connect this variant to public-data training and frozen-feature provenance
before running a new representation comparison. A useful prospective contrast
is one 4×4 target block versus four 2×2 blocks: both retain 16 target patches
and 36 context patches under the default fraction, while changing their layout
and predictor grouping. This is a candidate protocol, not an established gain.
Bind actual image order, model initialization, encoder update/data budgets,
controls, classifier selection and then-current exposure history in advance;
report differing predictor computation. Keep the remaining 1,754 unused
validation rows reserved until the full protocol and scoring path are ready.

Interview explanation: “I implemented several target blocks sharing one context
encoding, checked that no hidden patch leaks into predictions, and made split
training reproduce the entire continuous run. The synthetic checks pass; a fair
real-data comparison is the next step.”
