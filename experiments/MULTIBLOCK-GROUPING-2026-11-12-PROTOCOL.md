# Predictor grouping comparison on CIFAR-10

Compare joint and separate prediction for the same four disjoint 2×2 target
tiles. Match all visible and target positions, initialization, training image
order, and encoder update counts within each seed. Separate prediction uses
four sequences of 40 tokens. Joint prediction uses one sequence of 52 tokens
and lets target mask tokens attend across blocks. The contrast includes this
change in attention and predictor work.

Use the [pinned recipe](multiblock-grouping-protocol-20261112.json) for seeds
29, 31, and 37, 32 updates per arm, four images per batch, and one CPU thread.
Use the same width-32 model, optimizer, EMA, and loss as the November 11 pilot.
Both arms predict 16 target patches from 36 visible patches on every image.

Score mean-pooled context embeddings on the previous pilot's 500 training rows
using the same five class-stratified ridge folds and regularization .1. Include
each seed's matching random encoder and six RGB statistics. The primary result
is the mean within-seed joint-minus-separate accuracy difference. Report all
three differences and each seed's overlap with encoder training. Treat them as
descriptive results from one fitting selection and a short training budget.

Pin the existing exposure receipt, retain its predecessor files, and keep the
official test set reserved. The driver indexes training rows only and stores
weights, sample journals, and detailed fold scores locally.

```sh
PYTHONPATH=src .venv/bin/python experiments/run_multiblock_grouping_20261112.py --preflight
PYTHONPATH=src .venv/bin/python experiments/run_multiblock_grouping_20261112.py
```

Rerunning with a completed directory verifies both saved encoders per seed and
recomputes the fixed classifier scores from the same images. Use the recorded
package sources and runtime to reproduce the pilot.
