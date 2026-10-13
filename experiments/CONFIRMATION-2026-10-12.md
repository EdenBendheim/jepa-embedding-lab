# Fresh classifier confirmation — October 12, 2026

The [protocol](CONFIRMATION-2026-10-12-PROTOCOL.md) and
[24 fixed source states](confirmation-sources-20261012.json) were committed
before the new scores were observed. The selector kept 500 fitted training
examples, excluded all **246 unique previously evaluated validation rows**, and
reserved **500 new validation examples, 50 per class**. They come from the
original validation partition; the official test set was not sampled/evaluated.

## Results

Every cell uses the original selected classifier, training-only mean/std and
settings. No classifier or normalization was fitted on confirmation data.
AdamW means average classifier seeds 29, 31 and 37; ridge has one deterministic
selected state per arm. These are descriptive results, not significance tests.

| Frozen representation | Dimensions | Original tuning AdamW mean | Fresh AdamW mean | Fresh AdamW sample SD | Fresh ridge |
| --- | ---: | ---: | ---: | ---: | ---: |
| Learned default, mean | 32 | 26.67% | 24.47% | 0.42 pp | 21.20% |
| Learned default, mean_std | 64 | 28.67% | 25.67% | 0.50 pp | 27.20% |
| Learned 4x4 mask, mean | 32 | 27.50% | 24.47% | 1.29 pp | 24.60% |
| Random seed 29, mean | 32 | 25.17% | 24.13% | 2.04 pp | 25.40% |
| Random seed 29, mean_std | 64 | 33.83% | 32.73% | 0.70 pp | 30.40% |
| Pixels | 3,072 | 28.00% | 28.20% | 1.40 pp | 30.60% |

The learned mean_std arm remains below its matching random encoder and pixels.
The 4x4-mask gain seen in tuning disappears in this fresh-data mean. No reliable
useful-learning advantage is established, and no new winner/settings are selected
from these scores. Stronger training and a pretrained comparator remain next.

## Verification and provenance

- All **24 selected classifier states** reproduce original training/tuning
  scores and match the committed file/state/source fingerprints before scoring.
- Every arm receives identical ordered fresh examples and labels. Frozen
  encoder weights, choice of pooling, random initialization and original
  classifier normalization must match their earlier provenance.
- Selection fingerprint:
  `decdf27565acb4db350754997cb230b0a1e836219d5895885d370da79ca358cd`.
- Fixed-source ledger fingerprint:
  `35d72dd90702f8fe0e3e5f5fd3010c35c395185be4398a619cd904d01fc3a024`.
- [Aggregate metrics and provenance](confirmation-results-20261012.json) retain
  every selected configuration, original accuracy, fresh class/confusion counts,
  and state/source hashes. They contain no raw images, feature arrays or weights.
- All **85 synthetic tests pass**. The driver also reproduces a byte-identical
  cached report with image extraction/sampling/fitting disabled; an intentionally
  corrupted cached feature is rejected without replacing the previous summary.

## Repeat locally

```sh
export PYTHONPATH=src
.venv/bin/python experiments/run_confirmation_20261012.py
.venv/bin/python experiments/run_confirmation_20261012.py --reuse-only \
  --output runs/confirmation-20261012/rechecked.json
```

The driver needs the earlier locally produced checkpoints, feature artifacts,
selected probes and reports named in the committed source ledger. It verifies
those exact files; this is not a claim of identical tensors/file hashes across
hardware or library versions. Data, features and weights remain ignored.

## Limits and next work

These examples were untuned relative to the declared earlier pilots, but share
the same dataset/validation partition. There is only one learned encoder seed
for each mask setting. Mean_std doubles dimensions; changed masks also change
token/compute budgets. Classifier-seed spread does not estimate encoder/data
uncertainty. Now that these scores have been inspected, these examples cannot
serve as untouched confirmation for later tuning decisions. Keep official test
reserved, predeclare stronger training and a frozen pretrained comparator, and
reserve another independent subset when a new protocol is selected.
