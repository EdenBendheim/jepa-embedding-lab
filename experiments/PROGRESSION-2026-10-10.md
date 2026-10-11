# October 10 controlled checkpoint-age comparison

Executed comparison A of the [declared protocol](PILOT-2026-10-10-PROTOCOL.md), committed as `0c0c71e` before extracting/scoring the early checkpoint on this pilot. [Machine-readable results](progression-2026-10-10-results.json) retain exact source/classifier fingerprints, selected configurations, candidate accuracies and paired class errors. Raw images, features and weights remain ignored.

## Same examples, same classifiers

Both checkpoints use the same 500 training / 200 validation examples, fixed normalization and context-token mean pooling. Architecture and recorded training settings match; the local October 9 log records the 23→223-step resume. The code verifies matching settings and increasing ages, but cannot prove ancestry from metadata alone.

| Representation | AdamW seeds 29 / 31 / 37 | Mean | Classifier-seed sample SD | Ridge |
| --- | --- | --- | --- | --- |
| 23-step checkpoint | 25.0% / 25.5% / 26.5% | 25.67% | 0.76 pp | 26.0% |
| 223-step checkpoint | 26.0% / 27.5% / 26.5% | 26.67% | 0.76 pp | 27.0% |
| Random encoder, seed 29 | 25.5% / 25.0% / 25.0% | 25.17% | 0.29 pp | 24.5% |
| Raw pixels | 28.0% / 27.5% / 28.5% | 28.00% | 0.50 pp | 30.0% |

The later checkpoint's AdamW mean is **1.0 percentage point** above the early checkpoint on this validation selection. Both remain below pixels. A single encoder trajectory with reused validation tuning does not establish a reliable training improvement. The eight-setting classifier grid was specified before this comparison; it differs from October 9's four-setting grid. The protocol records a correction to its initial mistaken historical reference, while preserving the new declared grid.

## What changed on individual examples

Candidate is the later checkpoint; baseline is the early checkpoint. The same image order and labels are verified before fitting or auditing.

| Classifier seed | Wrong→correct | Correct→wrong | Net accuracy change |
| --- | --- | --- | --- |
| 29 | 12 | 10 | +1.0 pp |
| 31 | 13 | 9 | +2.0 pp |
| 37 | 19 | 19 | 0.0 pp |
| Ridge | See complete paired counts in JSON | See JSON | +1.0 pp |

The unchanged seed-37 aggregate hides 38 opposing changes. For seed 31, ships gained four recoveries against one regression; birds lost four correct predictions against two recoveries. This makes class errors a useful next diagnostic without claiming a significant class-specific effect from only 20 validation examples per class.

## Checks and boundaries

- All 16 selected classifiers (12 AdamW, four ridge) reloaded and exactly reproduced recorded train/validation scores. Every paired diagnostic also reproduced from saved weights without refitting.
- Shared random/pixel sources were byte-fingerprint checked and reused. Each baseline classifier was fitted once per classifier seed; deterministic ridge once total. No redundant baseline fits per age.
- All 62 local tests passed before the run, including synthetic regressions, seed/RNG isolation, source mismatch rejection, saved-source preservation and zero raw-image reads during the comparison workflow. Early feature preparation reads selected official training-split images; comparison/audit only reads verified features.
- No official test image sampling or scoring. Classifier-seed standard deviations quantify classifier initialization only; they are not confidence intervals or independent encoder-training uncertainty.

## Reproduce

Prepare early features with `jepa_lab.embeddings --checkpoint checkpoints/cifar10-pilot.pt --selection runs/pilot-selection-20261009.json` for `train` and `validation`, retaining the exact existing late/random/pixel sources. The full `jepa_lab.progression` command is in the README. Runtime mismatches are rejected where recorded; cross-platform bitwise reproduction is not guaranteed.

## Interview explanation

“I compared two checkpoints on the same examples and classifier budgets, then checked which predictions improved or regressed. Longer training gained one percentage point in this pilot, but pixels still did better. That keeps me from confusing a lower training loss with better representations.”
