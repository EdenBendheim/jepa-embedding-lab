# Predictor grouping with a longer training budget

On 300 previously unused CIFAR-10 validation images, the 960-update joint
encoder averaged 27.11% accuracy across three seeds. The 658-update separate
encoder averaged 25.22%. The longer joint arm recovered five to seven net
correct predictions per seed. RGB statistics averaged 27.33%, so this
experiment leaves a learned-feature advantage over simple color features open.

| Frozen features | Seed 29 | Seed 31 | Seed 37 | Mean |
| --- | ---: | ---: | ---: | ---: |
| Separate, 658 updates | 24.67% | 24.00% | 27.00% | 25.22% |
| Joint, 658 updates | 25.67% | 24.00% | 25.33% | 25.00% |
| Joint, 960 updates | 26.33% | 26.33% | 28.67% | 27.11% |
| Matching random encoder | 26.33% | 24.33% | 25.67% | 25.44% |
| RGB statistics | | | | 27.33% |
| Raw pixels | | | | 20.33% |

The shared-update joint-minus-separate difference was −0.22 percentage
points on average. The 960-update joint-minus-separate difference was +1.89
points. Relative to its own 658-update checkpoint, the longer joint arm gained
2.11 points. All comparisons share the same images. Each cell above uses one
fixed ridge fit on 500 training rows. The final two baselines each use one head
shared across seeds.

| Seed | Joint 658 vs separate, recovered / regressed | Joint 960 vs separate | Joint 960 vs joint 658 |
| ---: | ---: | ---: | ---: |
| 29 | 7 / 4 | 10 / 5 | 7 / 5 |
| 31 | 8 / 8 | 15 / 8 | 19 / 12 |
| 37 | 4 / 9 | 14 / 9 | 16 / 6 |

The [protocol](MULTIBLOCK-BUDGET-2026-11-18-PROTOCOL.md) was committed in
`2fe1a9aafd0c6591bf6d739b02d56f7473e81106` before training. Each seed matched
initial weights, masks and image order through update 658. The longer joint
checkpoint retained its entire shared journal. Four independent predictor
passes and one joint pass used the same target union and context.

| Seed | Separate 658 loop | Joint 658 loop | Joint 960 loop | Separate 658 trainer | Joint 960 trainer |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 29 | 4.85 s | 3.41 s | 4.99 s | 5.91 s | 6.49 s |
| 31 | 4.85 s | 3.46 s | 5.03 s | 5.50 s | 6.57 s |
| 37 | 4.88 s | 3.40 s | 5.08 s | 5.54 s | 6.63 s |

The extended joint arm processed 46% more updates with about 3–4% more loop
time. Its trainer wall time was about 10–20% higher, including repeated setup,
resume validation and checkpoint writes. The seven-second plan was based on
earlier trial costs and a margin. These are fixed update budgets with measured
differences in CPU time on one Apple M1 Max thread.

The audit counted 19,916 training-partition image reads, 300 registered
validation reads and zero official test image reads. The new exposure bundle
retains 3,546 observed validation rows and 1,454 remaining under the recorded
history. It is stored at `runs/multiblock-budget-20261118/observed-exposure.json`,
with the previous bundle and original source files retained. Use that latest
bundle for a future reservation.

Replay reproduced the same report byte for byte with encoder training and
ridge fitting disabled. It indexed the same 300 registered validation images,
zero training images and zero test images. The 31 multi-block tests, seeded
mask demo, three-step training demo and aggregate checks passed.

The published aggregate contains per-class counts, confusion matrices,
paired errors, checkpoint hashes and elapsed times. Heads, checkpoints, source
images and row identifiers stay local. The experiment has one small reserved
validation sample shared by all seeds. It provides descriptive comparisons,
with no test-set result or statistical estimate of generalization.
[Aggregate results](multiblock-budget-results-20261118.json)

For interviews, “I tested whether one faster predictor pass could buy more
updates. I matched both arms through 658 updates, then let the joint model
reach 960. It improved on the separate model with a similar training-loop
cost, while a simple color baseline still scored slightly higher.”
