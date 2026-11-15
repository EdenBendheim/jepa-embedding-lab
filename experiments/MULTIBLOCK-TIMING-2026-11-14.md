# CPU timing for joint and separate prediction

Joint prediction took 30.4% less wall time per training step by the median
paired ratio. The median trial mean fell from 7.76 ms to 5.42 ms on one Apple
M1 Max CPU thread. The difference was present in both run orders, and most
of it came from the optimizer update phase, which includes forward and
backward passes.

| Seed | First arm | Separate mean step | Joint mean step | Joint / separate |
|---:|---|---:|---:|---:|
| 29 | Separate | 7.82 ms | 5.42 ms | 0.693 |
| 29 | Joint | 7.54 ms | 5.41 ms | 0.717 |
| 31 | Separate | 7.67 ms | 5.30 ms | 0.692 |
| 31 | Joint | 7.75 ms | 5.42 ms | 0.699 |
| 37 | Separate | 8.50 ms | 5.83 ms | 0.686 |
| 37 | Joint | 7.76 ms | 5.62 ms | 0.724 |

The [protocol](MULTIBLOCK-TIMING-2026-11-14-PROTOCOL.md) was committed in
`9173bf6958776213d2265fa410284d6c47abd5a8` before timing. Each arm ran 32
updates, with the first four excluded from per-step summaries. All 1,536
indexed images came from the fixed training split. Each seed's four
trials used identical image order, mask schedules and initial weights,
also verified against the November 12 comparison. The timing path fit no
classifier. Checkpoint writes and evaluation are outside the timed loop.
[Measured trials and budget projection](multiblock-timing-results-20261114.json)

The planned ten-second budget per arm projects 941 updates for separate
prediction and 1,000 for joint after applying a 25% margin to the slowest
trial mean. The current 1,000-update cap binds for joint prediction, so these
counts approximate different elapsed times. At seven seconds the same formula
projects 658 separate and 960 joint with neither arm capped. An equal-update
study can use the smaller count in either case. These projections budget the
training loop on this machine. A longer experiment should record realized
wall time and account for setup, checkpoint writes and evaluation separately.

The timing run produced no new accuracy scores. The earlier 32-update
study used one fitting selection and found 23.47% joint, 23.80% separate,
and 24.40% random. Longer trajectories and independent scoring remain open.
Budget estimates can change with hardware, batch size, thread count and
system load. [Prior accuracy pilot](MULTIBLOCK-GROUPING-2026-11-12.md)

For interviews, “I timed both execution orders with the same training images,
initial weights and masks. Joint prediction cut CPU step time by 30%. I used
the slowest measured trial and a margin to plan a longer comparison.”
