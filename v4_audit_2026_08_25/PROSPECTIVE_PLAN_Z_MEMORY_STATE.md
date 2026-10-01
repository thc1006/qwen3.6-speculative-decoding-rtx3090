# Plan Z: memory state, and whether it can reach A16's step

**This cannot be executed yet.** Six things have to exist first and they are the
last section. Written 2026-10-01, before any measurement it describes, and the
figures it quotes about data already committed are derived by
[`analysis/plan_z_power.py`](../analysis/plan_z_power.py) rather than read off by
hand.

## What this is for, and what it is not

ERRATA A16 is a **3.93 %** step between two levels of `spec-dflash-n2`'s decode
rate inside run T4, with nothing recorded distinguishing them. Three quantities
have been named as unrecorded: host processor placement, GDDR6X memory-junction
temperature, and page-cache or allocator state.

Run Y settled the size of the first. A full move to the eight efficiency cores
costs **14.42 %** of the decode rate, so A16's step is **0.27** of a full
displacement, or roughly two of eight threads landing slow. That is the right
shape and the right order of magnitude, and it is arithmetic until a run sets the
number of displaced threads and measures the response. That successor is the
leading candidate, run Y's own outcome section names it, and nothing has listed it
as an item yet; `RETEST_TODO.md` carries the page-cache hypothesis as the
successor to run X and not this one.

This plan is about the second quantity, and the reason to spend anything on it
now is that it became measurable on 2026-10-01. A16 said the junction temperature
needs a sensor the platform does not offer; it does not, NVML does not expose it,
and the register is readable. The reason to spend *little* is below: the data
already committed points away from it.

**It does not hunt the step.** That step is a telegraph between invocations, run
T4 caught it once across six repeats, and a run cannot summon it. This measures
the **sensitivity** of the decode rate to memory state and asks whether that
sensitivity, times the thermal divergence this card can actually produce, could
reach 3.93 % at all.

## What the committed data already says

Three readings, all from files in this repository, all derived by the script
named above.

**The two levels of A16's step have the same recorded GPU state, and the faster
one ends hotter.** Per repeat of the arm the step is about, from the snapshots
the runner takes at each arm-run's own boundaries:

| level | decode | core before | core after | SM clock | power |
|---|---:|---:|---:|---:|---:|
| slow, repeats 0 to 2 | 139.671 tok/s | 63.0 °C | 69.0 °C | 1930 MHz | 316.9 W |
| fast, repeats 3 to 5 | 145.161 tok/s | 62.7 °C | 70.7 °C | 1930 MHz | 324.1 W |

The gap is 3.93 % and the faster level ends **1.67 °C hotter**. A thermal
explanation predicts hotter and slower. The starting temperatures repeat with the
schedule's period, 62, 60, 67 against 61, 60, 67, so the thermal trajectory is a
function of position in the cycle and is the same on both sides of the step.

This is core temperature, not memory temperature, and memory temperature was not
recorded and cannot be recovered. For the memory to explain the step it would
have to have diverged between the two halves while the core, the clocks and the
power did not, under identical work and identical cooling.

**T4's telemetry trace cannot be attributed to an arm-run.** The trace is
complete, 272 samples at 5.02 s over the whole run, and it is useless for this
question. Arm-runs stamp requests with `CLOCK_MONOTONIC`, the sampler writes wall
clock, and the only anchor the run offers is the manifest's `created` against the
first request, which assumes no server launch precedes it. One does. Reconstructed
that way the window **lengths** come back to 924 s against 924 s of recorded
request spans, and the alignment is still wrong: checked on a signal the outcome
does not use, the three arms hold different drafters and so have different VRAM
footprints, and **all three** arms' windows hold a footprint that is not theirs.

So the within-run thermal profile of the step is not recoverable from what the run
recorded. That is why the first prerequisite below is a timestamp.

**The sensor works and the range is wide.** Measured on 2026-10-01 on this bench
host, idle and under a pure memory-bandwidth load: 42 °C idle against a core
reading of 35, rising monotonically to 90 °C at 828 GB/s sustained, staying above
the core reading throughout, and dropping thirty degrees in the half minute after
the load stops. The core reaches 77 °C at the same moment, against a slowdown
point of 95 and a maximum operating temperature of 93, so under a pure bandwidth
load **the core is the first to approach its limit**, not the memory.

## The levers, and which of them is a treatment

Verified on the bench host on 2026-10-01, each applied and reverted once:

| lever | range | used as |
|---|---|---|
| memory clock, `-lmc` | 405, 810, 5001, 9501 MHz | the treatment in layer B |
| core clock, `-lgc` | 210 to 2100 MHz | held fixed |
| power limit, `-pl` | 100 to 350 W | held at 350, recorded, never binding |
| fan | Coolbits is 28 and the display refuses a non-interactive session | not available |
| thermal state | 42 °C idle to 90 °C under load, falling 30 °C in 30 s | the treatment in layer A |

The power limit is **not** a treatment. With the clocks locked a lower limit
either does nothing or forces the clocks off their lock, and the second is a
different manipulation wearing the first one's name.

The fan is the manipulation this design would have preferred: it changes cooling
at identical clocks and identical work. `nvidia-smi` offers no fan setter, and
`nvidia-settings` needs an X authority this host's session does not hand to a
non-interactive login. Thermal history is what is left, and its decay rate is the
constraint that shapes layer A.

## Layer A, which is cheap and may end the question

**Does achieved memory bandwidth move with thermal state at locked clocks?**

Core clock locked, memory clock locked at 9501, power limit 350 W, one streaming
kernel over two buffers, nothing else on the card. Two phases:

- **ascending**: one continuous load from a cold card to its plateau, bandwidth
  reported per fixed window alongside both temperatures
- **descending**: after the plateau, a duty cycle of a short measured window and a
  long idle gap, so the card cools while each window is the same work

Each window is preceded by an unmeasured warm-up, so a window from a cooler card
is not also a window from a colder cache.

The two phases traverse the same temperature range in opposite directions at
different elapsed times. **If bandwidth is a function of thermal state the two
phases trace one curve; if it is a function of time since the load began, they do
not.** That is the control, and it is the reason the descending phase exists.

Layer A does **not** separate memory temperature from core temperature. It does
not need to: what is being excluded is the thermal channel as a whole, and a
bound that covers both is stronger than one that covers either.

**Pre-registered reading.** Let *R* be the change in achieved bandwidth across
the full reachable range, idle to plateau, read as a fraction of the cold value.

> **If *R* is below 3.93 %, memory thermal state cannot produce A16's step.**

Because reaching 3.93 % of decode rate from *R* of bandwidth requires an
elasticity above one, which is the fully memory-bound ceiling, **and** a
divergence equal to the card's entire thermal range, while the divergence actually
observed between the two levels is under two degrees and has the wrong sign. If
*R* is below 3.93 % the hypothesis is excluded and layer B is not run.

This is deliberately generous to the hypothesis. It grants the maximum possible
elasticity and the maximum possible divergence.

## Layer B, only if layer A does not end it

**How much of the decode rate rides on memory bandwidth?**

Memory clock is the treatment: **9501 against 5001 MHz, a 47.36 % cut** in clock
and so in bandwidth. Core clock locked at one value for every arm-run, power limit
350 W, threads and affinity pinned to the eight performance cores as run Y
established they must be, identical prompts, identical binary, one invocation.

Two arms, `spec-dflash-n2` and `baseline`, at each of the two clocks: **four
arm-runs to a block**. Paired within the block, twelve within-block log ratios per
arm, a Student t interval on eleven degrees of freedom, the same estimator run Y
used and the same one `analysis/rederive_run_y.py` derives.

| blocks | hours | one-sided bound under a null | ± under a null | ± under an effect |
|---|---:|---:|---:|---:|
| 6 | 0.33 | 0.62 % | 0.80 % | 2.19 % |
| **12** | **0.67** | **0.39 %** | **0.48 %** | **1.33 %** |
| 18 | 1.00 | 0.31 % | 0.38 % | 1.04 % |

Twelve is the choice, and the hours are not an estimate: run Y ran four arm-runs
to a block over twelve blocks and its invocation span is recorded, so the
per-block cost of **3.3 min** is measured. The two spreads are named rather than
assumed: **0.760 %** is the within-invocation figure
`analysis/load_run_power.py` takes from T4's adjacent differences, and **2.089 %**
is what run Y measured on its own twelve block ratios, which is the one that
applies when there is a real effect to measure.

A null here is informative too. Bounding the cost of halving the memory clock
below 0.39 % would say the decode rate barely rides on memory bandwidth, and the
hypothesis would be excluded from the other end.

## The arithmetic, pre-specified

With *E* the elasticity layer B measures, in per cent of decode per per cent of
bandwidth, and *B* the slope layer A measures, in per cent of bandwidth per
degree, the divergence A16's step would need is

    ΔT = 3.93 / (E × B)

and the reading is pre-registered: **if ΔT exceeds the full range layer A
traversed, memory thermal state is excluded for A16's step.** If it does not, the
hypothesis survives, and the successor is not another sweep: it is recording the
memory temperature per arm-run from then on and waiting for the telegraph to
appear in a run that measures it.

## What this cannot do

- It cannot test the page-cache and allocator hypothesis. That one is untested
  rather than excluded and this plan does not touch it.
- It cannot test thread displacement, which run Y made the leading candidate.
- It cannot separate memory temperature from core temperature, by design.
- It cannot reproduce A16's step. Nothing can, on demand.
- It says nothing about any card but this one. The register offset is per
  architecture and the thermal range is this cooler and this case.

## Prerequisites

None of this is executable until all six exist. Each is a defect this repository
has already paid for once, in the form the payment took.

1. **A wall-clock timestamp per arm-run, or a monotonic one in the sampler.**
   Without it a trace cannot be attributed to an arm-run, which is what makes
   T4's complete trace useless above, and layer A's whole output is a temperature
   attributed to a window.
2. **The runner locks and restores both clocks, records what was applied, and
   refuses when the readback differs from the request.** Run Y's lesson: a
   treatment that is requested and not verified is a manifest entry, not a
   treatment. `-lgc`, `-lmc`, `-rgc` and `-rmc`, with the restore on every exit
   path including a crash.
3. **Each arm-run records the memory temperature before and after**, from the
   register, beside the core temperature it already records.
4. **A bandwidth microbenchmark in this repository**, reporting a time series
   rather than one aggregate, with its noise floor measured at a plateau and
   published. Layer A's reading is a slope against temperature and a slope needs
   the scatter it is fitted through.
5. **A derivation script for every table either layer publishes**, wired into the
   claims job, as `analysis/plan_z_power.py` is for this document. Run Y's result
   table was hand-computed and three of its columns were wrong.
6. **A declared restore path for the clocks, exercised.** A card left locked at
   5001 MHz would silently halve every later measurement on this host, and the
   runs that followed would look like a new result.
