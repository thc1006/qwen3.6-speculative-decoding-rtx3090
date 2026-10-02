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
displacement: the right order of magnitude, and the host rather than the card.

That fraction is **not** a count of threads, and
[`PROSPECTIVE_PLAN_SIBLINGS.md`](PROSPECTIVE_PLAN_SIBLINGS.md) gives the reason:
every layer is on the card, so those threads do no arithmetic while decoding and
are not on the critical path, and a barrier takes its slowest thread's time either
way. What the host variable actually is, that plan pre-registers: what shares a
physical core with the main thread, which is the leading candidate and is three
arms of thirty minutes rather than a sweep.

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
host with [`bench/vram_temp.py`](../bench/vram_temp.py) and the predecessor of
[`bench/vram_bandwidth.cu`](../bench/vram_bandwidth.cu), which had the same kernel
and the same geometry and differed only in reporting one aggregate instead of a
row per window. Idle and under a pure memory-bandwidth load: 42 °C idle against a core
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

**Which card, and which index.** The slope is the transferable half of this and the
range is not, so it matters which card each is measured on. The bench host's card is
`GPU-f9db9841`, bare metal, 350 W. The box these sessions run on holds a second
RTX 3090, `GPU-f71a8f68`, in a KVM guest at 420 W: the same GA102 and the same
memory, a different cooler and a wider power envelope. A slope measured there is
evidence about the mechanism and **not a coefficient that transfers**, because board
designs differ in how the back-side memory modules are cooled and the onset
temperature can differ with them. The asymmetry is what makes a pilot worth running:
a slope indistinguishable from zero across forty degrees would be strong evidence
against the mechanism as a class and would end this plan cheaply, and a non-zero one
would have to be re-measured on the card A16's step happened on.

On that second card the memory temperature **cannot be read at all**:
`CONFIG_IO_STRICT_DEVMEM=y` makes the kernel refuse to map any IO region a driver
has claimed, the `nvidia` driver claims BAR0, and both routes fail with what look
like two unrelated errors, `EINVAL` through the card's own `resource0` and `EPERM`
through `/dev/mem`. `iomem=relaxed` on the kernel command line disables that check;
it needs a reboot and weakens a hardening, and this plan does not ask for it. The
thermal index there is the core temperature `nvidia-smi` reports everywhere, which
is what the paragraph above already licenses.

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

### What the pilot read, and what it does not yet settle

Layer A ran on 2026-10-02 on the pilot card, published as
[`../v6_pilot_layer_a_2026_10_02/`](../v6_pilot_layer_a_2026_10_02/) and derived by
[`../analysis/layer_a_slope.py`](../analysis/layer_a_slope.py), which also reads that
round's README back row by row. Clocks locked at 1800 and 9501 MHz and held for all
2358 telemetry samples at a spread of zero; the calibration flat to 0.205 % across
five core clocks, so the load was memory bound; the control executed, with nine
temperatures visited by both phases. Across 51.6 to 69.0 °C, **R is 0.014 %**.

So the rule above fires: the hypothesis is excluded for A16's step and layer B is
not run. Put against A16's own figures rather than an invented divergence, its two
levels differ by 1.67 °C, so the sensitivity would have to be 2.35 % per °C, which
is 1,678 times the upper bound measured. A16's own thermal record also has the
faster level as the hotter one.

**That reading is on `GPU-f71a8f68` and the step is on `GPU-f9db9841`.** By the
paragraph above, a null on the pilot card is strong evidence about GA102 with
GDDR6X as a class and is NOT the bench card's number: board designs differ in how
the back-side memory modules are cooled, so the onset temperature can differ with
them. **Repeating layer A on the bench host is what converts this from evidence
about the class into the exclusion for A16**, and until that runs the exclusion is
asserted of the class and not of the card the step happened on. It is about an hour
and it is the only GPU work this plan still needs.

Two things that repeat should carry. The control ran but cannot attribute: the two
phases agree to 0.022 % at matched temperatures and they are two separate
invocations of the load, so a constant offset across every bin is what run-to-run
variation looks like -- seven invocations at a flat plateau spread 0.013 %, about
the same size. Attribution needs replicated pairs, or both phases inside one
process. And the memory register IS readable on the bench host, so that repeat can
index thermal state by the memory temperature rather than by the core's.

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
- It bounds the BANDWIDTH channel only. A thermal effect acting through memory
  LATENCY rather than streaming throughput is untouched: the load is a pure stream,
  and a decode step's KV-cache access is scattered, where row-buffer behaviour and
  latency matter rather than peak throughput. That is a different hypothesis, this
  plan does not test it, and the exclusion above does not reach it.
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
   register, beside the core temperature it already records. The reader exists,
   as `bench/vram_temp.py`, and returns the reason rather than raising when it
   cannot map the BAR, which is the shape `bench/retest_runner.py` already uses
   for `nvidia-smi`. **Landed on 2026-10-02**: the runner calls it through
   `sudo -n`, with a timeout so a measurement cannot stall on an instrument, and
   records the reading or the reason. Mapping a PCI BAR needs root, so the reason
   is the usual case off the bench host, and the reader explains a refusal now
   rather than reporting an errno.
4. ~~**A bandwidth microbenchmark in this repository**, reporting a time series
   rather than one aggregate~~ **Landed on 2026-10-02 as
   `bench/vram_bandwidth.cu`**, which emits a csv row per window with a wall-clock
   stamp in the spelling the sampler writes, so a window joins a temperature trace
   without a mapping between two clocks. Its `--idle` flag is what makes the
   descending phase possible with one program. **Its noise floor is still
   unmeasured and the file has never been compiled**: the bench host has `nvcc` and
   went offline before the rewrite existed. Layer A's reading is a slope against
   temperature and a slope needs the scatter it is fitted through, so that plateau
   measurement is a prerequisite in its own right and nothing may be claimed from
   this tool until it is published.

   **First readings, 2026-10-02, on the pilot card.** It compiles clean under
   `nvcc 13.3` with `-Wall -Wextra`, sustains **828.4** to **828.7 GB/s** at the
   documented four gibibyte buffers, which is the same figure the bench host gives
   and is the two cards hitting the same specification at the same efficiency, and
   its window-to-window spread over four consecutive windows is **0.036 %**. That is
   two orders of magnitude below the step this plan is about, so the instrument has
   the resolution. Four windows is not a noise floor; a plateau of thirty, with its
   spread published, is.
5. **A derivation script for every table either layer publishes**, wired into the
   claims job, as `analysis/plan_z_power.py` is for this document. Run Y's result
   table was hand-computed and three of its columns were wrong.
6. **A declared restore path for the clocks, exercised.** A card left locked at
   5001 MHz would silently halve every later measurement on this host, and the
   runs that followed would look like a new result.
