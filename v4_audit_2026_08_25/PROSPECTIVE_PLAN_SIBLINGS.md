# Plan S: what shares a core with the main thread

**One item in the last section is still open, and it is not a blocker.** It is
the first, and only its second half: the seventy eight runs already committed will
never carry the two threadpool parameters, which is a permanent fact about the
corpus rather than something to be done. Everything the arms below need now
exists. Two changed shape after the flags were traced through the pinned
commit's own sources rather than master's, two were closed on 2026-10-03 by going
to the bench host and measuring rather than reading, and the sixth exists because
of what that visit found about the binary. Written 2026-10-02, before any
measurement it describes, and every figure it quotes about data already committed
is derived by
[`../analysis/plan_siblings_power.py`](../analysis/plan_siblings_power.py).

**The third arm was changed on 2026-11-13, before any run of it, because the
flag it was built on reaches nothing in this binary.** The server these runs use
is an OpenMP build, and ggml reads `--poll` only when it is built without OpenMP.
Item 7 has the lines. The third arm now differs from the second in the server's
environment, `OMP_WAIT_POLICY=passive`, which is the setting OpenMP's own workers
read, and every arm-run records each server thread's processor time, which is the
first observation of the mechanism this plan names. What had been seen of any run
when this was changed is in the reading section.

## Why this plan exists, and why the previous reading was wrong

ERRATA A16 is a **3.93 %** step between two levels of one arm's decode rate inside
run T4, with nothing recorded distinguishing them. Run Y measured the host channel:
moving llama.cpp's eight threads from eight distinct performance cores to eight
efficiency cores costs **14.42 %**, so A16's step is **0.27** of a full
displacement. The right order of magnitude, and the host rather than the card.

Four documents then read that fraction as a count: *roughly two of eight threads
landing slow*. It cannot be one, and the reasons are not subtle.

**The eight threads are not on the critical path.** Every target layer and every
drafter layer is on the card — the memory fitter chose that, and ERRATA records the
fit. One operation per forward pass is known to stay on the host by design: the input layer,
which `src/llama-model.cpp` keeps on the CPU at 1463 to 1465 because offloading it
gains little, and its one operation, the embedding lookup, is planned as a single
task, `ggml/src/ggml-cpu/ggml-cpu.c` at 2336 to 2342, so it runs on the main
thread. Unless some other operation falls back to the CPU, which nothing here has
checked, the other seven threads have no arithmetic to do while decoding. What run
Y's displacement moved was the work that remains on the host: sampling, the
embedding lookup, graph build and launch, the speculative accept and reject
bookkeeping, and the synchronisation with the driver. That work is on the **main**
thread.

**And a barrier does not interpolate.** If those threads did split work statically,
the step time is the slowest thread's, so displacing one thread would cost what
displacing eight does. There is no linear scale between zero and a full
displacement on which to place a fraction.

So the host variable is not *how many* threads are slow. It is **what shares a
physical core with the main thread**.

## The candidate, stated before it is tested

The server these runs use is an OpenMP build, so the eight threads are libgomp's.
When a parallel region ends, a libgomp worker spins before it sleeps: up to
`GOMP_SPINCOUNT` rounds, 300,000 when `OMP_WAIT_POLICY` is unset, which is what
every run here left it. A worker spinning on the main thread's hyperthread sibling
takes issue slots from it. Whether the scheduler puts one there is drawn afresh at
every server launch, and run T4, where the step lives, passed no affinity at all,
which is exactly a bimodal, per-invocation, unrecorded variable.

This document first named ggml's own threadpool and its `--poll` level, 50 by
default, which spins 6,553,600 `PAUSE` iterations per worker per wait. That pool
exists only in a build without OpenMP. Item 7 has the lines, and the third arm
was changed for it before it ran.

**There is a reason to doubt the workers spin at all while decoding**, and it was
found from the source before any run. A worker spins only after a region it took
part in, and the only operation this document can name on the CPU during decoding,
the embedding lookup above, is planned as one task, so the region it would open
has one thread and wakes no worker. If nothing else falls back to the CPU, no
worker spins, the candidate below is wrong at its root, and a cost in `packed`
would have to come from something else that shares the core. That is what the
per-thread processor time every arm-run now records can show, because a worker
that spins accrues processor time and one that sleeps does not.

`OMP_WAIT_POLICY` is read once per process and applies to every OpenMP team in
it, so the third arm reaches the drafter's waiting as well as the target's. The
first version of this plan carried an asymmetry here, `--poll` reaching the
target's pool and not the drafter's; under OpenMP it does not arise.

**This is an inference and nothing here has measured it.** No committed run in
this repository records which processor any llama.cpp thread ran on, so the
placement has never been observed, let alone its cost. The arm-runs of this plan
will be the first to record it, each server thread's processor time and the
processor it last ran on. What follows is why it is worth half an hour rather than
why it is true.

The ordering is what fits. A full displacement of the host work costs 14.42 %; the
step is roughly a quarter of that, and about five times the within-invocation spread
of 0.760 % that run T4's own adjacent differences give. So the step sits between a
whole-threadpool move and the noise this harness can resolve, which is where sharing
one core with one spinning thread belongs. A yielding `PAUSE` loop gives up most of its
issue slots, so it costs a few per cent rather than a fifth of the throughput a
compute-bound sibling would take, and it is nowhere near the full displacement that
putting the main thread on an efficiency core would cost.

What that ordering does not give is a predicted size, because the host's share of a
token is itself unmeasured here. Run Y bounds it only as a product: the share times
the extra time an efficiency core takes over that work is the growth in time per
token that a 14.42 % loss of rate implies, and the second factor is unrecorded. The
arms below measure the cost directly instead of inferring it from that product.

Run Y's own fast set, `0,2,4,6,8,10,12,14`, happens to take one sibling from each of
eight physical performance cores, so it avoided this by accident. Run T4, where the
step lives, passed no mask and let the kernel pack as it liked.

## The three arms

Each differs from the one above it in exactly one thing.

| arm | affinity | and |
|---|---|---|
| `distinct` | `0,2,4,6,8,10,12,14` | one thread from each of the eight performance cores. Run Y's fast condition |
| `packed` | `0,1,2,3,4,5,8,9` | four performance cores, both threads of each |
| `packed-passive` | `0,1,2,3,4,5,8,9` | and `OMP_WAIT_POLICY=passive` in the server's environment, so idle OpenMP workers sleep at once |

[`../bench/run_s_siblings.sh`](../bench/run_s_siblings.sh) is the driver, and
those three names are conditions rather than the runner's. It runs them as
`spec-dflash-n2`, `spec-dflash-n2-packed` and `spec-dflash-n2-passive-packed`,
twelve blocks of three in `latin` order, all in ONE invocation, every request
generating to a cap of `300` tokens. That ONE invocation is not a convenience:
A16's step is BETWEEN invocations, so an arm measured in its own invocation is
confounded with the variable this plan exists to isolate. An environment variable
set for the invocation would reach all three arms, so the third condition is an
entry in the runner's `ARM_ENV` table, which gives one arm's server a variable the
others do not get. It differs from its base in exactly that variable, and every
arm-run records what its server was given as `server_env`, read back from the
server's `/proc/<pid>/environ` rather than from what the runner meant to pass.

**The order is `latin`, and the first attempt at this run was stopped for not
being.** `BENCH_ORDER=mirrored` was copied from run Y's driver, which has four
arms: reversing an even-length order gives each arm two positions. Three arms
reverse to leave the middle one where it was, and rebuilding the order from the
fifteen arm-runs of the stopped attempt showed exactly that: `spec-dflash-n2-packed`
at position 2 in every block it ran and the other two only ever at 1 and 3. Over
twelve blocks the schedule would have put the outer two at {1: 6, 3: 6} and the
middle one at {2: 12}, which the runner now computes and refuses. Mean position is still
balanced, so a linear drift within a block cancels, but a first-position effect
does not: the outer arms take it half the time each and the middle one never, and
the arms are then not exchangeable. This repository has discarded a run for the
same class of defect before, RETEST_TODO P1-3, where every freerun block ran
before every hard-cap block, so it is disqualifying rather than a caveat. `latin`
over twelve blocks puts each of the three arms in each of the three positions
four times and makes each of them first in four blocks, and the runner validates
that before it spends any GPU time.

What `latin` does not balance is the PREDECESSOR. Within a block a cyclic
rotation leaves the adjacent pairs one-directional: `distinct` is
always followed by `packed`, `packed` by `passive`, `passive` by `distinct`, eight times each over
the twelve blocks, and the reverse pairs occur only across the eleven boundaries
between blocks. So within a block each arm always has the same arm before it.
`williams` does not remedy that here: the runner's Williams square needs an even
number of arms and refuses three, and a carryover-balanced design for three arms
is two squares, six sequences, which the runner does not build. Twelve blocks
would hold it twice, so precision is not what rules it out; it is a design this
plan would have to add, and it keeps `latin`.

The argument that the residual is immaterial is an argument and not a
measurement, and it is this: the three arms do identical GPU work. Same model,
same drafter, same prompts, same draft length, three hundred tokens each. They
differ in which processors the server may use and in one environment variable, and every
arm-run starts a fresh server, so nothing of the previous arm's placement
survives into the next process. What survives is the card's thermal and power
state, and that is set by the GPU work, which does not differ. If that argument
is wrong the effect is a constant per arm, which is exactly what the within-block
contrast cannot separate, so it would have to be addressed by a design and not by
a correction.

The driver refuses six things before it spends any GPU time, and each is a
condition the reading depends on: a set that is not the thread count, an
efficiency core in either set, unequal exposure to the favoured processors, a
distinct set where two processors share a core, a packed set where one does not,
and the two sets spanning the same number of cores. The logic is
`validate_masks` in [`../bench/cpu_siblings.py`](../bench/cpu_siblings.py) rather
than in the driver, because a runner has four processors, no `cpufreq` and no
siblings it will admit to: a refusal that lives only in the shell script is one
CI never executes. `bash bench/run_s_siblings.sh --check` runs all six against
the host without touching the card.

Same binary, same model, same prompts, same thread count, all three on performance
cores. `packed` against `distinct` is the hyperthread question. `packed-passive`
against `packed` asks whether the cost is the OpenMP workers' spinning, and if it
is, that variable is a remedy for it.

**`packed` is not the obvious mask, and the measurement is why.** The obvious one
is `0,1,2,3,4,5,6,7`, which is what this document said first. Measuring the bench
host's processors one at a time, which is what
[`../bench/cpu_siblings.py`](../bench/cpu_siblings.py) does and
[`topology/topology_3090_20261003.txt`](topology/topology_3090_20261003.txt)
records, found that `8`, `9`, `10` and `11` run about five per cent faster than the
other twelve performance-core threads: the fastest of the four and the fastest of
the twelve at 172.6 against 163.7 thousand blocks per second, and every one of the
four faster than every one of the twelve. Those are the two cores Turbo Boost Max
3.0 favours.

The measurement confirmed this rather than discovering it, and that is the part
worth recording against myself. `BENCHMARK_ENV.md`'s CPU addendum has said since
2026-09-17 that four of the thirty two logical processors reach 5800 MHz while
twelve reach 5500, which is the same fact: that ratio is a twentieth, and the
throughput ratio measured here matches it to a fraction of a per cent, which is
also what says this kernel is limited by clock rather than by cache. What the
addendum did not say is WHICH four, so these masks were first written from the
plausible assumption that the eight performance-core threads are interchangeable.
They are not, the tree already held the reason, and reading it would have been
cheaper than measuring it.

`0,2,4,6,8,10,12,14` puts two of its eight threads on favoured processors.
`0,1,2,3,4,5,6,7` puts none there. So those two masks differ in hyperthread sharing
AND in clock, the main thread lands on a favoured processor a quarter of the time
in one arm and never in the other, and the expected difference from that alone is
about one and a third per cent. The band this document pre-registers for "not the
mechanism" is one per cent. A confound larger than the smallest effect the design
would call real is not a caveat, it is a different experiment.

`0,1,2,3,4,5,8,9` is four cores with both threads of each, and two of its eight
threads on favoured processors, the same as `distinct`. One thing differs: how many
physical cores the eight threads span, four against eight.

One asymmetry is left and it is worth naming because the direction matters.
`distinct` loads one thread of each favoured core, so both of them are busy;
`packed` loads both threads of one, so the other is idle. Two favoured cores
running together lose about five per cent against either alone, which the same
measurement shows at `8 + 10`. That makes `distinct` the slower arm by a little,
which works against finding a cost in `packed` rather than for it. A residual that
biases toward the null is one this plan can carry; the one it replaced biased the
other way.

## The design

Within-block log ratios against `distinct` measured in the same block, their mean,
and a Student t interval, which is the estimator run Y used and
`../analysis/rederive_run_y.py` derives. The spread is the within-invocation figure
`../analysis/load_run_power.py` measures from run T4's adjacent differences,
**0.760 %**, and the cost per arm-run is run Y's own: its twelve blocks of four
arm-runs over the span it recorded.

| blocks | arm-runs | hours | 95 % half width | the step in half widths | under run Y's spread | the step in those |
|---|---:|---:|---:|---:|---:|---:|
| 6 | 18 | 0.25 | 0.80 % | 5 | 2.19 % | 2 |
| **12** | **36** | **0.50** | **0.48 %** | **8** | **1.33 %** | **3** |
| 18 | 54 | 0.75 | 0.38 % | 10 | 1.04 % | 4 |

Twelve is the choice. Half an hour, and an effect the size of A16's step would be
eight half widths from zero; six blocks would still see it at five, and eighteen
buys a tenth of a per cent for another quarter hour.

**The last two columns were added on 2026-11-13, and they are the less
comfortable reading.** The first three were computed on one spread, the 0.760 %
run T4's adjacent arm-runs give, which is the right one when the two arms do not
differ. Plan Z, written the same week, carries a second beside it, because a real
effect widens the spread and the precision under an effect is the one that
matters: run Y's own twelve within-block log ratios spread 2.089 %. This plan did
not, so its table stated only the optimistic precision. Under run Y's spread
twelve blocks give 1.33 %, the step is three half widths from zero rather than
eight, and the interval is wider than the null band: a point estimate inside it
comes with an interval reaching at least one neighbour, and both when it is within
a third of a per cent of zero, which the reading section reports as reaching them
rather than resolving. These half widths are on the scale of a mean log ratio
times a hundred, the scale of the spreads they come from; the derivation prints
them on that scale and the run's interval back-transformed, each labelled. Twelve
stays the choice, because
it was registered before the first attempt ran and the reading does not depend on
which spread holds; what twelve buys under each is stated so that a reader does
not have to work it out, and the derivation reports the observed spread beside
both.

## The pre-registered reading

Primary: the `packed` against `distinct` ratio, as a signed percentage change
from `distinct` to `packed`. The edges below are written as digits because
`../analysis/rederive_run_s.py` parses them out of this section: a band that lives
both here and in a table inside that script is two copies of one reading, and two
copies drift.

The first version of this list gave three bands and no direction, and the script
that implemented it read a `packed` three per cent FASTER as "the candidate
stands". A gain refutes the mechanism; the omission was here, not only there.

- **Inside -1 % to +1 %**: hyperthread sharing is not the mechanism, and the host
  variable is something else. That is a real outcome and it would send this back to
  the list, not to a bigger version of the same run.
- **-6 % to -1 %, packed slower**: the size a yielding spin predicts, and A16's
  step is inside that band. The candidate stands only if the workers were seen
  to spin: the other threads' processor time in `packed-passive` at most half of
  that in `packed`, per arm-run. Otherwise sharing a core costs this much and
  the spinning worker is not why, and the reading says that instead.
- **Below -10 %**: too large for a yielding spin. Something is taking the core
  rather than sharing it, and the reading would be about that instead.
- **Above +1 %, packed FASTER**: not predicted by anything here. Packing four cores
  cannot speed the main thread up, so a gain of that size means the arms differ in
  something this design did not control, and the reading is about finding it rather
  than about the mechanism.
- Between -10 % and -6 %: no band was reserved, and this says so rather than
  rounding into a neighbour.

An edge belongs to the band nearer zero, so exactly -1 % and exactly +1 % read as
the null band, exactly -6 % as the predicted one and exactly -10 % as the gap.
The rule is the conservative one: a value on an edge is read as the smaller
finding. The reading is the band of the point estimate. Its 95 % interval is
reported beside it with every band the interval reaches, and an interval that
reaches a second band is reported as reaching it rather than resolved into one.

**When this section was revised, and what had been seen by then.** The bands were
rewritten with digits and a direction on 2026-10-03, after the first attempt at
this run had been stopped, and sat uncommitted until 2026-11-13, when they were
committed together with the edge rule and the interval sentence above; the commit
date is the one a reader can check. What had been seen of that attempt is exactly
this: the driver's progress lines for its first arm-run, `spec-dflash-n2` in block
0, which print each request's decode rate, ten of them; and every arm-run's
`t_start`, read to rebuild the order. No rate of `spec-dflash-n2-packed` or
`spec-dflash-n2-nopoll-packed`, the third arm's name then, was displayed and no
contrast was computed, so no
revision here could have been fitted to one. The attempt's fifteen arm-runs stay
on the bench host beside an `ABANDONED.txt` that says why, and nothing in this
repository is derived from them.

**Revised again on 2026-11-13, and this time the arm, not the bands.** A review of
this branch found that the third arm's `--poll 0` reaches nothing in the OpenMP
build the runs use, item 7, and the arm became `OMP_WAIT_POLICY=passive`; the
secondary reading below and the expectation after it were rewritten for it, and
the paragraph on whether the workers spin at all was added, and the predicted
band's "the candidate stands" was made to wait on that observation, with the
rule written in the band. Nothing of any run had
been seen since the inventory above: the bench host has been unreachable since
before the bands were committed, and the only arm-runs of this plan that exist
are that stopped attempt's fifteen, on that host. The primary contrast, its band
edges and its expectation are unchanged.

Secondary: `packed-passive` against `packed`. If it removes the cost, a spinning
OpenMP worker is the mechanism and `OMP_WAIT_POLICY=passive` is the remedy. If it
does not, the per-thread processor time separates two readings. If the other
threads' processor time is clearly lower under `passive`, the workers were
spinning and their spin is not the cost: spinning is then excluded as the
mechanism for this build, and a cost in `packed` is something else that shares
the core. If it is not lower, the workers were not spinning in either arm, which
is what a decode whose only CPU operation is a single task predicts, and the
secondary contrast tests nothing. It does not say spinning is innocent in a build
or a fit where the workers have work. The threads that time does not separate are
the server's own: libgomp's workers carry the server's name, so it is read as a
difference between the two packed arms and not as a level.

I expect `packed` to lose 2 % to 5 % of the decode rate against `distinct`; that
was written on 2026-10-02 and stands. For the third arm I make no prediction of
size, and since the single-task finding above I would not be surprised by none.
Writing the first number here is the point: if `packed` comes back inside one per
cent, this plan was wrong and the step is not about hyperthread sharing.

## What this cannot do

- It cannot reproduce A16's step on demand. Nothing can: that step is a telegraph
  between invocations and run T4 caught it once in six repeats.
- It cannot test the page-cache and allocator hypothesis, A16's third, which is
  untested rather than excluded.
- It says nothing about the memory-subsystem hypothesis, which
  [`PROSPECTIVE_PLAN_Z_MEMORY_STATE.md`](PROSPECTIVE_PLAN_Z_MEMORY_STATE.md)
  excludes for the bandwidth channel on a pilot card.
- It measures what a chosen placement costs. It does not measure how often the
  kernel chooses it, which is what would turn a cost into an explanation of a
  bimodal history. That needs the placement recorded per arm-run over many
  invocations, and it is cheap once the recording exists.
- The whole argument rests on every layer being on the card. If a run ever fits
  differently, the threadpool has arithmetic to do and the reasoning above has to be
  redone.

## Prerequisites

The flags were traced through the pinned commit's own sources before this list
was written, and two of the five items changed shape because of what that found.
The trace missed one thing, that the build is an OpenMP one, and item 7 is what
that cost. Every line number below is `3737e4137`'s.

1. **CLOSED on 2026-10-03 for the runner, and permanently open for the corpus.**
   Neither default is neutral: `ggml_threadpool_params_init` in `ggml/src/ggml.c`
   sets `poll = 50` and `strict_cpu = false`, and the binary's own help prints the
   same two values, so they are a property of every run here rather than a guess.
   The 50 is not a small thing. `ggml_graph_compute_poll_for_work`,
   `ggml/src/ggml-cpu/ggml-cpu.c:3172`, spins `1024 * 128 * poll` times, so
   **6,553,600 `PAUSE` iterations** per worker per wait before it falls back to a
   condvar sleep. At `--poll 0` it is zero rounds and the worker sleeps at once.
   All of that is the threadpool of a build WITHOUT OpenMP. In the build these runs
   use none of it executes, item 7, which is why the third arm no longer uses it.

   `bench/retest_runner.py` now takes `BENCH_POLL` and `BENCH_CPU_STRICT`, screens
   them, puts `--poll` into argv beside `-t`, and records `poll`, `poll_batch` and
   `cpu_strict` in the arm-run record and in the run manifest. They are recorded
   whether or not they are passed, because a **null field** is what makes the
   absence visible and argv alone never prompted anyone to look. `--poll-batch`
   goes with `--poll`: the runner passes `-tb` beside `-t`, which gives the batch
   parameters a thread count of their own, and `postprocess_cpu_params`,
   `common/common.cpp` at 288 to 297, copies the poll level into them only when
   that count is unset. So `--poll` alone would leave the batch parameters at 50,
   the two sets would no longer match, `ggml_threadpool_params_match`,
   `ggml/src/ggml.c` at 8061 to 8066, and `common_threadpools::init` would build a
   second pool, `common/common.cpp` at 1800 to 1826; every speculative
   verification is a batch of the sampled token and the draft and runs on that
   one, `src/llama-context.cpp` at 1385 and 2479.

   `BENCH_CPU_STRICT=1` is refused, and the reason is not `taskset`. Strict
   placement reads `params.cpuparams.cpumask`, which only `--cpu-mask` fills;
   `taskset` sets the process affinity and leaves that mask empty.
   `ggml_thread_cpumask_is_valid` is true only if some bit is set,
   `ggml-cpu.c:2682`, and ggml applies affinity only when it is, 3211, so with no
   `--cpu-mask` the strict branch walks nothing. Accepting 1 would put a treatment
   in the manifest that did not happen, so the knob exists to record the value and
   raising it needs `--cpu-mask` first.

   The seventy eight committed runs will never carry either field. That is why no
   arm of this plan can be compared with them on this axis, and why the contrast
   lives inside the plan's own blocks.
2. **`taskset` is the instrument, and it is not a compromise.** This item said the
   opposite first, that the affinity should be llama.cpp's own instead, and that
   was wrong in two ways rather than one. It would have removed control rather
   than adding it, and `--cpu-strict` without `--cpu-mask` does nothing at all,
   which item 1 now records with the lines that show it. The arms above are
   realised by `taskset` alone, and they do not need a mask flag: in `packed`
   every processor in the set is a sibling of another one in it, so sharing
   happens wherever the kernel puts a thread, and in `distinct` no two are, so it
   cannot. The placement within the set is the kernel's and the contrast does not
   depend on it.
   In the OpenMP build these runs use, neither context's pool has threads of its
   own: both run their graphs as `#pragma omp parallel` regions on libgomp's
   workers, `ggml-cpu.c` at 3383 to 3402, so the threads `taskset` places are
   libgomp's, and the drafter's work runs on them too.
   The server's TARGET context gets a threadpool built from `params.cpuparams`:
   `tools/server/server-context.cpp:1051` calls `common_init_from_params`, whose
   constructor at `common/common.cpp:1290` calls `threadpools.init` at 1408, and
   that attaches a pool made from the mask, the strict flag and the poll level at
   1826. The DRAFTER context does not. `common/speculative.cpp` builds it with
   `llama_init_from_model` directly, 2412 and 2424, copying only the thread
   count, so no pool is attached and `ggml_graph_compute` makes a disposable one
   from the library defaults. `taskset` is a process-wide affinity the kernel
   applies to every thread including the drafter's; `--cpu-mask --cpu-strict 1`
   places the target pool alone. Dropping `taskset` for it would lose the
   drafter's threads, which is why run Y's instrument was the broader one.
3. **CLOSED on 2026-10-03: the topology is measured and recorded.**
   `/sys/devices/system/cpu/cpu*/topology/thread_siblings_list` states it, but on a
   virtual machine it states what the hypervisor chose to say, so it was measured
   instead and then compared.
   [`topology/topology_3090_20261003.txt`](topology/topology_3090_20261003.txt) is
   that run: 32 processors, 24 physical cores, the eight performance cores pairing
   as `0+1` through `14+15` and the sixteen efficiency cores alone. All 15
   pairs agree with `/sys`, sibling pairs measuring 0.492 to 0.496 of their two
   solo rates and distinct pairs 0.950 to 1.001, against a same-processor control
   at 0.501. So the masks above are a checked claim rather than an assumed one.

   It also decides where this can run. The development box these documents are
   written on reports eight processors and eight distinct entries, because it is a
   KVM guest and the hypervisor gives it no siblings to pack: `packed` and
   `distinct` would be the same arm there, and measuring it confirmed that, all 28
   of its pairs landing between 0.989 and 1.002. This is a bench-host run or no
   run. No test here asserts the topology, because a test that reads `/sys` or
   times a processor is green on the machine it was written on and red on a runner;
   what the suite asserts is the arithmetic that turns rates into a verdict.
4. **CLOSED on 2026-10-03: the binary the runs used accepts the flags.** The
   source says so at the pinned commit, `common/arg.cpp` giving `-C/--cpu-mask` at
   1534, `-Cr/--cpu-range` at 1544, `--cpu-strict` at 1554 and `--poll` at 1571
   with no `.set_examples(...)` restricting any of them. The artefact now says so
   too. Seventy seven of the seventy eight committed runs record
   `server_sha256` `b6a5c490bb932ffa`, run T4 and run Y among them, and the file
   with that hash on the bench host is built from `3737e41370da1830a44c663f9929a0f27591ffa6`.
   Its own help prints `--cpu-strict <0|1>` with `default: 0` and
   `--poll <0...100>` with `default: 50`, which is what the library's
   `ggml_threadpool_params_init` sets and what makes those two values a property of
   every run here rather than a guess about one.

   **And it prints a trap.** The same help lists `--spec-draft-poll`,
   `--spec-draft-cpu-mask`, `--spec-draft-cpu-strict` and their batch variants, so
   a reader would reasonably try `--poll-draft 0` for the drafter's workers. On this
   code path those flags land nowhere. `common_base_params_to_speculative` copies
   exactly two fields out of the draft's `cpuparams`, `n_threads` and
   `cpuparams_batch.n_threads` at `common/speculative.cpp` 2335 and 2336, so the
   mask, the strict flag and the poll level never cross; and the draft context is
   built by `llama_init_from_model` at 2412, while the only call to
   `common_threadpools::init` anywhere in `common/` is `common.cpp:1408` inside
   `common_init_result`'s constructor, which the draft context never goes through.
   The flags are accepted and silently ineffective. That is worse than their
   absence, and it is the reason item 2 says `taskset` stays.
5. **CLOSED on 2026-11-13: the derivation runs in the claims job.**
   [`../analysis/rederive_run_s.py`](../analysis/rederive_run_s.py) verifies the
   design out of the arm-runs before it derives either contrast, and refuses on
   any condition the reading depends on: a mask the plan did not assign or the
   kernel did not grant, two server builds, two library sets beside the binary
   or two sets of shared objects actually mapped, one server process answering
   two arm-runs, a thread count that is not the mask's size, a threadpool flag
   in any arm, a server environment other than the one the arm table gives, a
   build that mapped no libgomp, a record without each thread's processor time, an
   order, token cap or build other than the ones registered here, a model,
   drafter, draft length or prompt set that differs between arms, a request that
   stopped short of the token cap, a crashed arm-run, a directory the runner did
   not attest as complete, or a position count that is not balanced. A field that
   is absent is a refusal, never a pass, and a record of the wrong shape is a
   refusal rather than a crash. The claims job and the local gate both run it.
   With no run committed it parses the arms, the masks, the block count and the
   reading's bands out of this document and checks that the bands partition the
   line, so it has been guarding the reading since the day it was wired in
   rather than from the day the data lands.
6. **The binary cannot start on the bench host as it stands.** It needs
   `libcudart.so.12` and `libcublas.so.12`, and its `RUNPATH` is only its own
   build directory. The DRIVER is installed and `ldconfig` knows it:
   `libcuda.so.1` resolves to `libcuda.so.580.178.04`. The TOOLKIT is not. There
   is no `/usr/local/cuda` of any version, no `nvcc`, and `ldconfig` has no entry
   for `libcudart`, `libcublas` or `libnvrtc`. The only copies of the two the
   binary wants are pip wheels inside unrelated Python virtual environments,
   `~/colabfold_venv` and a PyTorch project's `.venv`, and that distinction
   matters: a driver library is a property of the machine, while a wheel in
   somebody else's environment is a file that can be removed by `pip uninstall`
   in a project that has nothing to do with this one.

   So every published run here was launched with a library path that nothing in
   this repository records. The manifests carry `server_lib_sha256` for the
   `libggml` and `libllama` objects beside the executable and nothing for the CUDA
   runtime underneath them, so two runs could differ in it while their manifests
   matched.

   **CLOSED for future runs on 2026-10-03.** `bench/retest_runner.py` records
   `server_mapped` in every arm-run record: every shared object the process has
   actually mapped, read from `/proc/<pid>/maps`, with its size, mtime in
   nanoseconds, inode and sha256. The runner already read `/proc/<pid>/status` for
   the allowed cpu set, so it is the same access.

   Three decisions in it are load-bearing. **Shared objects only**, because
   llama.cpp mmaps the model by default and a 21 GiB gguf is a file-backed mapping
   too; its hash is already `target_sha256` and hashing it here would add twenty
   one gigabytes per arm-run to a measurement. **Hashes cached on identity, not on
   path**, because there is no run-level moment at which to take them -- the server
   starts per arm-run and nothing is mapped before it -- so the first arm-run pays
   for the set and the rest pay nothing, while a library swapped between two arms
   gets a new key and is hashed again. Keyed on path it would have served the first
   hash and reported a swap as no change, which is the defect this exists to catch.
   And **a failure is an error, not an empty dict**: an empty one reads as "nothing
   was mapped", which is false of any dynamically linked process and is exactly the
   invisible absence that let the CUDA runtime go unrecorded for seventy eight runs.

   Per arm-run that is a few dozen `stat` calls. The cost was measured before the
   design was chosen: `libcublasLt.so.12` alone is 714 MiB and the mapped CUDA set
   is 815, so hashing it per arm-run would be nearly 29 GiB over thirty six of them,
   inside a measurement, and CPU work inside a measurement is what the host guard
   exists to refuse.

   The residual is stated rather than closed: a replacement identical in size,
   mtime and inode would reuse the cached hash. That is not a realistic accident.
   And the seventy eight runs already committed carry none of this, which is why
   the arms of this plan compare against each other and not against them.
7. **CLOSED on 2026-11-13: the build is an OpenMP one, and the third arm was
   changed for it before it ran.** `ggml/CMakeLists.txt` turns `GGML_OPENMP` on
   by default at 245, and nothing this repository records as a build command turns
   it off. Run Y's own server log, on the bench host and this binary, prints
   `OPENMP = 1` in its `system_info` line. In that build ggml's threadpool has no
   worker threads, `ggml-cpu.c` at 3314 to 3346, and every multi-threaded graph is a
   `#pragma omp parallel` region, 3383 to 3402; the poll level is read only in
   `ggml_graph_compute_poll_for_work`, 3172, inside the `#ifndef GGML_USE_OPENMP`
   block that runs from 3140 to 3276. So `--poll 0` would have been a treatment
   recorded in every manifest and applied to nothing.

   libgomp's workers wait as libgomp says. With `OMP_WAIT_POLICY` unset a worker
   spins up to `GOMP_SPINCOUNT` rounds, 300,000 by default, or 100 when the process
   runs more OpenMP threads than it may use processors, and then sleeps; `passive`
   makes it sleep at once. ggml sets `KMP_BLOCKTIME` when it is absent, which is
   the variable of LLVM's and Intel's runtimes; the libgomp on the development box
   carries no such name and the one on the bench host has not been checked.

   So the third arm is `OMP_WAIT_POLICY=passive`, given to that arm's server alone
   by the runner's `ARM_ENV`. The runner refuses any other OpenMP variable in the
   environment it was started with, because that would reach every arm unrecorded,
   and records `server_env` for every arm-run from `/proc/<pid>/environ`. The
   driver refuses a server that does not load libgomp, and `rederive_run_s.py`
   refuses an arm-run that did not map it. Every arm-run also records
   `server_threads`, each thread's processor time across the measured requests and
   the processor it last ran on, read from `/proc/<pid>/task`, which is what says
   whether the workers spun at all.

