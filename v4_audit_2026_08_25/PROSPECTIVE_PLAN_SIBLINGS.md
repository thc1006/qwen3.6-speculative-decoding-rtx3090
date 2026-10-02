# Plan S: what shares a core with the main thread

**This cannot be executed yet.** Four of the five items in the last section are
still open, and two of them changed shape after the flags were traced through the
pinned commit's own sources rather than master's. Written 2026-10-02, before any
measurement it describes, and every figure it quotes about data already committed
is derived by
[`../analysis/plan_siblings_power.py`](../analysis/plan_siblings_power.py).

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
fit. So llama.cpp's CPU threadpool does no arithmetic while decoding. What run Y's
displacement moved was the work that remains on the host: sampling, graph build and
launch, the speculative accept and reject bookkeeping, and the synchronisation with
the driver. That work is on the **main** thread.

**And a barrier does not interpolate.** If those threads did split work statically,
the step time is the slowest thread's, so displacing one thread would cost what
displacing eight does. There is no linear scale between zero and a full
displacement on which to place a fraction.

So the host variable is not *how many* threads are slow. It is **what shares a
physical core with the main thread**.

## The candidate, stated before it is tested

`--poll` defaults to **50**, and that is not a nominal amount of spinning:
`ggml_graph_compute_poll_for_work` turns it into 6,553,600 `PAUSE` iterations per
worker per wait before the worker falls back to a condvar sleep. A worker spinning
on the main thread's hyperthread sibling takes issue slots from it. Whether the
scheduler puts one there is drawn afresh at every server launch, and every
published run here passes no affinity at all, which is exactly a bimodal,
per-invocation, unrecorded variable.

One asymmetry to carry into the reading, because it bounds what the third arm can
show: `--poll` reaches the target context's threadpool and not the drafter's. The
drafter's context is built without an attached pool and takes the library default
whatever the flag says. Prerequisite 2 has the lines that show it.

**This is an inference and nothing here has measured it.** No run in this
repository records which processor any llama.cpp thread ran on, so the placement
has never been observed, let alone its cost. What follows is why it is worth half
an hour rather than why it is true.

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
how much slower an efficiency core runs that work equals 14.42 %, and the second
factor is unrecorded. The arms below measure the cost directly instead of inferring
it from that product.

Run Y's own fast set, `0,2,4,6,8,10,12,14`, happens to take one sibling from each of
eight physical performance cores, so it avoided this by accident. Run T4, where the
step lives, passed no mask and let the kernel pack as it liked.

## The three arms

Each differs from the one above it in exactly one thing.

| arm | affinity | and |
|---|---|---|
| `distinct` | `0,2,4,6,8,10,12,14` | eight distinct physical cores. Run Y's fast condition |
| `packed` | `0,1,2,3,4,5,6,7` | four physical cores, both siblings of each |
| `packed-nopoll` | `0,1,2,3,4,5,6,7` | and `--poll 0`, so the target pool's idle workers sleep |

Same binary, same model, same prompts, same thread count, all three on performance
cores. `packed` against `distinct` is the hyperthread question. `packed-nopoll`
against `packed` asks whether the cost is the target pool's spinning, and if it is,
that flag is a remedy for the part of the spinning the flag reaches.

## The design

Within-block log ratios against `distinct` measured in the same block, their mean,
and a Student t interval, which is the estimator run Y used and
`../analysis/rederive_run_y.py` derives. The spread is the within-invocation figure
`../analysis/load_run_power.py` measures from run T4's adjacent differences,
**0.760 %**, and the cost per arm-run is run Y's own: its twelve blocks of four
arm-runs over the span it recorded.

| blocks | arm-runs | hours | 95 % half width | the step in half widths |
|---|---:|---:|---:|---:|
| 6 | 18 | 0.25 | 0.80 % | 5 |
| **12** | **36** | **0.50** | **0.48 %** | **8** |
| 18 | 54 | 0.75 | 0.38 % | 10 |

Twelve is the choice. Half an hour, and an effect the size of A16's step would be
eight half widths from zero; six blocks would still see it at five, and eighteen
buys a tenth of a per cent for another quarter hour.

## The pre-registered reading

Primary: the `packed` against `distinct` ratio.

- **Inside ± one per cent**: hyperthread sharing is not the mechanism, and the host
  variable is something else. That is a real outcome and it would send this back to
  the list, not to a bigger version of the same run.
- **Between one and six per cent**: the size a yielding spin predicts, and A16's
  step is inside that band. The candidate stands and the next question is the second
  arm.
- **Above ten per cent**: too large for a yielding spin. Something is taking the
  core rather than sharing it, and the reading would be about that instead.

Secondary: `packed-nopoll` against `packed`. If it removes the cost, the spinning
worker is the mechanism and `--poll 0` is the remedy. If it does not, the reading
is weaker than it looks: the flag quiets the target context's pool and the
drafter's workers keep spinning either way, so a null says the target pool's spin
is not the whole cost. It does not say spinning is innocent. The run that could
say that is one which places the drafter's threads as well, and today the only
instrument that reaches them is `taskset` with the placement recorded.

I expect `packed` to lose two to five per cent of the decode rate against
`distinct`, and `--poll 0` to recover part of that. How large a part I will not
predict, because the drafter's pool is outside the flag's reach and what share of
the spinning is the drafter's has not been measured. Writing the first number here
is the point: if `packed` comes back inside one per cent, this plan was wrong and
the step is not about hyperthread sharing.

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
Every line number below is `3737e4137`'s.

1. **`--poll` and `--cpu-strict` have to be passed and recorded.** The runner
   passes neither, and neither default is neutral: `ggml_threadpool_params_init`
   in `ggml/src/ggml.c` sets `poll = 50` and `strict_cpu = false`, so every
   published run here carries both unrecorded. The 50 is not a small thing.
   `ggml_graph_compute_poll_for_work`, `ggml/src/ggml-cpu/ggml-cpu.c:3172`, spins
   `1024 * 128 * poll` times, so **6,553,600 `PAUSE` iterations** per worker per
   wait before it falls back to a condvar sleep. At `--poll 0` it is zero rounds
   and the worker sleeps at once.
2. **`taskset` stays, and the native flags are added beside it.** This item said
   the opposite first, that the affinity should be llama.cpp's own instead, and
   that was wrong in a way that would have removed control rather than added it.
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
3. **The sibling topology has to be recorded, and it decides where this can run.**
   Nothing in this repository says which processors share a physical core, so the
   masks above rest on a claim the tree cannot check.
   `/sys/devices/system/cpu/cpu*/topology/thread_siblings_list` is what says it,
   one line per processor. The development box this was written on reports eight
   processors and eight distinct entries, because it is a KVM guest and the
   hypervisor gives it no siblings to pack: `packed` and `distinct` would be the
   same arm there. So this is a bench-host run or no run, and no test here asserts
   the topology, because a test that reads `/sys` is green on the machine it was
   written on and red on a runner.
4. **The built binary's own help still has to be read.** The flags are defined in
   the pinned commit: `common/arg.cpp` gives `-C/--cpu-mask` at 1534,
   `-Cr/--cpu-range` at 1548, `--cpu-strict` at 1554 and `--poll` at 1571, and
   none of them carries a `.set_examples(...)` restricting it to one binary, so
   the server accepts all four. That is the source, not the artefact. What is on
   the bench host is a build, and whether it was configured and linked the way
   those lines assume is a question only `--help` from that file answers.
5. **A derivation script**, which exists, and the claims job has to run it.
