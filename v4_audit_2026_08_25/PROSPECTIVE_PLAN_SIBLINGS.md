# Plan S: what shares a core with the main thread

**This cannot be executed yet.** Five things have to exist first and they are the
last section. Written 2026-10-02, before any measurement it describes, and every
figure it quotes about data already committed is derived by
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

`--poll` defaults to **50**, so llama.cpp's idle workers spin rather than sleep. A
worker spinning on the main thread's hyperthread sibling takes issue slots from it.
Whether the scheduler puts one there is drawn afresh at every server launch, and
every published run here passes no affinity at all — which is exactly a bimodal,
per-invocation, unrecorded variable.

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
| `packed-nopoll` | `0,1,2,3,4,5,6,7` | and `--poll 0`, so the idle workers sleep |

Same binary, same model, same prompts, same thread count, all three on performance
cores. `packed` against `distinct` is the hyperthread question. `packed-nopoll`
against `packed` asks whether the cost is the spinning, and if it is, that flag is
not only a diagnosis but a remedy.

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
worker is the mechanism and `--poll 0` is the remedy. If it does not, the sharing
costs something that is not the spin, and the main thread is contending with
whatever else the scheduler put there, which the next run would have to name.

I expect `packed` to lose two to five per cent of the decode rate against
`distinct`, and `--poll 0` to recover most of it. Writing that here is the point: if
`packed` comes back inside one per cent, this plan was wrong and the step is not
about hyperthread sharing.

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

1. **`--poll` has to reach the server and be recorded.** The runner passes neither
   it nor `--cpu-strict`, and both have defaults that are not neutral: 50 and 0. So
   every published run here has two unrecorded threadpool parameters.
2. **The affinity should be llama.cpp's own, not `taskset`.** `taskset` sets one
   mask for every thread and the kernel places within it; `-C/--cpu-mask` with
   `--cpu-strict 1` walks the mask and gives worker N the Nth bit. For run Y the two
   were equivalent, eight threads into eight processors. For a mask with more
   processors than threads they are not, and the arms above need the strict form to
   mean what they say.
3. **The sibling topology has to be recorded.** Nothing in this repository says
   which processors share a physical core, so the masks above rest on a claim the
   tree cannot check. `/sys/devices/system/cpu/cpu*/topology/thread_siblings_list`
   is what says it.
4. **The pinned build has to accept the flags.** They are in upstream master; this
   repository pins `3737e4137` and nothing here has checked that build's own help.
5. **A derivation script**, which exists, and the claims job has to run it.
