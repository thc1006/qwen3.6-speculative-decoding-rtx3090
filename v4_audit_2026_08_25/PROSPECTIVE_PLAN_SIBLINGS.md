# Plan S: what shares a core with the main thread

**This cannot be executed yet.** Two of the six items in the last section are
still open. Two changed shape after the flags were traced through the pinned
commit's own sources rather than master's, two were closed on 2026-10-03 by going
to the bench host and measuring rather than reading, and the sixth exists because
of what that visit found about the binary. Written 2026-10-02, before any
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
| `distinct` | `0,2,4,6,8,10,12,14` | one thread from each of the eight performance cores. Run Y's fast condition |
| `packed` | `0,1,2,3,4,5,8,9` | four performance cores, both threads of each |
| `packed-nopoll` | `0,1,2,3,4,5,8,9` | and `--poll 0`, so the target pool's idle workers sleep |

Same binary, same model, same prompts, same thread count, all three on performance
cores. `packed` against `distinct` is the hyperthread question. `packed-nopoll`
against `packed` asks whether the cost is the target pool's spinning, and if it is,
that flag is a remedy for the part of the spinning the flag reaches.

**`packed` is not the obvious mask, and the measurement is why.** The obvious one
is `0,1,2,3,4,5,6,7`, which is what this document said first. Measuring the bench
host's processors one at a time, which is what
[`../bench/cpu_siblings.py`](../bench/cpu_siblings.py) does and
[`topology/topology_3090_20261003.txt`](topology/topology_3090_20261003.txt)
records, found that `8`, `9`, `10` and `11` run about five per cent faster than the
other twelve performance-core threads: 172.6 against 163.7 thousand blocks per
second, four processors each way, with every processor's own three repeats inside
two tenths of a per cent. Those are the two cores Turbo Boost Max 3.0 favours.

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

1. **CLOSED on 2026-10-03 for the runner, and permanently open for the corpus.**
   Neither default is neutral: `ggml_threadpool_params_init` in `ggml/src/ggml.c`
   sets `poll = 50` and `strict_cpu = false`, and the binary's own help prints the
   same two values, so they are a property of every run here rather than a guess.
   The 50 is not a small thing. `ggml_graph_compute_poll_for_work`,
   `ggml/src/ggml-cpu/ggml-cpu.c:3172`, spins `1024 * 128 * poll` times, so
   **6,553,600 `PAUSE` iterations** per worker per wait before it falls back to a
   condvar sleep. At `--poll 0` it is zero rounds and the worker sleeps at once.

   `bench/retest_runner.py` now takes `BENCH_POLL` and `BENCH_CPU_STRICT`, screens
   them, puts `--poll` into argv beside `-t`, and records `poll` and `cpu_strict`
   in the arm-run record and in the run manifest. They are recorded whether or not
   they are passed, because a **null field** is what makes the absence visible and
   argv alone never prompted anyone to look.

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
   1534, `-Cr/--cpu-range` at 1548, `--cpu-strict` at 1554 and `--poll` at 1571
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
5. **A derivation script**, which exists, and the claims job has to run it.
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

   What would close this is recording what the process actually MAPPED rather than
   what it was offered, read from `/proc/<pid>/maps` after the server is healthy;
   the runner already reads `/proc/<pid>/status` for the allowed cpu set, so it is
   the same access. Hashing them per arm-run is not the way: `libcublasLt.so.12`
   alone is 714 MiB and the set is 815, which over thirty six arm-runs is 28 GiB
   of hashing inside a measurement, and CPU work inside a measurement is what the
   host guard exists to refuse. Identity per arm-run from `stat`, hashes once at
   run start, and a stated residual that a replacement identical in size, mtime
   and inode would pass.
