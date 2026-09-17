# Prospective plan for run X: does host CPU load move the arm A16 is about

Written and committed before the run. A hypothesis picked after looking at the
data is not a hypothesis, and the outcome section at the end of this file is
empty until the data exists. This repository has a retracted headline that
asserted a mechanism chosen after the fact, which is the habit the genre exists
to break.

**This plan cannot be executed yet.** Five adversarial passes over it found six
things the harness cannot express and several figures that did not re-derive.
The prerequisites are listed near the end, each naming the file and the change,
and the run does not start until they land. A pre-registration that describes a
run nobody can perform is worth less than one that says what has to be true
first, and saying it here is cheaper than finding it at block two.

## Why this run exists

[A16](../ERRATA.md#a16-two-runs-identical-in-every-recorded-respect-and-byte-identical-in-output-differ-by-34--on-one-arm)
ends by naming two quantities its instruments could not see. One of them, the
GDDR6X memory-junction temperature, needs a sensor NVML does not expose on Linux
and cannot be read on this host at all. The other is host CPU load.

`bench/host_guard.py --sample` records it, and has since commit `81b30fe` on
2026-08-28. That date is not before every run here: run W2 was measured on
2026-08-30, with the sampler already sitting in the tree, and it does not carry
the column either. No run in this repository does, in any of the seventy-seven
committed run directories, and none of the seventeen committed telemetry traces
holds anything but `nvidia-smi` fields.

So the second of the two named gaps is testable, the instrument is in the tree,
nothing has used it, and the reason is no longer that it did not exist.

## What A16 established, and what it did not

Taken as given here, with the figures re-derived from the committed arm-runs
rather than quoted from prose:

- run T4 caught `spec-dflash-n2` stepping **inside one invocation** with the
  telemetry running. Its six per-repeat pooled rates are 139.359, 139.725,
  139.930, 145.044, 146.014 and 144.425 tok/s: three low, three high. The two
  level means are 139.671 and 145.161, a gap of **3.93 %**. A16's prose rounds
  this to "a 3.5 % step" and "two levels about 3.5 % apart"; the measured gap is
  what this plan uses as a simulation input, and A16's wording is what it quotes
  when it quotes A16.
- across that step the two arms interleaved with it held at a CV of 0.12 %
  (`spec-draft-n8`) and 0.55 % (no speculation)
- run T against run T3, two invocations two hours apart, moved the same arm by
  −3.40 % while `spec-draft-n8` moved −0.11 % and no speculation moved +0.79 %.
  These are between-invocation figures. The within-invocation step is what run X
  is built around, because run X is one invocation
- ruled out by measurement: core temperature, SM clock, board power, the
  preceding arm, the fitter's decisions, free GPU memory at start, and time to
  become healthy
- both levels produce byte-identical output and identical draft counts

Not established: any cause. The instrument list that failed to distinguish the
levels was `nvidia-smi` alone.

One framing A16 withdraws, and this plan does not reinstate: pooled across runs
the distribution is clustering with a heavy low tail, not a clean two-state
system. "Two levels" describes what this one arm does within and between
invocations, not the pooled population.

A16 also names a third hypothesis that run X does **not** test: machine history
between T and T3, two rebuilds and a killed rehearsal, which changes page cache
and allocator state and is recorded by no field here. The outcome section is
forbidden below from forgetting it.

## What the bench host actually is

Stated because both adversarial passes reasoned about the wrong machine, and
because two of these facts are candidate explanations that cost one column to
rule out.

The bench host is bare metal: `systemd-detect-virt` reports none,
`/proc/cpuinfo` carries no hypervisor flag, and `/proc/stat`'s steal column is
zero. Hypervisor descheduling, which would look exactly like A16 and is
invisible to every guest counter, is therefore not available as an explanation
here. The development box these plans are written on IS a KVM guest with a live
steal column, which is why the distinction is worth writing down.

It is an i9-13900K: thirty-two logical processors across three frequency tiers,
5800 MHz on four of them, 5500 MHz on twelve and **4300 MHz on sixteen**. The
slow sixteen are efficiency cores, and the spread between tiers is twenty-two to
twenty-six per cent.

Nothing in `bench/` or `analysis/` pins any process to any processor: there is
no `taskset`, no `sched_setaffinity`, no cpuset. And no run in this repository
records which processor anything ran on.

The server's thread count is not recorded either. The committed argv carries no
`-t` and no `--threads`, so llama.cpp chooses, and what it chose is in the
server log, which is a release asset rather than a file in this tree. On a
hybrid processor the thread count is exactly what decides whether work lands on
efficiency cores at all, so run X pins it explicitly and records it in the
manifest beside everything else.

So core placement is an uncontrolled and unrecorded variable whose magnitude is
the right order, whose timescale is a scheduler's, which produces byte-identical
output, and which `nvidia-smi` cannot see. That is A16's signature. This plan
does not claim it is the cause — the arm's step persisted across three
consecutive fresh server processes, which a single migration does not explain —
but it is cheap to record and run X records it. It also matters for the design:
**applying CPU load changes core placement**, so a positive result in a run that
did not record placement would not be attributable.

## What the committed data already says, before any run

A third pass asked whether this run is the one worth making, and answered it
partly from data already in the tree. Two results, both at zero GPU cost, and
both of which change what run X is for.

**Across T4's own step the arms moved in opposite directions.** Splitting T4 at
its step, reps 0 to 2 against reps 3 to 5:

| arm | low | high | change |
|---|---:|---:|---:|
| `spec-dflash-n2` | 139.671 | 145.161 | **+3.93 %** |
| `baseline` | 116.533 | 115.667 | **−0.74 %** |
| `spec-draft-n8` | 30.799 | 30.847 | +0.16 % |

One arm got faster while another got slower. **No arm-agnostic host cost can do
that**, under any of the three denominators tabulated later, because a cost per
token, per forward pass or per target step moves every arm the same way. The
plan's own pre-registered statistic agrees: D over those six blocks is −25.309
before the step and −25.529 after, a shift of +0.220 ms per token with a Welch t
of 6.25 and an interval excluding zero. The split point was chosen by eye and
this is therefore corroboration and not a test, which is exactly why the run
below fixes the split in advance.

**And no whole-host component is detectable between any pair of arms.** Aligning
the three arms by block within each of the 43 run directories that hold all
three for at least four blocks, and correlating their residual milliseconds per
token, gives +0.073 for `baseline` against `spec-dflash-n2`, +0.026 for
`spec-dflash-n2` against `spec-draft-n8`, and −0.083 for `baseline` against
`spec-draft-n8`. A shared host cost of any denominator moves the arms together
and would show as a correlation near one. Nothing here does.

That second result has two readings and this plan does not pick between them:
either the arms do not share a host channel, or the ambient host state on a
quiet bench machine does not vary enough to correlate anything. **Those are
distinguishable only by setting the level rather than watching it**, which is
the argument the next heading makes and which this result sharpens rather than
undermines. What it does do is lower the prior: the most likely outcome of run X
is that both arms hold, and the plan should be read knowing that.

## What should run before this, and why it is not this

The same pass costed a cheaper experiment and it is the right one to run first.

The bench host is hybrid, and pinning the server to its efficiency cores is a
deliberate worst case: a clock cut of about a quarter, far beyond anything
ambient scheduling could produce. Two arms, two pinnings, six blocks is 24
arm-runs and **0.34 hours**; twelve blocks is 48 and 0.68 hours. Against the
corpus per-arm-run SD of about one and a half per cent, six paired blocks detect
about three per cent and twelve detect about two.

A null inside one per cent there bounds the **entire host-CPU-speed family** at
once: core placement, CPU clock, and ambient load acting through the scheduler.
It would make run X unnecessary. A positive sends you to a crossed design,
placement against load, two arms and twelve blocks, at 1.36 hours, which also
supplies the heat-without-contention cell this plan lists as a limitation.

Run X as written is 3.03 hours with its overhead and 3.63 with the washout it
mandates below. It is pre-registered here because the question is worth a
pre-registration and because writing it is what found everything above. It
should not be the next thing on the card.

## Why this is an intervention and not an observation

Recording host load during an ordinary run and looking for a correlation would
settle nothing. Ambient load on a quiet bench host may not vary enough to show
anything, and a null result would then be a statement about the ambient
distribution rather than about the mechanism. So the load is applied as a factor
whose level is chosen, not observed.

## The design

One invocation. Twenty-four blocks. Three arms per block:

- `spec-dflash-n2`, the arm that moves
- `spec-draft-n8`, which held at a CV of 0.12 % across the T4 step
- `baseline`, no speculation

**The load is a within-block factor.** Each arm is run twice inside its block,
once with the load and once without, back to back, so the two members of a pair
are one arm-run apart rather than up to eleven blocks apart. That is the change
that makes this run able to answer its question, and
[the power](#the-power-from-this-repositorys-own-variability) is the arithmetic
for why.

The arm order advances by one each block, so each arm visits each position eight
times. Within a block the loaded run comes first in half the blocks and second
in the other half, alternating ABBA over the twenty-four.

### The power, from this repository's own variability

`analysis/load_run_power.py` is committed with this plan and `--check`
re-derives
what it covers, comparing it against this document row by row rather than by
searching for a value anywhere: the power table, the cross-arm table, the three
chi-squares, the three corpus correlations, the four measured inputs and the
wall
clock. What it does not yet cover is named here rather than left to be assumed:
the hazard interval, the sensitivity rows at its ends, the washout table's other
rows, and the D values. Its inputs are measured, not chosen:

- the per-arm-run residual log SD is **0.537 %**, the measured adjacent-repeat
  difference SD of 0.760 % with the level change excised, over root two
- the level gap is the measured **3.93 %**
- arm-run durations 39.8, 116.0 and 44.1 s, so a three-arm pass is 200 s
- the washout between the two members of a pair is **30 s**
- the switch hazard is **one per 641 s**, with a ninety-five per cent interval
  of 455 to 863 s

The hazard needs a paragraph, because two earlier versions of this plan got it
wrong in the same direction. The first took it from run T4 alone: one transition
across five adjacent gaps, a single event whose exact Poisson interval runs from
179 s to 39 498 s. The second counted changes across the whole corpus, fifty
runs, 100 changes across 291 adjacent gaps and 112 295 seconds, and divided by
elapsed time. That is still not the estimator: there are two levels, so two
transitions inside one gap return the arm to where it started and record as no
change at all, and the mean gap is 386 s, the same order as the hazard. The
likelihood over the gaps, each with its own duration, gives one per 641 s where
the count gave 1123. A hazard that is too slow makes every design built on it
look better than it is.

The inputs also have to decompose, and the first version of this table is why
that is checked rather than assumed: T4's block CV of 2.145 % **already contains
the level change**, and a simulation that adds a telegraph on top of it implies
2.909 % of block spread, **thirty-six per cent** more than T4 shows. It
published a detection rate of about one in ten for a design whose real rate is
under three in ten. `hypot(gap / 2, residual)` is 2.037 % against the measured
2.145 %, and `tests/test_harness_invariants.py` holds the identity with the
faulty reading as its known-positive.

Forty thousand draws, by the decision rule defined below:

| design | truth −2 % | truth −3.93 % | truth 0, declares holds |
|---|---:|---:|---:|
| between-block, 6 against 6 (first draft) | 0.28 | 0.78 | 0.11 |
| within-block, 12 pairs | 0.63 | 1.00 | 0.37 |
| within-block, 18 pairs (the void floor below) | 0.78 | 1.00 | 0.58 |
| **within-block, 24 pairs (this plan)** | **0.89** | **1.00** | **0.77** |

At the fast end of the hazard interval the last row is 0.82 and 0.64; at the
slow end, 0.92 and 0.84.

### What the washout costs, stated because it is not free

The washout between pair members is there for one-directional carryover, and it
is paid for in specificity, because it lengthens the window in which the arm can
change level inside a pair:

| washout | truth −2 % | truth 0, declares holds |
|---:|---:|---:|
| none | 0.95 | 0.89 |
| 15 s | 0.92 | 0.83 |
| **30 s, this plan** | **0.89** | **0.77** |
| 60 s | 0.82 | 0.63 |

Two earlier versions of this plan mandated a washout and simulated none, so
their tables described a design nobody was going to run. Thirty seconds is the
choice and this is its price. Recovering it means more blocks, not a shorter
washout, because the carryover the washout is for is what makes the pairing
trustworthy in the first place.

### What the block sequence balances, and what it does not

The loaded-first and loaded-second blocks have equal mean position, so a linear
drift over the run cancels exactly in the contrast, and the sequence is a
palindrome, so it cancels in wall-clock terms too. Enumeration also kills a
quadratic. A general monotone drift is not cancelled and the first version of
this sentence said it was.

What counterbalancing cannot fix is that the carryover here is **one
directional**: whatever a loaded run leaves in page cache or allocator state
flows forward onto the run after it and never backward. Over twenty-four blocks
most unloaded arm-runs immediately follow a loaded one. Alternating the order
spreads that contamination evenly across the contrast rather than removing it.
So the plan inserts an explicit equal washout between pair members and records
`/proc/meminfo`'s `Cached` at every arm-run boundary, which makes the carryover
a measured covariate instead of an argument.

## What the load is

**A stated number of CPU-busy worker processes at nice 0**, started before a
loaded arm-run and stopped after it, pinned in the run's own configuration and
recorded in its manifest.

The previous version of this plan specified the repository's own perturbation
suite instead, and justified it as reproducing "the incident this repository
recorded". That justification was false twice over and the correction matters
more than the choice:

- `ERRATA.md` says the contention incident was recorded by **a sibling
  project's** harness, that its log is not published here, and that what this
  repository attests is only the absence. A plan may not cite as its own
  measurement a thing its own errata disclaims.
- `bench/host_guard.py`'s own notes say the perturbation suites are sequential,
  that neither can produce the load on its own, and that attributing the burst
  to them "would have been a plausible story that no measurement supports, which
  is the mistake ERRATA A12 was written about".

And the suite could not have delivered the treatment in any case.
`tests/data_mutate.py` calls `host_guard.protect()`, and `protect()` runs
`limit_threads()` and `be_nice(10)` **before** its `BENCH_ALLOW_CONTENDED` early
return, so the load would have arrived at nice 10 with one BLAS thread: a
treatment engineered by the very guard it was invoking not to interfere. It also
leaks its scratch when killed, and its shards inherit the whole-host lock file
descriptor, so a second start inside the same run fails immediately and silently
leaves the block unloaded.

Spinners have a narrower footprint and this plan states the consequence rather
than dressing it up: **this is a CPU time-share intervention only.** Page cache
and disk are untouched, so a null result does not cover them, and the
page-cache hypothesis stays exactly where A16 left it, as the successor run.

One level, and one only. A dose response is a separate run and is warranted only
if this one is positive.

## What is recorded, and the check that the treatment arrived

Beside the GPU telemetry every run here already carries, pinned to `raw` at one
second:

- `bench/host_guard.py --sample` at one second for the whole invocation
- the **runqueue wait** of the serving process, from `/proc/<pid>/schedstat`'s
  second field, at each tick
- the processor each of the server's threads last ran on, from
  `/proc/<pid>/task/*/stat`, at each tick, as the check that the pinning held
- `/proc/meminfo`'s `Cached` at each arm-run boundary
- wall-clock start and end of every arm-run

**Placement is pinned, not merely recorded.** Recording it is not enough and the
previous version of this plan implied it was. Applying load changes which
processors are free, so placement is a post-treatment mediator: conditioning on
it afterwards to recover a direct effect needs an assumption about unmeasured
mediator-outcome confounding that nothing here supports, and if the load
displaces the server systematically there is no within-stratum comparison left
to make at all. The recorded column would then document the confound without
separating it. The record is also an undersampled proxy: the field says where a
thread last ran, not where it spent its time, and a decode step is a few
milliseconds against a one-second tick. So the server is pinned to a fixed
processor set in every arm-run, the load's processor set is a design factor
rather than whatever the scheduler decides, and the recorded column becomes a
check that the pinning held rather than a covariate to adjust for.

**The manipulation check is on interference, not on presence.** The previous
version gated on the load's own CPU percentage, which certifies that something
was running, not that it reached the thing being measured — and a load niced out
of contention passes such a gate while delivering nothing. The gate is instead
that the server's runqueue wait per unit of decode time is at least an order of
magnitude higher in loaded arm-runs than in unloaded ones, with the threshold
fixed from a calibration pass before the run and written into this document
before the run proper starts. A run that fails it is **void, not negative**:
without this, a load that failed to arrive produces two arms holding still,
which selects the branch that retires A16's last testable hypothesis here.

Two further gates. If an arm-run fails, its whole pair is dropped, because half
a pair is not a paired observation; a pair is dropped when either member has a
non-null `crashed`, or fewer rows than the prompt set. Fewer than eighteen
surviving pairs for any arm voids that arm's reading.

Two things the sampler does not see, stated so a null is not read as covering
them: `/proc/stat`'s idle and iowait are summed, so a load that is mostly disk
wait records as low `busy_pct`; and per-process accounting only sees processes
alive at two consecutive ticks, so a process born and reaped inside one second
contributes nothing.

## The prediction, fixed before the data

### Per arm

For each arm, take its surviving pairs; for each pair compute the natural log of
the loaded pooled decode rate over the unloaded pooled decode rate of the same
arm in the same block; take the mean, its Student t interval on **(surviving
pairs minus one)** degrees of freedom, and exponentiate. Pooled decode rate is
generated tokens over decode milliseconds summed across the arm-run's prompts,
which is the definition the rest of this repository uses. The point estimate is
the centre of that interval and not a separately pooled figure.

Percentage change and not percentage points: A16's own figure is −3.40 % of a
decode rate, and a threshold has to be in the unit of the thing it explains.

If host CPU load is the mechanism behind A16, `spec-dflash-n2` **moves** and
`spec-draft-n8` **holds**, as defined below. That is a necessary condition and
not a test: the next heading shows every arm-agnostic mechanism this plan can
name produces the same pattern, so it selects nothing on its own and the fit is
what decides.

### Across arms

On run T4's own pooled means the three arms decode at 142.416, 116.100 and
30.823 tok/s, which is 7.022, 8.613 and 32.443 ms per generated token: a spread
of **4.62 times**. A uniform increase in host cost per token therefore produces
very different percentages, and the first version of this plan predicted exactly
that pattern and read it as specific to one configuration.

"The same cost per token" is only one arm-agnostic hypothesis. Two others are as
mechanically plausible and their denominators are recorded per arm-run: a cost
per model forward pass, and a cost per target-model step. **The round count is
`drafted / draft-max`, and getting that wrong is what invalidated the previous
version of this section.**

It had used generated minus accepted, on the reasoning that a round drafting k
tokens with a accepted yields a+1 tokens. That is true of the mechanism and
false of the counter. ERRATA A1 quotes the server source: on partial acceptance
the checkpoint-and-restore branch returns **before** the accepted counter is
incremented, so on any arm taking that branch the server under-counts. A13
measures it on these two arms: `spec-dflash-n2` reads 72.8 % against the
drafter's 73.0 % with zero checkpoints, and `spec-draft-n8` reads 29.7 % against
41.3 % with 772. Drafted over draft-max is exact for both, because the drafter
always proposes its maximum: 14646 over 2 and 33408 over 8 are both whole
numbers, and the acceptance they imply is 72.90 % and 41.38 % against A13's
73.0 % and 41.3 %.

The check that let the wrong reading through was "drafted per round does not
exceed the arm's draft maximum". The wrong round count satisfies it too, at
4.109 against a maximum of 8. **Integrality is the test that separates them**,
and it is held as a regression now.

Scaling each hypothesis so that `spec-dflash-n2` lands at exactly −2 %:

| arm-agnostic cost | `spec-dflash-n2` | `baseline` | `spec-draft-n8` |
|---|---:|---:|---:|
| per generated token | −2.00 % | −1.64 % | −0.44 % |
| per model forward pass | −2.00 % | −1.34 % | −0.75 % |
| per target-model step | −2.00 % | **−3.93 %** | −0.25 % |

`spec-draft-n8` holds under all three, at −0.44, −0.75 and −0.25 per cent, every
one inside the holding band. So "the arm moves and the control holds" is
satisfied by every arm-agnostic mechanism this plan can name and discriminates
nothing on its own.

**And a single contrast cannot fix it either.** The previous version replaced
that prediction with D, the within-block difference in milliseconds per token
between `spec-dflash-n2` and `spec-draft-n8`, and pre-registered the
arm-specific branch as the one-sided claim that D lies wholly above zero. That
worked only while every hypothesis put D on the same side of zero, which was an
artefact of the wrong round count. With the drafter's count the three give
+0.000, −0.102 and **+0.062** ms per token: a purely arm-agnostic
per-target-step cost now produces exactly the signature the rule reserved for
arm-specificity.

**So the test is a fit, not a contrast.** Each row above is one free scale over
fixed per-arm weights, so with three arms it leaves two degrees of freedom and
can be rejected on its own. Measure the change in milliseconds per generated
token for each arm, with its paired interval; for each hypothesis fit the single
scale by weighted least squares and compute the chi-square of the residuals on
two degrees of freedom.

- **every hypothesis rejected**: the cost is not any arm-agnostic form this plan
  can name, and the effect is specific to the configuration
- **exactly one survives**: that is the mechanism, and the outcome section names
  it
- **more than one survives**: the run cannot distinguish them and says so

T4's own step already fails all three, at chi-square 89, 98 and 129 on two
degrees of freedom against a one per cent critical value of 9.21, because the
arms moved in opposite directions across it. Those standard errors are the
paired-block ones the three low and three high blocks give; an earlier version
of this paragraph quoted 46, 40 and 42, computed from a standard error invented
as a quarter of each arm's own change, which makes the statistic a function of
the number that was chosen. The split point there was chosen by eye, which is
why this run fixes it in advance.

### Why 2 % and not 3.93 %

The prediction is that the mechanism is available on this host, not that this
run reproduces the size of a step measured on another day under an unknown
ambient load. A recovered effect smaller than the incident is the expected shape
of a positive result; a threshold at the incident's own size could only be met
by a coincidence. The power table is computed against both.

## What would falsify what

Two words, defined on the **interval** so that both are reachable and neither
is an absence-of-evidence claim, and **two sided**, because a load that makes
an arm faster is a real finding and the first version of this plan filed it as
undecided:

- an arm **moves** when its whole interval lies outside ±1 %; the sign is
  reported with it
- an arm **holds** when its whole interval lies inside ±1 %
- otherwise it does neither, and that is an outcome with a name

**The arm moves and every arm-agnostic fit is rejected.** The mechanism exists
on this host and it is not any host cost this plan can name. This does not
establish that ambient load caused the steps A16 recorded: those runs have no
load column and cannot be revisited. What it establishes is that the column has
to exist from now on, and that every decode-rate comparison **on this host** is
exposed to it. The other hosts carry different toolchains and the question is
open on each of them separately.

**The arm moves and one arm-agnostic fit survives.** Host load costs this
machine a fixed amount per token, or per forward pass, or per target-model step,
and the outcome section names which row survived and reports its chi-square.
Worth recording, and not A16's shape. If more than one survives the run cannot
tell them apart and says that instead.

**Both named arms move.** General starvation of the serving process, whatever
the fit says. This branch is selected before the fit is read, because two arms
moving is not a pattern any single host cost produces at these rates, and the
outcome section reports the fit beside it rather than choosing between them.

**Both named arms hold.** Host CPU load at this level is not the mechanism, and
no fit is read because there is nothing to fit. Of the
hypotheses A16 names, the junction temperature stays untestable here, and the
page-cache and allocator-state one is untested rather than excluded; it is the
successor run and is listed in `RETEST_TODO.md` as such. This branch does not
license the sentence that A16 has no testable hypothesis left, which is what
the first version of this plan pre-committed to and which A16's own text
contradicts.

**Anything else: inconclusive, and it licenses nothing.** An arm landing between
the thresholds, or an interval that spans them, is a run that did not decide.
The entry it produces says the run was made and did not decide. It does not get
read as weak support, it does not get rescued by pooling the pairs a different
way, and it does not get a second analysis chosen after seeing it. What it
licenses is a decision about whether a longer run or a higher load level is
worth the hours.

`baseline` does not select the branch, because the prediction was not written
about it. It enters the normaliser table, where it is the arm that separates a
per-target-step cost from the other two. If `baseline` itself lands between the
thresholds, that is reported and the normaliser table still decides.

## What must change before this run can start

Six things, each small, each named. The run does not start until they land,
and the reason each is here is that an adversarial pass found the run producing
nothing without it.

1. **The sampler cannot follow the server.** `bench/retest_runner.py` starts a
   fresh `llama-server` inside `run_arm` and stops it in the same function, so
   run X has one server per arm-run and about a hundred and fifty pids.
   `host_guard.sample()` latches one `root_pid` once and compares its start time
   each tick, so after the first server exits it reports `root-gone` forever and
   every process, including each new server, lands in `other_pct`. A
   `--root-pid-file` mode that re-reads the pid each tick fixes it. Without this
   the run fails its own gates silently and looks like a careful instrument
   refusing bad data.
2. **The two members of a pair collide on disk.** Arm-runs are written as
   `{arm}__rep{rep}.json`, so the unloaded run overwrites the loaded one and
   `validate_run` then reports every arm-run recorded twice and writes
   `RUN_FAILED.json`. The loaded variant needs to be a distinct arm key; the
   suffix machinery the hard-cap runs use is the precedent.
3. **There is no order mode that keeps a pair adjacent.** `build_schedule`
   refuses to call anything balanced that it does not generate, and none of its
   modes produces "rotate the arms, and run each arm twice back to back with the
   treatment order alternating".
4. **The manipulation check has no join key.** Arm-runs record durations and the
   sampler records wall-clock times, and there is no recorded wall-clock
   start or end per arm-run to join them on. Two lines in `run_arm`.

5. **Nothing pins any process to any processor.** There is no `taskset`, no
   `sched_setaffinity` and no cpuset anywhere in `bench/` or `analysis/`, while
   this plan mandates that the server run on a fixed processor set in every
   arm-run and that the load's set be a design factor. It is one line where the
   server command is built, and the treatment then records itself through the
   `argv` field every arm-run already carries. Two earlier versions of this list
   left it out while the body of the plan required it.

6. The server's thread count has to be passed explicitly and recorded,
because llama.cpp's own choice is the variable that decides whether any of this
work reaches an efficiency core, and the tree does not record what it was.

One more is not a prerequisite but is worth fixing while these are open:
`host_guard.protect()` applies `limit_threads()` and `be_nice()` before its
`BENCH_ALLOW_CONTENDED` escape, so a caller that has declared its contention
still gets de-prioritised. Whatever the escape is for, it is not that.

## What this run cannot do

It cannot explain the historical steps, only show whether the mechanism is
available. It is one host, one binary, one set of model files, one arm known to
move, and one load level. Nothing from it transfers to another card, and the
fleet's hosts carry different toolchains, so only within-host contrasts compare.

It cannot separate CPU contention from chassis heating by itself. Busy
processors warm the case and therefore the card's inlet, and the quantity that
would mediate that is GDDR6X junction temperature, which is A16's other
unmeasured quantity and is unreadable here. No CPU package temperature is
recorded anywhere in this repository either. The order alternation is a
mitigation, not a separation. The contrast that would separate them is a third
level with the load pinned to processors the server does not use, which heats
the box identically and removes the contention; it is named here so that a
positive result is not read as more than it is.

The configuration is pinned to run T4's: the same model files by SHA-256, the
same `--fit-target 3072`, the same ten prompts, seed 42, concurrency 1, all
recorded in the run's manifest. One difference is deliberate. T4 used the
split-timer instrumented build, which adds host-side work and is therefore not
neutral in an experiment about host-side cost, so run X uses the stock build and
the variability figures above are an estimate for run X rather than a
measurement of it.

The wall-clock figure is a floor, not a budget. Twenty-four blocks of six
arm-runs is 9600 s of `ready_s` plus summed request time, which is two hours
forty minutes. Run T4's own invocation prices what that leaves out: 1365 s of
wall clock against 1198.8 s of `ready_s` plus request time, which is **9.23 s
per arm-run** of warm-up, teardown settle, two `nvidia-smi` calls and the JSON
writes. At 144 arm-runs that is 3.03 hours, and the washout this plan mandates
is per pair, one pair per arm per block, so 72 of them at 30 s add another 0.60.
**3.63 hours is the honest figure.** An earlier version of this paragraph
costed the washout at 20 s while the design specified 30 and published the total
that gave; `analysis/load_run_power.py` derives it now, so changing one changes
the other.

The run's data, manifest and telemetry are committed together with the outcome
section filled in, in one commit, so that the plan and the result cannot drift
apart.

## Where this document sits, and what it will cost

It is in `analysis/table_coverage.py`'s excluded list rather than its censused
one. The reason recorded there is the release, and not the genre: three
prospective plans of exactly this kind are censused, and one of them,
`PROSPECTIVE_ANALYSIS_PLAN_W2.md`, is censused today carrying no tables of its
own — run W2's data is committed, but the plan document holds none of it.

What excludes this one is that `analysis/verify_claims.py` pins the number of
censused documents at eleven and the decimal prose census at its current pair
of values, and that checker is one of the six files
`bench/check_release_binding.py` compares between the `v4.2` tag and HEAD. Note
where the freeze bites: `table_coverage.py` is not itself bound, so growing its
list is not editing a frozen file. It is editing an unfrozen one in a way that
makes a frozen one fail, which cannot then be fixed without changing the frozen
one, and the tag would stop publishing the tree that verifies it. This document
also quotes decimals from A16, so censusing it would move the decimal census as
well as the count.

When run X produces data, that changes. The honest accounting is that the census
is not what costs anything: run X's raw logs go into `EVIDENCE_MANIFEST.sha256`
and its entry into `RUN_REGISTRY.json`, both of which are bound, so committing
the evidence re-cuts the release binding whatever happens to this file.
Censusing it then rides along at no additional cost.

The cost this document's own commits do incur is a coverage probe re-run each
time: adding lines to a censused document moves the table line numbers the
existing attestations pin, so all thirty-two shards have to be produced again.

# Outcome

Not run yet. This section is filled in when the data is committed, and until
then its emptiness is the point.
