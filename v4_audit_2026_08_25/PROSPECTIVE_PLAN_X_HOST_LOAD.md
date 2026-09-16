# Prospective plan for run X: does host CPU load move the arm A16 is about

Written and committed before the run. A hypothesis picked after looking at the
data is not a hypothesis, and the outcome section at the end of this file is
empty until the data exists. This repository has a retracted headline that
asserted a mechanism chosen after the fact, which is the habit the genre exists
to break.

**This plan cannot be executed yet.** Two adversarial passes over it found four
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
`/proc/cpuinfo`
carries no hypervisor flag, and `/proc/stat`'s steal column is zero. Hypervisor
descheduling, which would look exactly like A16 and is invisible to every guest
counter, is therefore not available as an explanation here. The development box
these plans are written on IS a KVM guest with a live steal column, which is why
the distinction is worth writing down.

It is an i9-13900K: thirty-two logical processors across three frequency tiers,
5800 MHz on four of them, 5500 MHz on twelve and **4300 MHz on sixteen**. The
slow sixteen are efficiency cores, and the spread between tiers is twenty-two to
twenty-six per cent.

Nothing in `bench/` or `analysis/` pins any process to any processor: there is
no `taskset`, no `sched_setaffinity`, no cpuset. And no run in this repository
records which processor anything ran on.

So core placement is an uncontrolled and unrecorded variable whose magnitude is
the right order, whose timescale is a scheduler's, which produces byte-identical
output, and which `nvidia-smi` cannot see. That is A16's signature. This plan
does not claim it is the cause — the arm's step persisted across three
consecutive fresh server processes, which a single migration does not explain —
but it is cheap to record and run X records it. It also matters for the design:
**applying CPU load changes core placement**, so a positive result in a run that
did not record placement would not be attributable.

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

`analysis/load_run_power.py` is committed with this plan and produces the table
below; it is not a figure typed in after reading a terminal. Its inputs are
measured from run T4's committed arm-runs:

- the per-arm-run residual log SD is **0.537 %**, which is the measured
  adjacent-repeat difference SD of 0.760 % with the step excised, over root two
- the telegraph gap is the measured **3.93 %**
- the switch hazard is **one per thousand seconds**: T4 shows one transition
  across five adjacent-repeat gaps of about two hundred seconds each. A16 states
  no switching rate and this figure is not attributed to it
- arm-run durations 39.8, 116.0 and 44.1 s, so a three-arm pass is 200 s

Those three decompose the block-level variability correctly, which is the error
the first version of this table made: T4's block CV of 2.145 % **already
contains the step**, and a simulation that adds a telegraph on top of it implies
2.768 % of block spread, twenty-nine per cent more than T4 shows. The corrected
figures, by the decision rule defined below, forty thousand draws:

| design | truth −2 % | truth −3.93 % | truth 0, declares holds |
|---|---:|---:|---:|
| between-block, 6 against 6 (first draft) | 0.34 | 0.84 | 0.15 |
| within-block, 12 pairs | 0.82 | 1.00 | 0.67 |
| within-block, 18 pairs (the void floor below) | 0.92 | 1.00 | 0.85 |
| **within-block, 24 pairs (this plan)** | **0.97** | **1.00** | **0.94** |

At half the hazard the last row is 0.99 and 0.98; at double it, 0.92 and 0.83.
The first draft's design reaches 0.21 to 0.48 against a true −2 % across that
same range. It is not hopeless, which is what the first version of this section
wrongly implied by publishing 0.09; it is inadequate against the threshold it
set itself, and the redesign is worth its cost for that reason and not for a
more dramatic one.

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
a
measured covariate instead of an argument.

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
  `/proc/<pid>/task/*/stat`, at each tick
- `/proc/meminfo`'s `Cached` at each arm-run boundary
- wall-clock start and end of every arm-run

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
`spec-draft-n8` **holds**, as defined below.

### Across arms

On run T4's own pooled means the three arms decode at 142.416, 116.100 and
30.823 tok/s, which is 7.022, 8.613 and 32.443 ms per generated token: a spread
of **4.62 times**. A **uniform** increase in host cost per token, with no
preference for any arm, therefore produces very different percentages, and the
first version of this plan predicted exactly that pattern and read it as
specific to one configuration.

It is worse than that, and the correction is the sharpest thing in this
document. "The same cost per token" is only one arm-agnostic hypothesis. Two
others are as mechanically plausible, and their denominators are already
recorded per arm-run: a cost per model forward pass, and a cost per target-model
step. A round that drafts k tokens and has a accepted yields a+1 tokens, so over
a run the round count is generated minus accepted exactly, which makes both
denominators derivable rather than assumed. `analysis/load_run_power.py`
computes the table; scaling each hypothesis so that `spec-dflash-n2` lands at
exactly −2 %:

| arm-agnostic cost | `spec-dflash-n2` | `baseline` | `spec-draft-n8` | D |
|---|---:|---:|---:|---:|
| per generated token | −2.00 % | −1.64 % | −0.44 % | +0.000 |
| per model forward pass | −2.00 % | −1.34 % | −0.83 % | −0.127 |
| per target-model step | −2.00 % | **−3.89 %** | −0.48 % | −0.014 |

Read the third column. **`spec-draft-n8` holds under all three**, at −0.44,
−0.83 and −0.48 per cent, every one of them inside the holding band. So the
prediction "the arm moves and the control holds" is satisfied by every
arm-agnostic mechanism this plan can name, and on its own it discriminates
nothing. The first version of this plan rested its headline on exactly that
pattern.

Two things do discriminate, and both are pre-registered here.

**D, the within-block difference of differences.** For each block, form

    D = (delta ms per token for spec-dflash-n2) - (delta ms per token for
spec-draft-n8)

and put one Student t interval on the mean of those. Every arm-agnostic
hypothesis in the table puts D at or below zero, so the arm-specific claim is
the one-sided statement that **D's interval lies wholly above zero**. This is
formed within blocks, so block-level noise shared between the arms cancels in D,
and it is one interval with one meaning rather than three intervals compared by
whether they overlap. Overlapping intervals is a conservative test of difference
being used as a test of equality, and it is favoured by noise: the wider the
intervals, the more confidently it would assert that nothing is arm-specific.

**`baseline`'s magnitude, which names the agnostic hypothesis.** If D does not
exclude zero, the three rows are separated by what `baseline` did: about −1.6 %
for a per-token cost, −1.3 % for a per-forward cost, and −3.9 % for a per-target
step cost, the last being the only hypothesis under which `baseline` moves more
than the arm this run is about. The outcome section names the row.

### Why 2 % and not 3.93 %

The prediction is that the mechanism is available on this host, not that this
run reproduces the size of a step measured on another day under an unknown
ambient load. A recovered effect smaller than the incident is the expected shape
of a positive result; a threshold at the incident's own size could only be met
by a coincidence. The power table is computed against both.

## What would falsify what

Two words, defined on the **interval** so that both are reachable and neither is
an absence-of-evidence claim, and **two sided**, because a load that makes an
arm faster is a real finding and the first version of this plan filed it as
undecided:

- an arm **moves** when its whole interval lies outside ±1 %; the sign is
  reported with it
- an arm **holds** when its whole interval lies inside ±1 %
- otherwise it does neither, and that is an outcome with a name

**The arm moves, the control holds, and D excludes zero.** The mechanism exists
on this host and it is not a uniform per-token cost. This does not establish
that ambient load caused the steps A16 recorded: those runs have no load column
and cannot be revisited. What it establishes is that the column has to exist
from now on, and that every decode-rate comparison **on this host** is exposed
to it. The other hosts carry different toolchains and the question is open on
each of them separately.

**The arm moves and D contains zero.** Host load costs this machine a fixed
amount per token, or per forward pass, or per target step, and the outcome
section says which. Worth recording, and not A16's shape.

**Both move.** General starvation of the serving process.

**Both hold.** Host CPU load at this level is not the mechanism. Of the
hypotheses A16 names, the junction temperature stays untestable here, and the
page-cache and allocator-state one is untested rather than excluded; it is the
successor run and is listed in `RETEST_TODO.md` as such. This branch does not
license the sentence that A16 has no testable hypothesis left, which is what the
first version of this plan pre-committed to and which A16's own text
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

Four things, each small, each named. The run does not start until they land,
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
   sampler records wall-clock times, and there is no recorded wall-clock start
or
   end per arm-run to join them on. Two lines in `run_arm`.

A fifth is not a prerequisite but is worth fixing while these are open:
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
forty minutes. It excludes the per-arm warm-up, which is two to ten seconds and
is recorded nowhere, the teardown settle of about two seconds per arm-run, two
`nvidia-smi` calls per arm-run, the load's start and stop, and a longer time to
become healthy under load. Three hours is the honest planning figure.

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
censused documents at eleven and the decimal prose census at its current pair of
values, and that checker is one of the six files
`bench/check_release_binding.py`
compares between the `v4.2` tag and HEAD. Note where the freeze bites:
`table_coverage.py` is not itself bound, so growing its list is not editing a
frozen file. It is editing an unfrozen one in a way that makes a frozen one
fail, which cannot then be fixed without changing the frozen one, and the tag
would stop publishing the tree that verifies it. This document also quotes
decimals from A16, so censusing it would move the decimal census as well as the
count.

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
then
its emptiness is the point.
