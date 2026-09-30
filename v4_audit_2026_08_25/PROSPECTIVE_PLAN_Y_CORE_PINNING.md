# Prospective plan for run Y: what a quarter less processor clock costs this arm

Written and committed before the run. The outcome section at the end is empty
until the data exists.

This is the run the plan for run X says should happen first, and it is cheaper
than run X by a factor of five. If it comes back with a tight enough bound, run X
is unnecessary.

## The question

ERRATA A16 has one arm, `spec-dflash-n2`, sitting at two levels 3.93 % apart and
moving between them inside a single invocation, with byte-identical output, and
with core temperature, SM clock, board power, the preceding arm, the fitter's
decisions, free GPU memory and time-to-healthy all ruled out by measurement.

The bench host is an i9-13900K. Sixteen of its thirty-two logical processors are
efficiency cores at 4300 MHz; the other sixteen are performance cores at 5500,
four of them at 5800. Nothing in `bench/` pinned any process to any processor
until the commit that carries this file, and no run in this repository records
which kind of core anything ran on. So core placement has been an uncontrolled,
unrecorded variable whose magnitude is about a quarter of the clock, whose
timescale is a scheduler's, which produces byte-identical output, and which
`nvidia-smi` cannot see.

That is A16's signature. This run does not test whether placement caused those
steps, which is not testable: the runs have no placement column and cannot be
revisited. It puts a **bound** on how much placement could ever have mattered, by
forcing the whole server onto the slow cores and measuring what that costs.

## Why a bound and not a verdict

The plan for run X specifies this as "a null inside one per cent". Simulated
against the measured variability of this arm, six blocks reach that reading only
0.355 of the time when the true effect is exactly zero: two runs in three would
come back inconclusive. That is the same defect the first draft of run X had, and
it is the reason this file exists rather than a paragraph.

A bounding experiment wants an upper confidence limit, which is always
computable. "The penalty is at most X" has no inconclusive branch.

## The design

One invocation. Twelve blocks. Two arms, each run twice per block:

- `spec-dflash-n2`, the arm A16 is about
- `baseline`, no speculation, which has no drafter and no acceptance check and
  therefore much less host-side work per token

Each arm-run is pinned. The two conditions are:

| condition | processors | kind | clock |
|---|---|---|---|
| fast | 0, 2, 4, 6, 8, 10, 12, 14 | one thread of each of the eight performance cores | 5500, two at 5800 |
| slow | 16 to 23 | eight efficiency cores, which have no second thread | 4300 |

Eight threads either way, eight distinct physical cores either way, `-t 8` and
`-tb 8` passed explicitly. Only the kind of core differs. Both sides are pinned,
because pinned against unpinned changes two things at once.

The slow condition comes first in half the blocks and second in the other half,
alternating, so that whatever the first run of a pair leaves behind falls on both
sides equally. The pair members are one arm-run apart, which is the separation
the run X plan argues for and about forty to a hundred seconds here.

`taskset` execs the server in place, so the pid the driver waits on is the
server's. Every arm-run records the cpu list the driver asked for **and**
`Cpus_allowed_list` as the kernel reports it, because a treatment recorded as
requested rather than as applied is a treatment nobody checked.

### Why twelve blocks

Two arms times two conditions times twelve blocks is forty-eight arm-runs, about
0.68 hours from run T4's own arm-run durations plus its measured per-arm-run
overhead. Six blocks is 0.34 and eighteen is 1.02.

From the measured adjacent-arm-run difference SD of 0.760 % for this arm, the
one-sided ninety-five per cent bound each gives, if the point estimate lands at
zero:

| blocks | hours | bound | two-sided reading reaches "holds" under a true null |
|---|---:|---:|---:|
| 6 | 0.34 | 0.62 % | 0.355 |
| **12** | **0.68** | **0.39 %** | **0.554** |
| 18 | 1.02 | 0.31 % | 0.759 |

Twelve is the choice. Six settles the question this run is for, because 0.62 % is
six times below A16's step, but it fails the secondary reading two times in three
and twenty more minutes halves the bound.

## The prediction, fixed before the data

For each arm: the twelve within-block log ratios of its slow-condition pooled
decode rate to its fast-condition rate, as a percentage change; their mean; and a
Student t interval on eleven degrees of freedom. Pooled decode rate is generated
tokens over decode milliseconds summed across the arm-run's ten prompts, which is
the definition the rest of this repository uses.

**Primary, and always computable: the one-sided ninety-five per cent upper limit
on the slowdown.** The claim it supports is of the form "forcing this server
entirely onto cores a quarter slower costs at most X per cent".

**Secondary: the two-sided reading the run X plan asked for.** An arm holds when
its whole interval lies inside plus or minus one per cent, moves when the whole
interval lies outside it, and otherwise does neither.

I expect both arms to hold. This arm offloads 41 of 41 target layers and 9 of 9
drafter layers, so the host does orchestration and an acceptance check and not
arithmetic, and a quarter off the clock of that orchestration should be a small
fraction of 6.9 milliseconds per token. `baseline` does even less on the host and
should move less still. The run is worth making because that expectation is an
argument and the column has never existed.

## What each outcome licenses

**The bound is below A16's 3.93 % step.** Then a full efficiency-core pinning
cannot produce that step, and ambient placement drift, which is a fraction of a
full pinning, cannot either. Core placement is excluded as an explanation of
A16, and with it the whole host-processor-speed family: clock, placement, and
ambient load acting through the scheduler, because all three reach this arm only
by changing how fast its host-side work runs. **Run X becomes unnecessary** and
the plan for it should say so.

**The bound is above the step, or either arm moves.** Processor speed reaches
this workload by enough to matter, the column has to exist from now on, and run X
becomes worth its three and a half hours because CPU contention is then a live
channel rather than a bounded one.

**`baseline` moves and `spec-dflash-n2` holds.** Not a pattern any account of
this predicts, and it would mean the contrast is measuring something other than
host-side cost. The outcome section would say the run did not behave and why that
matters, and nothing would be retired.

There is no inconclusive branch for the primary reading, which is the point of
choosing a bound.

## What this run cannot do

It cannot show that placement caused A16's steps. Those runs have no placement
column.

It is one host, one binary, one set of model files, one thread count and one pair
of core sets. Nothing transfers to another card, and the fleet's hosts carry
different toolchains.

Its absolute rates are not comparable with any earlier run here, because no
earlier run was pinned at all and none passed an explicit thread count. Only the
within-run contrast is a measurement; the levels are not.

And it runs the stock master build, not the split-timer build run T4 used, which
adds host-side instrumentation. The variability the power arithmetic above rests
on was measured on T4, so it is an estimate for this run and not a measurement of
it. If the observed pair-to-pair spread comes back wider than 0.760 %, the bound
widens with it and the outcome section says so rather than quoting the number
planned for.

It bounds processor speed. It does not bound memory bandwidth contention, page
cache or allocator state, which A16 names separately and which stays the
successor run.

## Where this document sits

In `analysis/table_coverage.py`'s excluded list, for the reason recorded there
and for the same reason the run X plan is: `analysis/verify_claims.py` pins the
number of censused documents and the decimal prose census, and that checker is
bound to the `v4.2` tag. When this run's data lands it joins the censused set,
which needs the binding re-cut, and committing the evidence re-cuts it anyway.

# Outcome

Run on 2026-09-26, forty-eight arm-runs of forty-eight, one invocation of
forty-one minutes. The data is in
[`v5_pinning_2026_09_26/`](../v5_pinning_2026_09_26/README.md), which is a round
of its own because the frozen claim checker pins the run-directory count under
this archive and is bound to the `v4.2` tag.

Every arm-run's `Cpus_allowed_list`, read from `/proc`, is the set the driver
asked for, and the two conditions ran on disjoint processors:

    spec-dflash-n2         0, 2, 4, 6, 8, 10, 12, 14
    spec-dflash-n2-ecore   16 .. 23
    baseline               0, 2, 4, 6, 8, 10, 12, 14
    baseline-ecore         16 .. 23

## The pre-registered reading

| arm | fast | slow | change | 95 % interval | one-sided upper limit |
|---|---:|---:|---:|---|---:|
| `spec-dflash-n2` | 140.327 | 120.068 | −15.57 % | [−16.90 %, −14.25 %] | 16.66 % |
| `baseline` | 114.664 | 97.763 | −15.94 % | [−16.76 %, −15.12 %] | 16.61 % |

Both arms **move**. Both bounds are far above A16's 3.93 % step.

**The prediction above was wrong.** It said "I expect both arms to hold", on the
argument that an arm offloading 41 of 41 target and 9 of 9 drafter layers leaves
the host doing orchestration rather than arithmetic, and that a quarter off the
clock of that orchestration should be a small fraction of 6.9 milliseconds per
token. It is not a small fraction. It is about half.

This is the plan's second branch: "Processor speed reaches this workload by enough
to matter, the column has to exist from now on, and run X becomes worth its three
and a half hours because CPU contention is then a live channel rather than a
bounded one."

## What makes it speed rather than work, and the host rather than the card

Three readings, each independent of the estimator:

- the output is identical to the token. 36 000 generated in both conditions, and
  29 292 drafted and 21 192 accepted in both conditions of the speculative arm
- the GPU's SM clock is the same or slightly **higher** in the slow condition,
  1928.8 MHz against 1920.0, so nothing is thermally throttled
- its power, utilisation and temperature all **fall** in the slow condition:
  326.8 W to 302.4, 75.2 % to 56.1, 73.8 °C to 71.2. The card is idle-waiting

Taking the clock ratio at face value, 4300 against 5800, the implied share of a
token's time that is host work scaling with clock is about 48 % for
`spec-dflash-n2` and 50 % for `baseline`. That is an inference from one ratio and
not a measurement: efficiency cores differ from performance cores in more than
clock, having no second thread and a different cache hierarchy, so the share is an
upper reading.

## What this does to run X, and what it does not settle

The plan for run X said a null here would make it unnecessary. The opposite
happened, so run X is warranted, and the section of that plan which recommended
this run first is corrected rather than left standing.

It does not settle A16. A16's step is 3.93 %, which is 0.23 of a full
displacement, or roughly two of the eight threads landing on efficiency cores.
Every published run here passes no thread count and no affinity, llama.cpp chooses
eight threads of the thirty-two logical processors, and each arm-run starts a
fresh server and therefore draws its own placement. That is A16's shape, per
arm-run and sparing its neighbours, and the right order of magnitude. It is an
arithmetic coincidence until a run sets the number of displaced threads and
measures the response, which is the successor and is listed in `RETEST_TODO.md`.

## Two things about the run itself, recorded because they are host work

The sampler was not stopped when the run ended, and two of them were running
during it rather than one. Both are described in the round's README with the
defect that caused it and the fix. Each is one `nvidia-smi` query per second and
the conditions alternate within every block, so neither can bias the contrast.
