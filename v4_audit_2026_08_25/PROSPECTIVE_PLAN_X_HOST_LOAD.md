# Prospective plan for run X: does host CPU load move the arm A16 is about

Written and committed before the run. A hypothesis picked after looking at the
data is not a hypothesis, and the outcome section at the end of this file is
empty until the data exists. This repository has a retracted headline that
asserted a mechanism chosen after the fact, which is the habit the genre exists
to break.

An adversarial review of the first draft of this plan killed its design twice
over, and both objections are answered below rather than quietly dropped. The
first draft compared six loaded blocks against six unloaded ones and predicted
that one arm would slow by at least 2 % while another held. Against this arm's
own measured block-to-block variability that design had a power of **0.09**, so
its most likely product was "inconclusive" even if its hypothesis was exactly
right, and roughly one time in five it would have reported a real effect as the
arm holding still. The second objection was worse, because it survived any
sample size: the predicted signature is what a mechanism with **no arm
specificity whatsoever** produces. Both are worked out below, with the numbers.

## Why this run exists

[A16](../ERRATA.md#a16-two-runs-identical-in-every-recorded-respect-and-byte-identical-in-output-differ-by-34--on-one-arm)
ends by naming two quantities its instruments could not see. One of them, the
GDDR6X memory-junction temperature, needs a sensor NVML does not expose on Linux
and cannot be read on this host at all. The other is host CPU load.

`bench/host_guard.py --sample` records it, and has since commit `81b30fe` on
2026-08-28. That date is not before every run here: run W2 was measured on
2026-08-30, with the sampler already sitting in the tree, and it does not carry
the column either. Nothing in this repository does, in any of the seventy-seven
committed run directories or in any of the seventeen committed telemetry traces
beside them, all of which hold `nvidia-smi` fields and nothing else.

So the second of the two named gaps is testable, the instrument is in the tree,
nothing has used it, and the reason is no longer that it did not exist.

## What A16 established, and what it did not

Taken as given here in A16's own words and figures, rather than re-derived:

- `spec-dflash-n2`'s decode rate sits at two levels about 3.5 % apart and moves
  between them on a timescale of minutes
- run T4 caught it stepping **inside one invocation** with the telemetry
  running: three repeats low, three high, a 3.5 % step partway through, while
  the two arms interleaved with it held at a CV of 0.12 % (`spec-draft-n8`) and
  0.55 % (no speculation)
- run T against run T3, two invocations two hours apart, moved the same arm by
  −3.40 % while `spec-draft-n8` moved −0.11 % and no speculation moved +0.79 %
- ruled out by measurement: core temperature, SM clock, board power, the
  preceding arm, the fitter's decisions, free GPU memory at start, and time to
  become healthy
- both levels produce byte-identical output and identical draft counts

The last two bullets of the first group are different measurements and this plan
keeps them apart. The within-invocation step is what run X is built around,
because run X is one invocation; the T-against-T3 figures are between
invocations and enter only as the size of the thing to be explained.

Not established: any cause. The instrument list that failed to distinguish the
levels was `nvidia-smi` alone.

One framing A16 withdraws, and this plan does not reinstate: pooled across runs
the distribution is clustering with a heavy low tail, not a clean two-state
system. "Two levels" is the entry's summary of what this one arm does within and
between invocations, not a claim about the pooled population.

A16 also names a third hypothesis that this run does **not** test, and the
outcome section is forbidden below from forgetting it: what sits between T and
T3 is machine history, two rebuilds and a killed rehearsal, which changes page
cache and allocator state and is captured by no field in this repository.

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

**The load is a within-block factor, not a between-block one.** Each arm is run
twice inside its block, once with the load and once without, back to back. That
is the single change that makes this run able to answer its question, and the
next two sections are the arithmetic for why.

The arm order advances by one each block, so each arm visits each position
eight times. Within a block the loaded run comes first in half the blocks and
second in the other half, alternating ABBA, so that whatever the first run of a
pair leaves behind in thermal state or allocator state falls on both sides of
the contrast equally.

Twenty-four blocks of six arm-runs is about **160 minutes** of wall clock, from
this repository's own measurement: run T4's arm-runs average 39.8 s, 116.0 s and
44.1 s including time to become healthy, so a three-arm pass is 200 s and a
doubled block is 400 s. The first draft's design was 40 minutes and could not
answer the question. This is the price of an answer and it is stated here rather
than discovered at block nine.

### Why a within-block factor, in numbers

A between-block factor is exposed to the very thing being studied. A16's arm
changes level on a timescale of minutes, and a block is minutes long, so the
arm's own state and the treatment vary on the same timescale and the design
cannot tell them apart.

The simulation below uses this repository's own measurements as its inputs: the
block-level CV of 2.15 % for `spec-dflash-n2` in run T4, and the standard
deviation of the log ratio of **adjacent** repeats in the same run, which is
0.76 % for that arm and 0.10 % for `spec-draft-n8` once the step itself is set
aside. Decision rule as defined under "What would falsify what", twenty thousand
draws, A16's telegraph switching with probability one in fifteen per pair
window:

| design | truth −2 % | truth −3.4 % | truth 0, declares holds |
|---|---:|---:|---:|
| between-block, 6 against 6 (first draft) | 0.09 | 0.32 | 0.00 |
| within-block, 12 pairs | 0.77 | 1.00 | 0.57 |
| **within-block, 24 pairs (this plan)** | **0.96** | **1.00** | **0.92** |

At a pessimistic switching rate of one in eight the twenty-four-pair row becomes
0.89 and 0.77, still adequate; the first draft's design does not reach 0.35 at
any of them. The control arm declares "holds" under the null at 0.99.

### What the block sequence still has to do

Pairing adjacent in time removes most of the exposure, not all of it, and the
order alternation is what handles the rest. Loaded-first and loaded-second
blocks are equal in number and their mean positions in the run are equal, so a
monotone drift over the two and a half hours cancels in the contrast, and the
sequence is a palindrome, so it cancels in wall-clock terms too even though
loaded blocks will take longer.

What no assignment can defend against is a state that happens to alternate in
step with the factor. That is why the decision rule below is built on an
interval and not on a point estimate.

## What the load is

The load is **this repository's own data-perturbation suite**,
`bench/run_data_mutations.sh` at a pinned shard count, started before a loaded
run and stopped after it.

The first draft specified a fixed number of CPU-busy spinners equal to the
processor count and justified it as matching the incident that motivated the
question. It does not match it. The incident this repository actually recorded
is two processes at 100 % and 97 %, and the suite that produced it spawns a
subprocess per perturbation, so it is a CPU **and** page-cache **and** disk
load with heavy process churn. A spinner reproduces one of those three, and
saturating every processor at nice 0 is not "a load of that order" but the
qualitative boundary beyond which every thread is time-shared one to one.

Using the pipeline itself also removes an argument about whether the model was
fair. It is the incident, at a pinned size, and how much of it actually arrived
is recorded rather than assumed.

The suite refuses to start on a host that is measuring, and it should: that
refusal is why it exists. Run X is the one case where the contention is the
declared treatment, so the driver starts it with `BENCH_ALLOW_CONTENDED=1`
scoped to that one child process, for the duration of one arm-run. The variable
is not exported for the session and no other guard is relaxed. A run that needs
the guard off for the session has stopped being an experiment about load.

One level, and one only. A dose response is a separate run and is warranted only
if this one is positive.

## What is recorded, and the check that the treatment arrived

`bench/host_guard.py --sample` at one second for the whole invocation, beside
the GPU telemetry every run here already carries, pinned to `raw` at one second
as well.

`--root-pid` is given **the llama-server's pid**, not the driver's. Attribution
is by descent, so with the server as the root the load generator is not a
descendant and its cost lands in `other_pct`, which is what the column has to
mean here: the incident being modelled was nobody's child. Rooted at the driver,
the treatment would be counted as the benchmark's own work and the trace would
show the run competing with itself. The sampler is likewise not a descendant of
the server, so its own small cost appears in `other_pct` in every block, loaded
and unloaded alike, and cancels in the contrast.

Those columns are percent of one processor, so they run past one hundred on a
host with many and are not shares of the same row's `busy_pct`, which is percent
of the whole host. A reader of the outcome section needs that sentence and the
first draft did not have it.

**The manipulation check, pre-registered as numbers.** Every loaded arm-run's
median `other_pct` must exceed 300, and every unloaded arm-run's must be under
50. A run that fails either is **void, not negative**: without this, a load
generator that failed to start produces two arms holding still, which selects
the branch that retires A16's last testable hypothesis, from a run in which the
intervention never happened.

Two further gates. If an arm-run fails, its whole pair is dropped, because a
half pair is not a paired observation. Fewer than eighteen surviving pairs for
any arm voids that arm's reading.

Two things this instrument does not see, stated because a null must not be read
as covering them: `/proc/stat`'s idle and iowait are summed, so a load that is
mostly disk wait records as a low `busy_pct`; and per-process accounting only
sees processes alive at two consecutive ticks, so a process born and reaped
inside one second contributes nothing. The suite's own churn is partly invisible
to the column that measures it, which is a reason to report the gate as a
floor rather than as a measurement of the treatment.

## The prediction, fixed before the data

### Per arm

The contrast for one arm is the mean of the twenty-four within-block log ratios
of its loaded rate to its unloaded rate, reported as a percentage change, with a
Student t interval on twenty-three degrees of freedom. The point estimate is the
centre of that interval and not a separately pooled figure.

Percentage change and not percentage points: A16's own figure is −3.40 % of a
decode rate, and a threshold has to be in the unit of the thing it explains.

If host CPU load is the mechanism behind A16, `spec-dflash-n2` **moves** and
`spec-draft-n8` **holds**, as those words are defined below.

### Across arms, which is the part the first draft got wrong

The three arms run at about 145, 116 and 31 tok/s, which is 6.9, 8.6 and 32.5 ms
per generated token. A **uniform** increase in host-side cost per token, with no
preference for any arm at all, therefore produces very different percentages:

| arm | ms per token | after a uniform +0.141 ms | percentage change |
|---|---:|---:|---:|
| `spec-dflash-n2` | 6.897 | 7.038 | **−2.00 %** |
| `baseline` | 8.598 | 8.739 | −1.61 % |
| `spec-draft-n8` | 32.468 | 32.609 | **−0.43 %** |

That is exactly the signature the first draft predicted and would have read as
"specific to one configuration". It is produced here by a mechanism that has no
specificity whatsoever, because `spec-draft-n8` is structurally 4.7 times less
able to show the same absolute cost.

So the cross-arm inference is pre-registered on **milliseconds per generated
token**, not on percentages. Let δ be an arm's change in ms per generated token
between its loaded and unloaded runs, with the same paired interval.

- **arm-agnostic**: the three δ intervals overlap. The load costs the host the
  same per token whatever the arm does, and nothing here is about speculation.
- **arm-specific**: δ for `spec-dflash-n2` lies above both others' intervals.
  This is the only pattern that supports A16's shape, in which one arm stepped
  while the arms interleaved with it did not.

The percentage thresholds decide whether each arm moved. The δ comparison
decides what moving means. A run in which `spec-dflash-n2` moves and the δ
intervals overlap is a run that found a host-wide cost, and the outcome section
must say so in those words.

### Why 2 % and not 3.4 %

The prediction is that the mechanism is available on this host, not that this
run reproduces the size of a step measured on another day under an unknown
ambient load. A recovered effect smaller than the incident is the expected shape
of a positive result; a threshold at the incident's own size could only be met
by a coincidence. The power table above is computed against both.

## What would falsify what

Two words, defined on the **interval** so that both are reachable and neither is
an absence-of-evidence claim. The first draft defined "moves" on an interval and
"holds" on a point estimate, which made "holds" unfalsifiable for the one arm
the run is about.

- an arm **moves** when its whole interval lies below −1 %
- an arm **holds** when its whole interval lies inside ±1 %
- otherwise it does neither, and that is an outcome with a name

**The arm moves, the control holds, and δ is arm-specific.** The mechanism
exists on this host. This does not establish that ambient load caused the steps
A16 recorded: those runs have no load column and cannot be revisited. What it
establishes is that the column has to exist from now on, and that every
decode-rate comparison **on this host** is exposed to it. The other hosts carry
different toolchains and the same question is open on each of them separately.

**The arm moves and δ is arm-agnostic.** Host load costs this machine a fixed
amount per generated token. That is worth recording and it is not A16's shape,
which spared the interleaved arms.

**Both move.** General starvation of the serving process. It does not explain
A16, whose step inside one invocation left the two interleaved arms at a CV of
0.12 % and 0.55 %.

**Both hold.** Host CPU load at this level is not the mechanism. Of the
hypotheses A16 names, the junction temperature stays untestable on this
platform, and the page-cache and allocator-state one is untested rather than
excluded; it is the successor run and it is listed in `RETEST_TODO.md` as such.
This branch does **not** license the sentence that A16 has no testable
hypothesis left, which is what the first draft pre-committed to and which A16's
own text contradicts.

**Anything else: inconclusive, and it licenses nothing.** An arm landing between
the thresholds, or an interval that spans them, is a run that did not decide.
The entry it produces says the run was made and did not decide. It does not get
read as weak support, it does not get rescued by pooling the pairs a different
way, and it does not get a second analysis chosen after seeing it. What it
licenses is a decision about whether a longer run or a higher load level is
worth the hours, which is a question about cost and not about this run's
evidence.

`baseline` does not select the branch, because the prediction was not written
about it; it enters the δ comparison, where its reading is what separates the
first branch from the second. If `baseline` itself lands between the thresholds,
that is reported and the δ comparison still decides, because δ does not depend
on either threshold.

## The analysis, fixed now, and the code that does not exist yet

The estimator, stated so that no choice is left to be made after the data
exists: for each arm, take the twenty-four pairs; for each pair compute the
natural log of the loaded pooled decode rate over the unloaded pooled decode
rate of the same arm in the same block; take the mean of those twenty-four
numbers, its Student t interval on twenty-three degrees of freedom, and
exponentiate both ends. Pooled decode rate is generated tokens over decode
milliseconds summed across the ten prompts of that arm-run, which is the
definition the rest of this repository uses. The same procedure on ms per
generated token gives δ.

**Nothing in this repository computes that today.** `analysis/paired_blocks.py`
pairs an arm against the baseline inside one block, which is the opposite shape:
this contrast is one arm against itself across a treatment. The
pre-registered prediction beside this file records what happens when a
pre-registered analysis has no code path, which is that recomputing it later
found eight wrong figures. So the
estimator is committed as code, with tests, **before** run X starts, and this
paragraph is the commitment.

## What this run cannot do

It cannot explain the historical steps, only show whether the mechanism is
available. It is one host, one binary, one set of model files, one arm known to
move, and one load level. Nothing from it transfers to another card, and the
fleet's hosts carry different toolchains, so only within-host contrasts compare.

It cannot separate CPU contention from chassis heating by itself. Eight busy
processors warm the case and therefore the card's inlet, and the quantity that
would mediate that is GDDR6X junction temperature, which is A16's *other*
unmeasured quantity and is unreadable here. No CPU package temperature is
recorded anywhere in this repository either. The order alternation inside each
block is what keeps a settled thermal state from aligning with the factor, and
it is a mitigation, not a separation. The contrast that would separate them is a
third level with the load pinned by `taskset` to processors the server does not
use, which heats the box identically and removes the contention; that is named
here so that a positive result is not read as more than it is.

The configuration is pinned to run T4's, which is the run this plan reasons
from: the same model files by SHA-256, the same `--fit-target`, the same ten
prompts, the same seed and the same concurrency, all recorded in the run's
manifest as every run here records them. One difference is deliberate and has to
be stated: T4 used the split-timer instrumented build, which adds host-side work
and is therefore not neutral in an experiment about host-side cost. Run X uses
the stock build, so the variability figures quoted from T4 are an estimate for
run X and not a measurement of it.

The run's data, manifest and telemetry are committed together with the outcome
section filled in, in one commit, so that the plan and the result cannot drift
apart.

## Where this document sits, and what it will cost

It is in `analysis/table_coverage.py`'s excluded list rather than its censused
one. The reason recorded there is the release, and not the genre: three
prospective plans of exactly this kind are censused, and one of them,
`PROSPECTIVE_ANALYSIS_PLAN_W2.md`, is censused today with no outcome section and
no data of its own. So "a plan is not censused" is not a rule here and this
document does not invent one.

What excludes this one is that `analysis/verify_claims.py` pins the number of
censused documents at eleven and the decimal prose census at its current pair of
values, and that checker is one of the six files `bench/check_release_binding.py`
compares between the `v4.2` tag and HEAD. Note where the freeze bites:
`table_coverage.py` is not itself bound, so growing its list is not editing a
frozen file. It is editing an unfrozen one in a way that makes a frozen one
fail, which cannot then be fixed without changing the frozen one, and the tag
would stop publishing the tree that verifies it. This document also quotes
decimals from A16, so censusing it would move the decimal census as well as the
count.

When run X produces data that changes, and the honest accounting is that the
census is not what costs anything. Run X's raw logs go into
`EVIDENCE_MANIFEST.sha256` and its entry into `RUN_REGISTRY.json`, and both of
those are bound, so committing the evidence re-cuts the release binding whatever
happens to this file. Censusing it then rides along at no additional cost.

The cost this document's own commit does incur, today, is a coverage probe
re-run: adding lines to a censused document moves the table line numbers the
existing attestations pin, so the thirty-two shards have to be produced again.

# Outcome

Not run yet. This section is filled in when the data is committed, and until then
its emptiness is the point.
