<!--
  The body of pull request #2, kept here so its numbers are checked like every
  other published table in this repository. `analysis/verify_claims.py` parses
  the eight tables below cell by cell and `tests/data_mutate.py` perturbs them;
  a figure that drifts out of agreement with the data fails CI.

  Publish with `python tools/publish_pr_body.py --write`, which strips this comment
  line by line and then reads the body back from GitHub to prove it landed.
  Do not strip it with a regex: the previous one-liner matched a literal
  closing marker inside its own pattern and published half of itself.

  No em-dashes or en-dashes in here, unlike the rest of this repository's
  prose: a local pre-tool hook refuses to post GitHub text containing them.
  Ranges are written "1 to 8" and asides take commas, colons or parentheses.
  U+2212 for a negative number is fine and the tables use it.
-->

This branch audits what this repository had published, measures what it had never
run, and then corrects the things the audit itself got wrong. Three external
reviews drove the last part; every specific accusation in each was verified
against the code and the data before anything was changed, and every one that
could be checked was right. The third review's findings are listed under
**What the third review found**, below.

## The result

One RTX 3090, llama.cpp `3737e4137`, Qwen3.6-35B-A3B-UD-Q4_K_XL, greedy, ten
prompts, thinking on, one request at a time. **Eight speculative configurations
and one no-speculation baseline** (nine arms) in **one Latin square balanced
for position**: nine blocks, every arm in every position exactly once, verified
from the
arm-runs' own monotonic timestamps rather than from the design.

The interval below is over the nine blocks **of this one invocation**, in one
fixed rotation. It is not a configuration-level interval, and A16 below is why:
this arm measured twelve times in one day spans **+17 % to +27 %**. Read
`+17 % to +27 %` as the result and `+26.3 %` as run O2's point estimate.

| arm | pooled tok/s | change | 95 % CI (t, blocks within this invocation) | draft/gen | acceptance |
|---|---:|---:|---:|---:|---:|
| **`spec-dflash-n2`** | **146.2** | **+26.3 %** | [+25.5 %, +27.1 %] | 0.81 | 72.3 % |
| `spec-mtp-n2` | 141.9 | +22.7 % | [+22.1 %, +23.3 %] | 0.77 | 78.4 % |
| `spec-dflash-n4` | 137.9 | +19.2 % | [+18.5 %, +19.9 %] | 1.24 | 55.2 % |
| **no speculation** | **115.7** | n/a | n/a | 0.00 | n/a |
| `ngram-map-k4v-m8` | 115.4 | −0.3 % | [−0.6 %, +0.0 %] | **0.01** | 50.0 % |
| `ngram-mod-n24` | 103.1 | −10.9 % | [−11.4 %, −10.5 %] | 0.19 | 5.0 % |
| `ngram-cache` | 93.7 | −19.0 % | [−19.4 %, −18.6 %] | 0.17 | 5.2 % |
| `spec-draft-n8` | 30.9 | −73.3 % | [−73.5 %, −73.2 %] | 1.86 | 29.5 % |
| `spec-draft-n1` | 29.2 | **−74.8 %** | [−74.9 %, −74.7 %] | 0.50 | **69.7 %** |

A factor of five between the best and worst speculative method. The two that
win both draft from inside the target: DFlash, and the model's own
multi-token-prediction head. The one that loses three quarters of the
throughput is a separate 0.8 B draft model. That is a grouping, not an isolated
cause: these arms differ in architecture, state reuse, rollback, quantisation
and per-round drafting cost at the same time, and this matrix does not separate
them. The published headline, "every tested condition that recorded speculative
activity was slower", was true of what v1 measured. v1 tested three methods, an
external draft model and two n-gram methods, and they occupy the bottom **four**
rows here because the external drafter appears twice. It did not test DFlash or
MTP.

`spec-draft-n1` falsifies acceptance as a sufficient predictor: **69.7 %
acceptance and −74.8 %**. A high-acceptance configuration can still lose badly
when proposal generation, verification, checkpointing and replay are all
expensive. Which of those dominates, this row does not say; the decomposition
below puts 24.2 % of the excess on the drafter's own `generate()`.
`draft/gen` is proposals per generated token, and it is what makes the acceptance
column readable: `ngram-map-k4v-m8`'s 50 % is 108 of **216 proposals over 27 000
generated tokens**, so it is neutral because it almost never fires.

## What it costs, timed in the source

An external drafter makes this hybrid target checkpoint and restore its state on
every partially accepted round. Rebuilt with `ggml_time_us()` around the four
calls (patch archived, 15 insertions, timing and logging only):

| | seconds | share of the excess |
|---|---:|---:|
| **excess over no speculation** | **71.4** | **100 %** |
| speculative checkpoint save (785) | 17.34 | 24.3 % |
| speculative checkpoint restore (728) | 21.74 | 30.4 % |
| drafter `generate()` | 17.27 | 24.2 % |
| unattributed | 15.05 | 21.1 % |

**What that boundary is, measured.** The timers surround `ckpt.update_tgt`,
`ckpt.load_tgt` and `ckpt.load_dft`, and those call
`llama_state_seq_get_data_ext` / `set_data_ext`, which at `3737e4137` begin with
`ctx->synchronize()` (`llama-context.cpp:4083`). So 39.07 s is elapsed **inside
the checkpoint API calls**, synchronisation included, which the third review
raised as a reason to distrust the 54.7 %, because a large wait would make it an
attribution to the API boundary rather than to the copying.

**Run T4 splits it.** `bench/apply_split_timers.py` drains the queue explicitly
just before each call and times the drain, so the call's own `synchronize()`
finds nothing outstanding and what is left is the state work. Six repeats, same
three arms, same configuration, on a build differing only by the extra timers:

| | seconds | share of the 71.49 s excess |
|---|---:|---:|
| inside the checkpoint calls | **39.09** | **54.7 %** |
| of which, waiting on `synchronize()` | **0.002** | **0.003 %** |
| of which, state work | **39.09** | **54.7 %** |

**Two milliseconds of thirty-nine seconds.** The queue is already drained when
the checkpoint is taken; the sampler has just read the logits back. Every
component reproduces (17.336 / 16.346 / 5.412 against 17.34 / 16.33 / 5.41), so
do the counts (785 creates, 728 restores, in all six repeats) and the excess
(71.49 s against 71.4). The concern was legitimate and the measurement refutes
it: **the 54.7 % is post-drain checkpoint state-save/restore API work.** Not
raw copy cost: the residual still contains serialisation, container
allocation and resize, host/device transfer, state traversal and the API's
own bookkeeping, and this run separates none of them from each other. What
it rules out is that the time was spent waiting on target work queued
before the call. The patch is archived, 31 insertions in one
file; the tree was restored to stock afterwards, verified by the library hash
returning to `a0cbe4d0…`.

**What the four-repeat timer matrix contains** is `baseline`, `spec-draft-n8`
and `spec-dflash-n2`, twelve logs, four repeats each. `spec-draft-n8` emits 1513
`AUDIT_US` records in every repeat and the other two emit **zero** in every
repeat. It contains **no MTP arm**. The MTP result, no speculative checkpoint
events at draft lengths 1 to 8, comes from the ordinary verbose logs, a
separate evidence chain, and DFlash's 1 to 16 range does too. Re-applying the
archived patch to a stock tree rebuilt the instrumented library to the same
`ce94855f…` byte for byte, and reverting rebuilt to `a0cbe4d0…`.

## Two findings the re-measurement produced

**A16. `spec-dflash-n2` moves between unexplained performance levels, and no
recorded field distinguishes them.**
`spec-dflash-n2` was measured **twelve times in one day** on the same models,
same policy, same prompts: **+17.3 % to +26.7 %**, on byte-identical output and
identical draft counts, while the no-speculation reference beside it holds a CV
of **0.42 %**. Pooled by block that is 43 measurements spanning 10.8 pp, and
`draft_n` is 2441 with acceptance 72.3 % in every single one, so the speculative
work is identical to the token and only the time differs.

Eleven of the twelve runs sit wholly above or below a +23 % split. Run O3 crosses
it at block 4, and **in those blocks only this arm moves**: −4.45, −4.66, −3.33,
−2.93 % against its own first block, while eight other arms, including
`spec-dflash-n4` (the same DFlash drafter at twice the draft length), never leave
±1.24 %. It survives the server restart between arm-runs and can change inside a
single run. **Run T4 cornered it further**: six repeats in one invocation read
139.36, 139.72, 139.93, then 145.04, 146.01, 144.43, a **3.9 % step partway
through**, on byte-identical output, while the two arms interleaved with it hold
0.12 % and 0.55 %. Three arms rotating means the predecessor varies, and it
explains nothing: both predecessors produce both levels. 272 telemetry samples
explain nothing either. Across the step the card is *hotter* (63.4 to 64.9 °C),
the SM clock is flat, the power is flat, and `sw_power_cap` is asserted in 17
samples while the arm is slow and 36 while it is fast. Every recorded physical
quantity either does not move or moves the wrong way. So A16's "invocation
effect" is really an **arm-run-level state that steps on a timescale of
minutes**. The cause is not isolated.

The consequence is that the paired-block interval measures the wrong variance
component. O2's is `[+25.5 %, +27.1 %]` and O3's is `[+21.4 %, +25.6 %]`: they
overlap by 0.1 pp, on 810 of 810 byte-identical request-pairs.

**A17. Every thinking-off comparison that let the arms stop where they liked
compares different amounts of work.**
Speculation is not output-preserving on this build, so with thinking off the arms
stop in different places; all 5904 thinking-on requests hit the cap and none
stopped early, while 881 of 1440 thinking-off requests did.

Run V measured it in one session with every freerun block before every hard-cap
block, which confounds the mode with sixteen minutes of elapsed time. That was
the third review's P0-1, and it was correct. **Run V2 is the crossover it asked
for.** Eight sessions of two halves on 2026-08-27 in `AB BA BA AB BA AB AB BA`
order, so each
mode ran first four times and second four times with the two orders balanced in
mean time position. Five arms, five repeats a half, run V's configuration
otherwise verbatim. **400 of 400 arm-runs, none failed.** The session is the
resampling unit and the baseline is measured inside the same half, so a
whole-invocation shift that moves every arm equally cancels.

| arm | freerun | hard cap | shift, 95 % t over 8 sessions |
|---|---:|---:|---:|
| `spec-dflash-n4` | **−1.66 %** [−1.98, −1.35] | **+10.37 %** [+10.20, +10.53] | **+12.03 pp** [+11.67, +12.38] |
| `spec-mtp-n2` | +11.48 % | +21.02 % | +9.54 pp [+9.14, +9.93] |
| `spec-draft-n8` | −76.77 % | −70.46 % | +6.31 pp [+6.29, +6.33] |
| `spec-dflash-n2` | +10.71 % | +16.63 % | +5.92 pp [+4.86, +6.99] |

**What the shift column is.** An **absolute change in percentage points** of
the arm-versus-baseline number, which is what `analysis/length_matching.py`
reports for the same arms by a different method. The analyser used to define
the session effect as a difference of log ratios in its documentation while
averaging the percentage-point form; those are different estimands. They agree
near the baseline and diverge away from it: `spec-draft-n8` is +6.31 pp and
**+27.15 %** as a log contrast, because it runs at a quarter of baseline speed.
Both are printed and serialised now, and the tables publish the pp form.

**The hard cap raises every arm and every interval excludes zero.**
`spec-dflash-n4`, the arm A17 was written about and published as "`n_max 4` goes
negative", is negative free-running and positive under the cap, both intervals
clear of zero on opposite sides.

That sign **depends on the stopping policy**, which is weaker than
calling it a length artefact. `ignore_eos` does not only equalise the token
count: forcing generation past a natural stop changes which tokens are
produced, the positions they occupy, the experts routed, and the acceptance
that follows. V2 shows the sign differs between a forced cap and natural
stopping, not that length alone causes it.

**And run V overstated one arm.** Its `spec-dflash-n2` shift, +9.26 pp, lies
above **all eight** sessions (max +8.07). The other three of its four numbers
land inside the eight-session intervals. So the review's objection was right and
the size of what it was right about is **about 3.3 pp on one arm**. Order was
not the cause: splitting the sessions by which mode ran first moves that arm by
1.40 pp and the others by 0.36 pp or less. The invocation was, which is what
averaging over eight of them removes.

**Run V3 puts both modes in one square, and for that same arm it disagrees.**
`BENCH_HARDCAP_SUFFIX` makes `<arm>` and `<arm>-cap` ten arms of one balanced
10×10 rotation, so the modes are minutes apart rather than sixteen. Two
sessions, 200 of 200 arm-runs.

| arm | V3, within invocation | V2, between invocations |
|---|---:|---:|
| `spec-dflash-n4` | **+12.17 pp** | +12.03 [+11.67, +12.38] |
| `spec-mtp-n2` | +9.53 pp | +9.54 [+9.14, +9.93] |
| `spec-draft-n8` | +6.30 pp | +6.31 [+6.29, +6.33] |
| `spec-dflash-n2` | **+8.65 pp** | **+5.92** [+4.86, +6.99] |

Three of four agree to a tenth of a point across two designs that share nothing
but the harness, and V3's own two sessions are 0.06 pp apart. The fourth does
not, and both schedules are cyclic rotations that balance treatment position
and leave the predecessor fixed, so neither could say why.

**Run W is the design that can.** Five sessions of a 10 x 10 Williams square,
row order shuffled from a per-session seed, 500 of 500 arm-runs. It uses V3's
treatment definitions, prompts, models and server build, with a Williams
schedule and a later harness revision, **not V3 verbatim**, which this section
used to claim. The two manifests record different runner hashes, and the diff
between exactly those blobs is archived at
[`v4_audit_2026_08_25/harness/V3_to_W_runner.diff`](v4_audit_2026_08_25/harness/V3_to_W_runner.diff)
with every hunk classified: 189 lines, all of it the schedule builder,
provenance assertions and provenance records, none of it reaching the request
body, server argv, timing collection, teardown or aggregation. Every arm
visits every position exactly once **and** is preceded by every other arm
exactly once within a repeat, verified
from the arm-runs' own `t_start` order rather than from the manifest, in all
five sessions. The analysis plan was committed while the run was at 360 of 500
and the checker asserts that commit is an ancestor of the one carrying the
data.

| arm | V2, between | V3, within | **W, within and carryover-balanced** |
|---|---:|---:|---:|
| `spec-dflash-n4` | +12.03 [+11.67, +12.38] | +12.17 | **+12.10** [+11.87, +12.34] |
| `spec-mtp-n2` | +9.54 [+9.14, +9.93] | +9.53 | **+9.53** [+9.34, +9.73] |
| `spec-dflash-n2` | **+5.92** [+4.86, +6.99] | **+8.65** | **+8.29** [+7.97, +8.60] |
| `spec-draft-n8` | +6.31 [+6.29, +6.33] | +6.30 | **+6.35** [+6.32, +6.38] |

`spec-dflash-n4`'s sign flip holds at +12.03, +12.17 and +12.10 across three
schedules. For `spec-dflash-n2`, W's interval **overlaps V3's and does not
overlap V2's at all**: the two within-invocation designs agree and the
between-invocation crossover does not.

**No capped-predecessor association was detected, at this power.** With every
arm preceded by every other exactly once within a row, the contrast between
running after a capped neighbour and after a free-running one is **−1.05 %**
[−2.97, +0.86] for `spec-dflash-n2`, the largest of any arm, pointing the way
A17 guessed, negative in four sessions of five, and **no arm's interval excludes
zero**. Reported as no detectable effect at this power, with the interval, not
as absence.

That figure is the **identity-matched** estimator. The grouped one this
paragraph used to quote, −1.20 % [−2.61, +0.22], is not a mode contrast: for an
uncapped arm `X` the capped group contains `X-cap`, its own twin, while the free
group cannot contain `X`, because nothing precedes itself, so it carries the
predecessor's mode, one unmatched predecessor identity, and a five-against-four
weighting together. Pairing each capped predecessor with its own free
counterpart and dropping the twin moves the estimate to −1.05 % and **widens**
the interval, which is where the removed precision was coming from.

This does not rule the predecessor out, and an earlier version of this paragraph
said it did, under the heading "It is not the predecessor", and with the claim
that an effect big enough to explain a 2.4 pp gap "would be far outside" the
interval. **−2.4 is inside [−2.61, +0.22]**, and inside the wider matched
interval as well. The two numbers are also different estimands: −1.05 % is a
relative difference in one arm's raw decode rate by predecessor mode, while
2.4 pp is a difference between arm-and-baseline mode effects. Neither bounds the
other.

**Run W2 rules it out, and its plan was committed before the driver was
invoked.** Twelve sessions of the same square on 2026-08-30 and 2026-08-31,
1200 of 1200 arm-runs, none failed, a separate invocation under its own
`BENCH_RUN_LABEL` so no analyser pools it with W. Session count, estimand and
the meaning of each outcome were fixed in
`v4_audit_2026_08_25/PROSPECTIVE_ANALYSIS_PLAN_W2.md`, which is an ancestor of
the commit carrying the data.

| quantity | W, 5 sessions | W2, 12 sessions |
|---|---:|---:|
| matched contrast, `spec-dflash-n2` | −1.05 % [−2.97, +0.86] | **−0.14 % [−0.68, +0.41]** |
| grouped contrast, the same arm | −1.20 % [−2.61, +0.22] | −0.21 % [−0.78, +0.35] |
| including row boundaries | not computed | −0.19 % [−0.75, +0.38] |
| `shift_pp` after capped minus after free | +1.52 pp [−0.98, +4.02] | **+0.49 pp [−0.80, +1.77]** |
| session SD of the matched contrast | 1.543 implied | 0.858 |

The fourth row is the one that answers the paragraph above. Because a rate
contrast does not bound a difference of two ratios, the predecessor is tested a
second time on `shift_pp` itself, the quantity A17's four-design table
publishes, computed on the arm-runs that follow a capped neighbour and again on
those that follow a free one. W's interval on that quantity **contains** 2.4;
W2's does not. That test is post hoc and grouped, the plan pre-registers the
matched rate contrast rather than it, and both facts are stated where the
number is.

Two arms of ten have matched intervals excluding zero, `spec-mtp-n2-cap` at
+0.11 % [+0.00, +0.23] and `spec-draft-n8` at +0.06 % [+0.01, +0.10]. Both are
under an eighth of a per cent, twenty intervals at 95 % are expected to produce
about one exclusion by chance, and the plan named this outcome in advance so
that it would be reported rather than absorbed.

**What is left is A16.** The difference is between measuring the two modes
inside one invocation and across two. W reproduces the instability that section
is about: mean within-session CV of **1.69 %** for `spec-dflash-n2` against
**0.31 %** for no speculation, on work identical to the token: every arm
produced one distinct output set and one distinct drafted/accepted pair across
all 5000 request rows, matching V2's and V3's counts exactly.

The thinking-**on** results, which is everything in the headline table, are
unaffected: the same test moves them by 0.00 pp, because all 5904 thinking-on
requests ran to the cap.

> **What neither design can test.** Both schedules are cyclic rotations, which
> balance *position* and fix the *predecessor*. The one predecessor contrast
> available is between the two designs and is confounded with the design. It
> points the same way each time, that arm being slower after a capped neighbour,
> but a fixed rotation cannot separate it, and this repository has said the
> wrong thing about `spec-dflash-n2` twice already. Randomising the order is the
> next experiment.
>
> *(That was written before run W. W is that experiment: five sessions of a
> 10 x 10 Williams square, 500 of 500 arm-runs, complete. The quoted block is
> left as it was written; what it asked for exists, and the section above
> reports it.)*

## What the audit got wrong about itself

Beyond the reviews' findings (a double-counted state volume, an unrecoverable
wall-clock share, a `latin` label on a schedule that was not balanced), the
adversarial pass over this branch's own commits found:

- **`tests/mutate.py` edited the real source files** and restored them in a
  `finally` that does not run when the process is killed. It committed one of its
  own mutations into the tree; `bench/retest_runner.py` was published with
  `body.pop("ignore_eos", None)`. It runs in a mirror now.
- **A13 was built from `*__rep0.log`**, one arm-run per arm however many repeats
  it had. Over every repeat of every run the base grows from 73 to **517** and the
  claim survives, with the margin between the two groups narrowing from 0.5 pp to
  **0.20 pp**.
- **The threshold scorecard**, which reads the same file and was keyed by
  `(run, arm)` in a dict, goes from **35 / 37 to 78 / 86**. Three of the eight
  misses are `spec-draft-n1`; the other five sit within 2 pp of the boundary.
- **The teardown guard** waited for the GPU to fall below an absolute 2048 MiB,
  so anything else using the card failed every arm-run. It compares against the
  pre-run reading now.
- **The published checkpoint total was 39.08 by double rounding**; it is 39.07.
- **A withdrawn figure was still in the README**: `101.3 MiB`, which is
  82.079 + 19.266, the sum A12 retracted. Every retracted number is now guarded.
- **Published tables were computed and asserted while the tables themselves were
  not parsed**, so planting wrong numbers in them passed every check. It has
  turned up in five places: three found by the review's own pass, then run M's
  aggregates found by the mutation suite, then the merged checkpoint cost table
  in `README.md` and in this body, which carried a restore share of 30.5 % where
  the row is 30.4 % and the column consequently added to 100.1 (**B9**). Those
  five are parsed cell by cell now, but "all of them" was a statement about the
  ones that had been found, not about the class, and each had been found by
  accident. The class is counted instead. Of 139 published tables 127 carry
  measurements and all 127 are parsed cell by cell. When 80 were still unparsed,
  perturbing one cell of each left 67 that accept a wrong number with nothing
  noticing; all 80 of those are parsed now, and no published table that
  carries a measurement is left unparsed.

  One cell per table was too weak a test to say a table is guarded. Perturbing
  every number instead, 1413 across the parsed tables, found 80 that no
  assertion notices: whole interval columns read as `.split("[")[0]` and
  stopped, a configuration column nobody read, the `100 %` on a total row. Each
  is compared against a value derived from the data, and the measurement says
  so: on 2026-08-31 the probe perturbed all 2 415 numbers across all 127
  parsed tables and caught every one, in eight shards whose control passed
  before the work and again after it. Their union is checked rather than
  assumed: `analysis/table_coverage.py --aggregate` over the eight real
  attestations reports **8 shards, one head, one checker, 2 373 locations
  covered exactly once, 0 survived**, each attestation carrying the head, the
  checker hash, the population digest, both controls and every location as a
  character span. Before 2026-08-30 nothing showed the eight had all run, or
  were disjoint, or were the same tree, and `--shard=8/8` selected nothing and
  exited 0. It took four runs, because the population
  grows as the coverage does and a clean run is only clean for the tree it ran
  on: the run before this one covered 2 252 numbers across 119 tables.
  `analysis/table_coverage.py --probe --covered --every-cell` is the
  measurement and **A19** the accounting; 78 code and 84 data and document perturbations
  remain permanent tests.

  Parsing the rest of them, rather than reading them, found seventeen more
  published statements wrong. The run registry had run C at three repeats
  where its manifest says five and run D at thirteen arms where it has five;
  it said "30 requests each" for runs A and B, which is B, while run A's two
  speculative arms abort part way at twelve. Run E's row named three of the
  four draft lengths it swept. A blank line had orphaned the tier registry's
  **v4 audit** row from its header, so GitHub rendered the controlled tier as
  literal pipes. A14 called two runs' recorded argv byte-identical when one of
  the thirty tokens, the listening port, differs. A17's split table carried run
  V's largest **mode** contrast in a column asking for its largest
  length-matching shift, said the external drafter appears three times across
  the thinking-off runs where it appears five, and quoted three coefficients
  under a four-row table so that a reader lining them up read the wrong arm's.
  A15's O2-against-T baseline row read +0.54 % where the pooled rates give
  +0.53 %. `BENCHMARK_ENV.md` put run N in the `-fit on` group when N ran
  pinned, named no run W, and counted seventeen telemetry traces where the tree
  holds sixteen. The v4 file map said forty-one directories a line above a row
  saying sixty-five, and the README's data map counted 62 v2 logs where the
  three directories hold 61. Four more came out of the last batch: the v4
  README's prompt-set table put a figure in a column headed "run M3" for an arm
  run M3 does not carry, and the number it held was run L's thinking-**on**
  figure; the thinking-off table beside it took one column from run M1's
  aggregate and the other from run M3's pooled rate without saying so; B8's row
  census still counted the tree as it stood before run W, 13 344 rows where it
  now holds 18 344; run W's carryover table said the arms it does not name move
  under 0.15 % where the largest moves 0.18 %; and the changelog credited run M
  with a range that is run O's two metrics for the same arm.

  Four more came out of the two tables nothing had ever read at all. The
  status board still credited the BOS defect with +0.3 %, which is A2's
  superseded figure, and A2's own entry still carried it too: the +0.3 % is
  `33.7 / 33.6`, the ratio of two rounded request-means from A2's own table
  on the master binary alone, and the `-1.2 % to +3.7 %` beside it is that
  binary's six v1-tagged prompts, which drops `zh_hant` at -2.2 %. Pooled
  over both binaries the difference is +0.2 %, and across the sixteen
  (binary, prompt) cells the span is -2.2 % to +3.7 %. Four places reported
  the thinking control as "50/50 in D, 0/50 in C" with nothing to say the 50
  was per arm; over the whole run it is 250 of 250 against 0 of 650. And the
  host table called the target model 22 GiB, in a cell arguing 29 GiB of free
  disk was too little for it, where this repository's own listing gives 21G.

  One of the new checks read a file no clone has. The fitter placement A14
  cites lives in `v4_audit_2026_08_25/data/matrix_M.log`, and `.gitignore`
  line 4 is `*.log`, so the eight v4 harness logs were on the bench host and in
  nobody's checkout. It surfaced the only way it could: the coverage probe runs
  the checker inside a clean checkout of HEAD, and there it died 373 assertions
  early, so the probe's own baseline was broken and it was measuring nothing.
  The logs are committed now and a test refuses any path the checker opens that
  a fresh clone would not have. A clean checkout runs all 3758 assertions and
  exits 0, which it did not before.

  Figures in those tables that are not re-derivable here say so rather than
  being checked against themselves: A14's two batch sizes, read from a server
  log this repository does not commit; C4b's ~83 °C, which is the card's
  datasheet; and the host table, which is a read-only probe of two machines
  on one day. Of its seventeen figures, four turn out to be in the archive
  after all, because every run manifest carries the same `82 MiB, 0 %` from
  its own `nvidia-smi`; three more are the model sizes `BENCHMARK_ENV.md`
  lists; four are upstream identifiers it also records; and the six that are
  left carry a dagger.

## What the third review found

Every item below was verified against the code or the data before it was
changed. Nothing in it was rejected.

- **A17's treatment order.** Answered by measurement: run V2's crossover and
  run V3's within-invocation square, above. Run V overstated one arm by about
  3.3 pp; the sign flip it was written about survives both designs.
- **A12's boundary.** Answered by measurement: run T4 splits the checkpoint
  timers and the wait is **0.002 s of 39.09 s**, above. The four-repeat timer
  matrix has no MTP arm.
- **Scope.** The controlled tier is runs A to W, not A to T3. Its findings are
  for llama.cpp `3737e4137` under the recorded configuration, not "current
  llama.cpp": #25004, #27705, #27572 and #24055 are all open and all touch
  recurrent rollback, output row ordering, MTP correctness or hybrid checkpoint
  invalidation. If any lands, the O3 headline subset, run V, the run T
  checkpoint subset and a long-prompt concurrent MTP smoke test all need
  repeating.
- **The headline interval** is within-invocation and is labelled so; the
  repeated-invocation range is the primary figure. The twelve runs are not
  twelve independent measurements either, because they mix builds, matrix sizes
  and neighbouring treatments, so `+17 % to +27 %` is a bound on what was
  observed, not a confidence interval, and the 43 nested blocks are not 43
  replicates.
- **`CITATION.cff`** claimed "recorded batch width" for what is concurrent
  client requests outstanding, and "preserves raw logs" for preserving their
  hashes. Both corrected.
- **Seven fail-open paths in the harness and analysis**, every one of them a
  silent change of treatment or of provenance:
  `BENCH_IGNORE_EOS=oen` selected off and only `BENCH_THINK` parsed strictly;
  `BENCH_EXPECT_LIB_SHA256` accepted a prefix and compared one library of four
  that decide what a decode does; the teardown guard allowed 2 GiB of residual
  VRAM and counted an unreadable `nvidia-smi` as settled; `paired_blocks.py`
  used a stricter definition of "balanced" than the runner and wrote the same
  JSON either way; its t table stopped at df=10 and fell back to the *normal*
  1.96; `tests/mutate.py` still carried a sidecar recovery that wrote an
  unresolved path into the checkout; `stage_mtp_source.py` could stage two
  checkpoint generations at once. Thirteen new mutations and eleven new tests.

- **The raw evidence is published**, so the extraction can be re-run instead of
  **The published tag does not publish this tree.** `raw-evidence-2026-08-27`
  points at `de6f33bf`, this branch's base: the whole `v4_audit_2026_08_25`
  directory, the verifier and run W's data all postdate it, while its assets do
  not: a third tranche was added to a mutable release whose source revision
  stayed behind. The tag is deliberately not retargeted, because silently moving
  a published tag destroys the one property a tag has; its notes now say where
  it points and why. A final versioned release is cut at the exact
  dataset-and-verifier commit instead, by
  [`v4_audit_2026_08_25/RELEASE_PROCEDURE.md`](v4_audit_2026_08_25/RELEASE_PROCEDURE.md),
  and `bench/check_release_binding.py` fails any `raw-evidence-*` tag whose
  commit does not carry the manifest, both registries and the three verifier
  scripts that `HEAD` does.

  trusted. `raw-evidence-2026-08-27` carries `raw_logs.tar.zst` (702 logs,
  4071 MB uncompressed) and `telemetry.tar.zst` (19 traces), plus a second
  tranche `raw_logs_20260827.tar.zst` (618 logs, 2.9 GB, sha256 `d56a7f88…`)
  for V2, V3 and T4, kept separate so the first archive's digest keeps meaning
  what it meant, and a third, `raw_logs_20260828.tar.zst`, for run W. The
  manifest is **3020 server logs and 23 telemetry traces**; the first 1820 of
  those logs are packaged in **three tranches** published as six release
  assets, and run W2's 1200 are hashed in the manifest but not yet packaged,
  which the registry records as a pending fourth tranche. Every new entry was
  verified against the manifest before publishing. These counts come from
  `v4_audit_2026_08_25/RUN_REGISTRY.json`, which the checker compares against
  the data directories and the manifest.
  `python analysis/rederive_from_logs.py <bench-root>` checks every file
  against `EVIDENCE_MANIFEST.sha256` and re-runs the extractors. **It
  regenerates four log-derived audit files, not the primary per-request
  benchmark JSON**: the server logs carry no per-request timing rows, so those
  files are integrity-checked against the manifest and nothing more.
  `v4_audit_2026_08_25/EVIDENCE_REGISTRY.json` says which is which, and holds
  each artifact's expected run set independently of the artifact, so a run
  omitted from an output fails the comparison instead of falling outside it:

  | derived file | records | identical | not regenerated |
  |---|---:|---:|---:|
  | `data/spec_accounting_20260826.json` | 12 | **12** | 0 |
  | `data/checkpoint_timers_20260826.json` | 12 | **12** | 0 |
  | `data/checkpoint_timers_20260827_split.json` | 18 | **18** | 0 |
  | `data/acceptance_counter_comparison.json` | 535 | **526** | 9 |

  **That was the script's output, and CI has now reproduced it.**
  `.github/workflows/evidence.yml` fetched the archive, checked it against the
  manifest, unpacked it, re-derived the committed JSON from the raw logs and
  ran the claim checker over the result, on 2026-08-29, and it passed. That was
  its first run: `workflow_dispatch`, `release: published` and the weekly cron
  read the workflow from the default branch and the file exists only on this
  one, so dispatching it returns 404 and the schedule never fires. What fired
  was the `push` filter, which reads the workflow from the ref being pushed.
  Those three become live when this merges.

  Zero records differ. The nine belong to three exploratory runs whose logs are
  in the archive and whose arm-run JSON is not committed, because they never
  completed their cell set and `check_data_integrity.py` refuses incomplete
  runs. They are nine of the 535 rows behind A13; the claim rests on the 526.

Both merge blockers that needed GPU time are now closed by measurement (runs V2
and V3 for P0-1, run T4 for P0-2), and the raw evidence for all three is
published. **9.5 hours on the bench card, 618 arm-runs, none failed**, and run
W added 500 more the next day. What keeps this a draft is below, under
*Not closed*.

## What the fifth review found, and what answering it found

Every P0 and P1 in the fifth review is closed, and none of them needed the
card. The concurrent path passes the arm's hard cap and `chat()` has no default
to fall back to; run W's mode estimate is published with the row-boundary
arm-runs excluded and the design is described as carryover-balanced *within
rows*; the predecessor contrast is matched pair by pair with the arm's own twin
dropped; the V3-to-W runner diff is archived and classified hunk by hunk; the
plan is a prospective analysis plan finalised at 360 of 500 and says so; one
registry says what was run and another says what the re-derivation covers; the
published tag is reported as not binding this tree rather than silently
retargeted; and the shard outputs carry a head, a checker hash, a population
hash and a control verdict at both ends, checked as a disjoint union.

Answering it turned up five more, all in the answering rather than in the
review:

- **Nine fixes shipped with a test and without a mutation.** This repository's
  rule is that a mutation which survives means its guard is decorative, and it
  had not been applied to the batch that answered the review. Two of the nine
  turned out to matter. Reverting the census threshold to `< 3` and running the
  whole claim checker produced **no new failure at all**, because no published
  table has one or two value cells: the fix was right and completely unguarded.
  Synthetic fixtures carry it now.
- **The mutation anchors had no uniqueness rule of their own**, and one had
  already gone ambiguous.
- **The host sampler picked its roots by searching every command line**, so a
  `llama-server` belonging to somebody else counted as ours and the
  interference it exists to see went invisible.
- **The census counted a table with no numbers in it** among those carrying
  measurements, because it asked whether a parser reads a table before asking
  whether the table has a value. 124, not 125.
- **An extractor's timeout had no test**, so removing it again would have cost
  nothing.

## Checking it

```
python analysis/rederive_from_logs.py bench   # raw logs -> four audit files
python analysis/verify_claims.py          # 3758 assertions, re-derived
python analysis/check_data_integrity.py   # structure of all 77 run directories
python -m unittest discover tests         # 279 regressions for defects shipped here
python tests/mutate.py                    # break each fix, require its test to fail
python tests/data_mutate.py               # perturb a measurement or a published
                                          #   figure, require the checker to fail
                                          #   78 code and 84 data perturbations,
                                          #   with a clean-mirror re-check after
                                          #   the last restore
python analysis/plot_v4_runs.py --check   # charts still match the data
```

CI runs all of it on every push, with actions pinned to commit SHAs, chart
dependencies hash-pinned, shellcheck at `--severity=style` and pyflakes. That
is `.github/workflows/audit.yml`, which is registered and green on this head.
`.github/workflows/evidence.yml` beside it has also run and passed on this head;
what it does and does not prove is described where it is introduced.

`verify_claims.py` parses its own AST and fails if any assertion compares two
literals. Six of them did, and were rewritten.

## Not closed

- **A16's cause.** Nothing recorded distinguishes a fast arm-run from a slow
  one: not temperature, not clock, not power, not throttle state, and not the
  work done, which is byte-identical. Run T4 narrows it from an invocation
  effect to an arm-run-level state that steps within one invocation; it does not
  explain it.
- **A randomised-order run at a power that could resolve the predecessor.**
  V2 and V3 balance position and fix the predecessor, so neither can test
  whether `spec-dflash-n2` is sensitive to what ran before it. Run W is that
  experiment and it is complete (500 of 500 arm-runs) and it returns no
  detectable predecessor-mode association, with the widest matched interval
  spanning [−2.97, +0.86] %. That is a bound, not an answer: an effect of the
  size that would matter sits inside it. What is still missing is the power to
  exclude one, which is more sessions rather than a different design.
- **The request-mean columns.** `predicted_per_second` is llama.cpp's own field
  and it divides `n − 1` tokens by the time for `n`, in 18 300 of 18 344
  committed request rows, exactly. Every **request-mean** column in this
  repository inherits that, understating by `(n − 1) / n`: 0.33 % at 300 tokens.
  No headline figure or published delta contains it, because those are pooled
  rates computed from the two raw fields, and on a run where every request hits
  the same cap the bias is identical on every arm. Recomputing several dozen
  published request-means to move them by a third of a percent is listed rather
  than done (**B8**), and the relationship is asserted so it cannot change
  silently.

  Measured rather than left as a caveat: the gap is 0.33 % at a fixed
  300-token cap, which is why the pooled headline does not move, and
  **0.90 % to 1.75 %** over all twenty thinking-off runs with a freerun arm,
  ninety-eight arm figures, where the lengths
  vary. Within one run it is nearly the same for every arm, and the widest
  spread across arms in a run is **0.26 pp**, so arm-vs-baseline ratios
  move far less than the absolute rates do. The runner records
  `request_tok_s` = `1000 × predicted_n / predicted_ms` beside the
  upstream field from now on, so a future analysis does not have to
  choose between an upstream quirk and a silent re-baselining, and the
  three bounds above are assertions rather than prose.
- The ~3 GB of llama-server logs are not committed. That is the size
  `bench/collect_evidence.sh` states in three places and the audit README
  beside it; this line said 7 GB.
  `v4_audit_2026_08_25/EVIDENCE_MANIFEST.sha256` holds the SHA-256 of all 3020
  of them and of the 23 telemetry traces; the three compressed tranches
  covering the first 1820 are published as release assets, run W2's 1200 are
  hashed and await a fourth, and `analysis/rederive_from_logs.py` re-runs the
  extractors against them.
- `draft-eagle3` needs three extract layers this model does not expose;
  `--spec-type draft-dspark` is DeepseekV4-only. Nine of eleven measured.


