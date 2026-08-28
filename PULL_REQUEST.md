<!--
  The body of pull request #2, kept here so its numbers are checked like every
  other published table in this repository. `analysis/verify_claims.py` parses
  the seven tables below cell by cell and `tests/data_mutate.py` perturbs them;
  a figure that drifts out of agreement with the data fails CI.

  Publish with `python tools/publish_pr_body.py`, which strips this comment
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
and one no-speculation baseline** (nine arms) in **one balanced Latin square**:
nine blocks, every arm in every position exactly once, verified from the
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
row order shuffled from a per-session seed, 500 of 500 arm-runs. Run V3
verbatim except for `BENCH_ORDER`. Every arm visits every position exactly once
**and** is preceded by every other arm exactly once within a repeat, verified
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

**It is not the predecessor.** With every arm preceded by every other exactly
once, the contrast between running after a capped neighbour and after a
free-running one is **−1.20 %** [−2.61, +0.22] for `spec-dflash-n2`. That is
the largest of any arm by six times, it points the way A17 guessed, and it is
negative in four sessions of five, but **no arm's interval excludes zero**.
Reported as no detectable effect at this power, with the interval, not as
absence. An effect
big enough to explain a 2.4 pp gap would be far outside it.

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
> next experiment and **it has not been run**.

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
  the row is 30.4 % and the column consequently added to 100.1 (**B9**). All are
  parsed cell by cell now, and 73 data and document perturbations are permanent
  tests.

## What the third review found

Every item below was verified against the code or the data before it was
changed. Nothing in it was rejected.

- **A17's treatment order.** Answered by measurement: run V2's crossover and
  run V3's within-invocation square, above. Run V overstated one arm by about
  3.3 pp; the sign flip it was written about survives both designs.
- **A12's boundary.** Answered by measurement: run T4 splits the checkpoint
  timers and the wait is **0.002 s of 39.09 s**, above. The four-repeat timer
  matrix has no MTP arm.
- **Scope.** The controlled tier is runs A to V3, not A to T3. Its findings are
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
  trusted. `raw-evidence-2026-08-27` carries `raw_logs.tar.zst` (702 logs,
  4071 MB uncompressed) and `telemetry.tar.zst` (19 traces), plus a second
  tranche `raw_logs_20260827.tar.zst` (618 logs, 2.9 GB, sha256 `d56a7f88…`)
  for V2, V3 and T4, kept separate so the first archive's digest keeps meaning
  what it meant. The manifest is 1320 logs and 21 traces; all 620 new entries
  were verified against it before publishing.
  `python analysis/rederive_from_logs.py <bench-root>` checks every file
  against `EVIDENCE_MANIFEST.sha256` and re-runs the extractors:

  | derived file | records | identical | not regenerated |
  |---|---:|---:|---:|
  | `data/spec_accounting_20260826.json` | 12 | **12** | 0 |
  | `data/checkpoint_timers_20260826.json` | 12 | **12** | 0 |
  | `data/checkpoint_timers_20260827_split.json` | 18 | **18** | 0 |
  | `data/acceptance_counter_comparison.json` | 535 | **526** | 9 |

  **That is the script's output, not CI's.**
  `.github/workflows/evidence.yml` does the same thing on `workflow_dispatch`,
  on `release: published` and weekly, and it has **never run**: GitHub
  registers workflows, schedules and dispatch targets from the default branch,
  and the file exists only on this one. Dispatching it returns 404. It becomes
  live when this merges; until then the numbers are what anyone with the
  release gets from that one command.

  Zero records differ. The nine belong to three exploratory runs whose logs are
  in the archive and whose arm-run JSON is not committed, because they never
  completed their cell set and `check_data_integrity.py` refuses incomplete
  runs. They are nine of the 535 rows behind A13; the claim rests on the 526.

Both merge blockers that needed GPU time are now closed by measurement (runs V2
and V3 for P0-1, run T4 for P0-2), and the raw evidence for all three is
published. **9.5 hours on the bench card, 618 arm-runs, none failed.** What keeps
this a draft is below, under *Not closed*.

## Checking it

```
python analysis/rederive_from_logs.py bench   # raw logs -> the committed JSON
python analysis/verify_claims.py          # 1815 assertions, re-derived
python analysis/check_data_integrity.py   # structure of all 65 run directories
python -m unittest discover tests         # 195 regressions for defects shipped here
python tests/mutate.py                    # break each fix, require its test to fail
python tests/data_mutate.py               # perturb a measurement or a published
                                          #   figure, require the checker to fail
                                          #   58 code and 84 data perturbations,
                                          #   with a clean-mirror re-check after
                                          #   the last restore
python analysis/plot_v4_runs.py --check   # charts still match the data
```

CI runs all of it on every push, with actions pinned to commit SHAs, chart
dependencies hash-pinned, shellcheck at `--severity=style` and pyflakes. That
is `.github/workflows/audit.yml`, which is registered and green on this head.
The evidence workflow beside it is not: see above.

`verify_claims.py` parses its own AST and fails if any assertion compares two
literals. Six of them did, and were rewritten.

## Not closed

- **A16's cause.** Nothing recorded distinguishes a fast arm-run from a slow
  one: not temperature, not clock, not power, not throttle state, and not the
  work done, which is byte-identical. Run T4 narrows it from an invocation
  effect to an arm-run-level state that steps within one invocation; it does not
  explain it.
- **The randomised-order run.** V2 and V3 balance position and fix the
  predecessor, so neither can test whether `spec-dflash-n2` is sensitive to what
  ran before it. That is the experiment that would settle its +5.9-vs-+8.7 pp
  disagreement, and it has not been run.
- **The request-mean columns.** `predicted_per_second` is llama.cpp's own field
  and it divides `n − 1` tokens by the time for `n`, in 13 300 of 13 344
  committed request rows, exactly. Every **request-mean** column in this
  repository inherits that, understating by `(n − 1) / n`: 0.33 % at 300 tokens.
  No headline figure or published delta contains it, because those are pooled
  rates computed from the two raw fields, and on a run where every request hits
  the same cap the bias is identical on every arm. Recomputing several dozen
  published request-means to move them by a third of a percent is listed rather
  than done (**B8**), and the relationship is asserted so it cannot change
  silently.
- The 7 GB of llama-server logs are not committed.
  `v4_audit_2026_08_25/EVIDENCE_MANIFEST.sha256` holds the SHA-256 of all 1320
  of them and of the 21 telemetry traces; both compressed tranches are published
  as release assets, and `analysis/rederive_from_logs.py` re-runs the
  extractors against them.
- `draft-eagle3` needs three extract layers this model does not expose;
  `--spec-type draft-dspark` is DeepseekV4-only. Nine of eleven measured.


