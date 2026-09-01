# Changelog

All notable changes to this bench are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning is not strictly semver; each numbered release is a public
publication point with its own data set.

<!-- A contents block, because these documents are linked into by section name from each other and a reader arriving cold had no way to orient but to scroll. Generated from the headings; `analysis/check_links.py` validates every anchor here, so a heading renamed without this list fails the static job rather than rotting quietly. -->

## Contents

- [[Unreleased] — the review pass, and an adversarial pass over it](#unreleased--the-review-pass-and-an-adversarial-pass-over-it)
  - [Run W, and the question it removes](#run-w-and-the-question-it-removes)
  - [The three runs the third review asked for](#the-three-runs-the-third-review-asked-for)
  - [The third review](#the-third-review)
  - [The harness fail-open paths the same review found](#the-harness-fail-open-paths-the-same-review-found)
  - [What the adversarial pass over this round found](#what-the-adversarial-pass-over-this-round-found)
  - [The raw evidence is published](#the-raw-evidence-is-published)
  - [Measured](#measured)
  - [What the adversarial pass found in this branch's own work](#what-the-adversarial-pass-found-in-this-branchs-own-work)
  - [Added](#added)
  - [Fixed](#fixed)
- [[v4.1] — 2026-08-26 · the controlled tier, and a reversal](#v41--2026-08-26--the-controlled-tier-and-a-reversal)
  - [Added](#added)
  - [Changed](#changed)
  - [Retracted](#retracted)
  - [Fixed](#fixed)
- [[v4.0] — 2026-08-25 · audit and retraction](#v40--2026-08-25--audit-and-retraction)
  - [Retracted](#retracted)
  - [Corrected](#corrected)
  - [Found during the audit](#found-during-the-audit)
  - [Added](#added)
  - [What survives](#what-survives)
- [Sibling-repo notice — vllm-2x3090 v5.0 (2026-05-17)](#sibling-repo-notice--vllm-2x3090-v50-2026-05-17)
- [[v3.0] — 2026-05-07](#v30--2026-05-07)
  - [Added](#added)
  - [Cross-method ranking (single 3090, Qwen3.6-35B-A3B Q4_K_XL target)](#cross-method-ranking-single-3090-qwen36-35b-a3b-q4_k_xl-target)
  - [Mechanism note](#mechanism-note)
- [[v2.3] — 2026-04-26 (afternoon)](#v23--2026-04-26-afternoon)
  - [Added](#added)
  - [Changed](#changed)
- [[v2.2] — 2026-04-26 · documented, never tagged](#v22--2026-04-26--documented-never-tagged)
  - [Changed](#changed)
  - [Added](#added)
- [[v2.1] — 2026-04-25](#v21--2026-04-25)
  - [Added](#added)
  - [Changed](#changed)
  - [Older Unreleased entries (carried over from earlier in 2026-04-22 → 2026-04-25)](#older-unreleased-entries-carried-over-from-earlier-in-2026-04-22--2026-04-25)
- [[v2.0] — 2026-04-22](#v20--2026-04-22)
  - [Added](#added)
  - [Changed](#changed)
  - [Key findings](#key-findings)
  - [Conclusion of v1 stands](#conclusion-of-v1-stands)
- [[v1.0] — 2026-04-21](#v10--2026-04-21)
  - [Added](#added)
  - [Key finding](#key-finding)

## [Unreleased] — the review pass, and an adversarial pass over it

**A published number is only checked if changing it breaks something, and that
is now measured rather than assumed.** `analysis/table_coverage.py` counts the
tables in the eleven published documents; `--probe` writes a wrong number into
one cell of each in turn and asks whether an assertion that was passing now
fails. Of 142 tables, 128 carry measurements and all 128 are parsed cell by
cell. When 80 were still unparsed, perturbing one cell of each left **67 that
accept a wrong number and nothing notices**; all 80 of those are parsed now.
Full accounting in [`ERRATA.md`](ERRATA.md) A19.

**One cell a table is too weak a test.** Perturbing every number of every
parsed table instead found 80 more that nothing read, nearly all a second or
third number inside a cell whose parser read the front and stopped: whole
interval columns reached through `.split("[")[0]`, a configuration column
nobody read, the `100 %` on a total row. Guarding those grew the parsed set and
the probe over the grown set found 33 more; the run after that perturbed all
2 252 numbers in the 119 tables parsed at the time and none survived, and
parsing the last six grew the population again, as run W2's tables did after
that. The complete pass on 2026-08-30 perturbed all 2 373 numbers across the
124 tables of commit 0334c60dac82 and caught every one, in eight shards whose
control passed before the work and again after it. The pass on 2026-09-01 perturbs all 2 446 numbers across the 128 tables of
commit `0ee748017143` and catches every one, in 32 shards whose control passed
both ends and whose attestations are committed. That is
what the W three-design table had done, in both documents that carry it: only
the W column was read, so V2's `+12.03` and V3's `+12.17`, two thirds of a
three-way comparison, could have been anything.

The measurement was wrong four times before it measured anything, and all
four are recorded in the file. The first used the claim checker's exit status, and
every table came back caught, because one unrelated assertion was failing at
the time and the checker exited non-zero whatever was done to the documents; a
probe whose control and treatment agree measures nothing, so it compares
failure *sets* now. The second matched a table header against the literals the
checker parses without asking which document the reader reads, so a table
duplicated into a second document counted as parsed there too, which is how the
changelog's own copy of the re-derivation table passed as covered while
accepting a wrong number. The third is below: the checker read a file
`.gitignore` excluded, so inside the probe's clean checkout it died 373
assertions early and every shard compared against a broken baseline. The
fourth is the same defect from the other side, and it is below: sixteen shards
against one `.git` made the git-gated assertions flake, so every baseline came
back with three failures a single shard does not have.

**And it had the defect it exists to find, twice more.** Its number extractor
fell back to integers only when a cell held no decimal, so `p_min 0 / 0.50 /
0.75 / 0.90 at n_max 8` was probed on its four decimals and never on the 8;
and its not-a-value filter was applied to whole cells, so one commit hash hid
every measurement beside it. The census and the probe now agree on what a
value is, and the census resolves the generic readers through the variable
that names their document.

**Four published statements were wrong.** Run C's caption said every other arm
sat between 0.03 and 0.48 tok/s of run-to-run SD once the cold start was
removed; five arms are above it and the true top is 1.01, so `ngram-cache` leads
by a factor of 1.8 and not by the wide margin claimed (ERRATA A18). B1's
long-output row labelled its request-mean delta `vs 300-tok base`; −13.0 % is
against `baseline-1000tok`, and against the 300-token baseline it is −14.6 %.
Run N's paragraph mixed two spans, quoting a per-repeat call count of 3271
against a thirty-request draft-token count. And run J's per-prompt table had
contained, since it was published, the fact that `n_max 8` loses 14.8 % on
aggregate while winning on three prompts of ten, with acceptance separating the
winners from the losers by 9.5 pp and no overlap; no sentence had drawn it out,
and now one does.

**And thirteen more were, once the tables were parsed rather than read.** The
run registry published run C as three repeats where its manifest says five, and
run D as thirteen arms where it has five of run C's thirteen; it said "30
requests each" for runs A and B, which is B, while run A's two speculative arms
abort part way and reach twelve. Run E's row named three of the four draft
lengths it swept and run K's named a range containing lengths it never ran. A
blank line inside the tier registry orphaned its **v4 audit** row from the
header, so GitHub rendered the controlled tier as literal pipes; a test now
refuses any table row with no header above it. A14 called two runs' recorded
argv byte-identical, and one of the thirty tokens differs: the listening port.
A17's length-matched split table carried run V's largest **mode** contrast,
11.90 pp, in a column asking for its largest length-matching shift, which is
14.02 pp; the same entry said the external drafter appears three times across
the thinking-off runs where it appears five; and the sentence under its
per-repeat table quoted three coefficients for a four-row table, so a reader
lining them up read `spec-dflash-n2`'s 1.84 % as `spec-mtp-n2`'s. A15's
O2-against-T baseline row read +0.54 % where the pooled rates give +0.53 %.
`BENCHMARK_ENV.md` put run N in the `-fit on` group when N ran pinned at
`-ngl 999 -c 16384`, named no run W at all, and counted seventeen telemetry
traces and twelve compact ones where the tree holds sixteen and eleven. The v4
file map's lead-in said forty-one directories while the row beneath it said
sixty-five, and the README's data map said 62 v2 logs where the three `v2_*`
directories hold 61, one of them the verbose trace it counts separately.

**And one check read a file no clone has.** The fitter placement ERRATA A14
cites lives in `v4_audit_2026_08_25/data/matrix_M.log`, which `.gitignore`
excluded along with every other `*.log`. It surfaced as a crash inside the
coverage probe's worktree, a clean checkout of HEAD, where the checker died
373 assertions early: the probe's own baseline was broken and it was measuring
nothing. The eight v4 harness logs are committed now and a test refuses any
path the checker opens that a fresh clone would not have.

**Figures in those tables that cannot be re-derived here say so.** A14's
`n_batch 2048, n_ubatch 512` came from a `-v` server log this repository does
not commit, C4b's ~83 °C is the card's datasheet slowdown point rather than a
reading, and the host table in [`RETEST_TODO.md`](RETEST_TODO.md) is a
read-only probe of two machines on one day: free disk, driver versions and
the other box's GPU and CUDA cannot be recovered from anything committed
here. Each is marked; the alternative was to check a literal against itself
and call it verified.

Eighty-four tables are parsed that were not, eight of them census corrections
and seventy-six new readers: every cell of run C's thirteen arms, run O's
head-to-head, the v1 representative table including both of its range rows,
runs J, K, L and N, the V2 and V3 columns of the W table, both run registries,
A4's log reconstruction, A14's M1-against-Q comparison, A16's six invocations,
A17's per-repeat rates and its length-matched split, B4's family minima, the
BOS-override table in the two documents that publish it, run I's acceptance
under batching, the three batching tables, C4b's thermal table, v3's
cross-method ranking, Exp 2's variance decomposition, the memory-policy table
and both file maps, and last the six that had held out: the status board and
the host probe in [`RETEST_TODO.md`](RETEST_TODO.md), A14's between-run
spreads, A8's `p_min` sweep, the README's upstream-issue table and the v4
audit's open gaps. The census itself is now a gate: `verify_claims.py` pins
the table count exactly and the parsed count as a floor, so a new table must be
parsed or accounted for, and every markdown file must be either censused or
excluded with a stated reason.

**And the probe's control was wrong a third time, in the direction that
reassures.** Run as sixteen shards against one `.git`, each shard's baseline
came back with three failing assertions that a single shard alone does not
have: several claim checkers calling `git` at once make the git-gated
assertions flake. A perturbation counts as caught when the failure set grows,
so a flaky baseline turns unread numbers into caught ones and the run reports
a clean tree it never measured. Both probes now refuse to report at all unless
the unperturbed worktree passes, which is the rule `tests/data_mutate.py` has
always had for its mirror, and they check it again after the work: passing
once does not bound half an hour of running beside seven other shards, and
the second check also catches a perturbation that was written and never
restored.

**Nine fixes answered a review, and none of them had a mutation.** This file
opens by saying that a mutation which survives means its guard is decorative,
and the rule had been applied to every earlier batch and not to the one that
answered the fifth review: the concurrent hard cap, the census threshold, the
extractor timeout, the sampler's root selection, the probe's signal handling
and its end-of-run control, and the publisher's counts all shipped with a test
and without a mutation. Nine are added, so the suite runs 69. Two of them were
found to be needed rather than assumed: reverting the census threshold to
`< 3` and running the whole claim checker produced no new failure at all,
because no published table has one or two value cells, so the fix was correct
and completely unguarded. Synthetic fixtures now carry it: a one-cell table, a
two-cell table, a path table, and a commit hash beside two real numbers.

**The mutation anchors had no uniqueness rule of their own.** `data_mutate.py`
has refused an ambiguous document anchor since the split table perturbed
whichever of two identical rows came first; `mutate.py` uses
`replace(..., 1)` on source and never checked, and one anchor had already gone
ambiguous, because two teardown tests share three verbatim lines. Both files
are held to the rule now, and a second check refuses a mutation that names a
test class which no longer exists, since the runner exits non-zero either way
and would report it caught.

**The host sampler chose its roots by searching every command line.** Its own
docstring says attribution is by descent and not by name, and the descent
started from roots picked with `any(n in cmd for n in ("llama-server", ...))`.
A `llama-server` belonging to somebody else became a root, its whole process
tree counted as ours, and the interference this sampler exists to see went
invisible in the one reading that mattered. It takes `--root-pid` from the
driver now, checked against `/proc/<pid>/stat`'s start time each tick because a
pid is reused and the pair is not; without one it falls back to the positional
matcher rather than a substring, and every row records which of the two
produced it.

**The census counted a table with no numbers among those carrying
measurements.** `README.md`'s `| Path | Contents |` has no numeric cell at all
and was counted because the classification asked whether a parser reads a table
before asking whether the table has a value in it, and a parser does read that
one, for its paths. The population is decided first now: on the tree that pass
ran on **124** tables carried measurements, not 125, and the number of numbers
in it did not move, still the 2 373 it then held, which is the check that the
table really carried none.

**An extractor could hang with nothing to stop it.** The timeout that bounds it
shipped without a test, so removing it again would have cost nothing; a test
runs an extractor that never returns and requires the failure to name it and
the last input it was given.

**The publisher called a character count a byte count.** It reported "31015
bytes" for a body GitHub's API returns as 31083, because the pull-request body
carries 36 non-ASCII characters, 28 of them U+2212, and `len` on a Python `str`
counts characters. The comparison behind it was right, and comparing the two
strings is a byte-for-byte comparison because UTF-8 encodes a string one way
only; the number printed beside it was the one that invited a mismatch that was
not there, in the one tool whose whole purpose is to prove the body landed
unchanged. It prints both counts now.

**And a probe stopped with `pkill` left its worktree behind.** The throwaway
checkout is removed by an ExitStack callback, and Python's default SIGTERM
handling ends the process without running `finally`, `atexit` or any such
callback, so every probe stopped that way leaked 163 MB and an entry in the
repository's worktree registry. Seven of them had a 16 GB tmpfs down to 176 MB
free, which is the kind of thing that breaks the next run rather than the one
that caused it. The probes now install a handler that raises `SystemExit`, so
the stack unwinds, and prune the registry on the way in for the case that
cannot be caught at all.

**The host guard refused the suites it protects.** Started side by side,
`tests/mutate.py` and `tests/data_mutate.py` each copy `bench/` into a
temporary directory and run the harness there, and each read the other's
`python3 /tmp/tmp*/work/bench/retest_runner.py` as a live measurement. The
match was right about the name and right about the position and still wrong
about what the process was: a measurement does not live under the temporary
directory. A test now pins both directions.

**The larger half is prose, and it is not fixed.** Most decimal numbers in
these documents are not in any table, and a seeded sample of them was perturbed
the same way: 36 of 40 accepted a wrong number, against 40 of 40 on the run
before, and the 95 % Wilson interval puts the unguarded fraction at 76 % or
above. The result is in ERRATA A19. The probe is not
run in CI; about twenty seconds a table is close to an hour, and it needs a git
worktree.

**This repository's verification pipeline invalidated somebody else's
measurement, so it can no longer start one.** On 2026-08-27 a benchmark harness
sharing the host logged two `host_contended` incidents against `python3`, and a
four-hour phase had to be re-run to clear the flags. The obvious explanation was
wrong: `tests/mutate.py` and `tests/data_mutate.py` are **sequential**, one
subprocess at a time, and neither can produce six cores of load. What did was
(a) several full pipelines launched back to back until they overlapped, each
single-threaded, and (b) `analysis/plot_v4_runs.py` importing matplotlib and so
numpy, whose OpenBLAS spawns a thread per core, with nothing in this repository
pinning it. Attributing the burst to "the suite is parallel" would have been a
tidy story no measurement supports, which is the mistake A12 was written about.

`bench/host_guard.py` refuses to start either suite while a GPU lock is held or
a benchmark process is running, takes a whole-host `flock` so two pipelines
cannot overlap, pins the BLAS thread variables for itself and every child, and
renices. CI is exempt, because there is no card there to contend for.

**A16 says "nothing recorded distinguishes them", and that is a statement about
the recording.** Two quantities were never in it, both named now, neither
offered as the explanation. The GDDR6X memory-junction temperature is a
separate sensor with its own throttle point that reduces **memory bandwidth**,
and NVML does not expose it on Linux; every `temp_c` column here is the core
die. Host CPU load was never sampled at all: `bench/gpu_telemetry.sh` queries
`nvidia-smi` and nothing else in all three of its schemas. The second is now
recordable: `bench/host_guard.py --sample` writes busy percent, load average
and the largest process that is not the benchmark's own descendant, attributed
by descent rather than by name, and **no run in this repository has it**, V2,
V3 and T4 included.

**CI got `shell: bash`.** A `run:` step with no shell gets `bash -e {0}`, which
has `-e` and not `-o pipefail`, so `checker.py | tail -1` reports `tail`'s exit
status and a checker can print FAIL into a green job. No step here pipes today;
naming the shell is what keeps that true when one does.

**And the workflow that would do that in CI had never run, until it did.**
`.github/workflows/evidence.yml` is wired to `workflow_dispatch`, to
`release: published` and to a weekly cron, and none of those three can fire:
GitHub registers workflows, schedules and dispatch targets from the **default
branch**, and that file exists only on `audit-2026-08-25`. Dispatching it
returns 404. Every re-derivation figure this repository published was therefore
produced by running the script, and the documents said "CI does it" in three
places. A `push` filter was added so the chain could be demonstrated rather
than asserted, and it fired for the first time on 2026-08-28: the workflow
fetched the archive, verified it against the manifest, unpacked it, re-derived
the committed JSON from the raw logs and ran the claim checker over the result,
and failed on the chart comparison alone. It passed in full thirty-five minutes
later and on five days since. It has failed since 2026-08-31 and for a
different reason: the tag it names became `v4.2`, which
this branch has not cut, so it stops at the download and skips the rest. The
other three triggers still become live only on merge.

### Run W, and the question it removes

Five sessions of a 10 × 10 Williams square on 2026-08-28, 500 of 500 arm-runs,
none failed. V3's treatment definitions, prompts, models and server build,
under a later harness revision, so not V3 verbatim: the exported treatment
variables were diffed and differ in three places, all of them the schedule.
Every arm visits every position exactly once **and** is preceded by every other
arm exactly once within a repeat, verified from the arm-runs' own `t_start`
order in all five sessions rather than from the manifest.

The analysis plan was committed in `PROSPECTIVE_ANALYSIS_PLAN_W.md` while the run was at
360 of 500, and the checker asserts that commit is an ancestor of the one
carrying the data. It fixed the estimators, the thresholds, and four things
that would cost something: that a three-way disagreement would be published as
one, that the estimand would not be switched afterwards, that no claim would be
made about identifying A16, and that a null on the predecessor question would
be reported with its interval rather than as absence.

**The mode effect survives a carryover-balanced schedule.** `spec-dflash-n4`'s
sign flip, the strongest result in A17, now holds at +12.03, +12.17 and
**+12.10** pp across three designs. `spec-mtp-n2` and `spec-draft-n8` agree
with both earlier designs to a tenth of a point.

**And the arm this repository has been wrong about twice resolves.**
`spec-dflash-n2` reads **+8.29 pp [+7.97, +8.60]**, an interval that overlaps
V3's and does not overlap the crossover's [+4.86, +6.99] at all. The two
within-invocation designs agree; the between-invocation one does not.

**The design can ask about first-order carryover; five sessions could not
answer.** With every arm preceded by every other exactly once, the contrast
between running after a capped neighbour and after a free-running one is
−1.20 % [−2.61, +0.22] for that arm (the largest of any by six times, pointing
the way A17 guessed, negative in four sessions of five) and **no arm's interval
excludes zero**. At five sessions that is a null at this power, reported with
the interval. An earlier version of this entry added that an effect large
enough to explain a 2.4 pp gap "would sit far outside it": −2.4 is inside
[−2.61, +0.22], so it said the opposite of what the interval supports, and the
two are different estimands besides. Run W2 below is what removes the
candidate; this entry claimed it two days early ([A19](ERRATA.md#a19-how-much-of-what-this-repository-publishes-would-notice-if-it-were-wrong) item 46).

**Run W2 answers it: twelve sessions, and the plan committed before the driver
ran.** 2026-08-30 to 2026-08-31, the same 10 × 10 Williams square, 1200 of 1200
arm-runs, none failed, 17.07 h of continuous telemetry, a separate invocation
under its own `BENCH_RUN_LABEL` so nothing pools it with W. The thermal
slowdown flags appear on 13 and 10 of its 10 271 loaded samples, the same order
as W's, and the card stays between 45 and 73 °C.

It also replicates the mode contrast it was not run to measure. A17's table
gains a W2 column and all four of its intervals overlap W's: +12.13 against
+12.10, +9.60 against +9.53, +8.50 against +8.29 and +6.34 against +6.35. For
`spec-dflash-n2` that puts a second within-invocation interval, [+8.08, +8.91],
clear of V2's [+4.86, +6.99] on twelve sessions rather than five. Nothing about
W2 was chosen after those four numbers were seen; its session count and
estimand were fixed before it started, and its estimand is the predecessor
contrast rather than this. The pre-registered
matched contrast for `spec-dflash-n2` is **−0.14 % [−0.68, +0.41]** against W's
−1.05 % [−2.97, +0.86], with a session SD of 0.858 where W's five sessions had
implied 1.543. Tested in the units the 2.4 pp gap is quoted in, the predecessor
moves A17's published `shift_pp` by **+0.49 pp [−0.80, +1.77]** where W's five
sessions gave +1.52 pp [−0.98, +4.02]: W's interval contains 2.4 and W2's does
not. Two arms of ten have intervals excluding zero, `spec-mtp-n2-cap` at
+0.11 % [+0.00, +0.23] and `spec-draft-n8` at +0.06 % [+0.01, +0.10], both
under an eighth of a per cent and both reported because the plan named that
outcome in advance. What is left is A16, unchanged.

What is left is A16, and W reproduces it on the largest dataset yet: mean
within-session CV of **1.69 %** for `spec-dflash-n2` against **0.31 %** for no
speculation, on work identical to the token. Across all 5000 request rows,
every arm produced one distinct output set and one distinct drafted/accepted
pair, matching V2's and V3's counts exactly.

**A process failure the run exposed.** W's manifests pin a `runner_sha256` that
resolves to no commit: the runner was deployed from the working tree, the run
started, and the file was edited twice more before being committed. The exact
source is archived under `v4_audit_2026_08_25/harness/`, verified to hash to
what all five manifests name, and the checker now requires every run's harness
hash to resolve **either** in history or in that directory, so a run whose
harness is neither cannot pass unnoticed. Archiving is not a substitute for
committing first.

### The three runs the third review asked for

Nine and a half hours on the bench card, 618 arm-runs, none failed.

**Run V2; the crossover.** Eight sessions of two halves in `AB BA BA AB BA AB
AB BA` order, so each mode ran first four times and second four times with the
two orders balanced in mean time position. 400 of 400 arm-runs. The hard cap
raises every arm and every interval excludes zero; `spec-dflash-n4` is **−1.66
%** free-running and **+10.37 %** capped, both clear of zero on opposite sides,
so the published "`n_max 4` goes negative" with thinking off is a length
artefact. Run V's own numbers land inside the eight-session intervals for three
of four arms, and **outside** for `spec-dflash-n2`, whose +9.26 pp is above all
eight sessions. That arm's shift is +5.92 pp [+4.86, +6.99]. The review's
objection was right and its size is **about 3.3 pp on one arm**.

**Run V3; both modes in one square.** `BENCH_HARDCAP_SUFFIX` puts `<arm>` and
`<arm>-cap` in the same balanced 10×10 rotation, so the contrast is
within-invocation. 200 of 200 arm-runs, two sessions. Three arms agree with the
crossover to a tenth of a point. `spec-dflash-n2` does not: **+8.65 pp** within
against **+5.92 pp** between, with V3's own two sessions 0.06 pp apart. The
absolute rates say why; four of five arms reproduce between the designs to
0.01–0.3 tok/s and the baseline to a hundredth, while that one arm moves −0.95
% free-running and +1.4 % capped, in opposite directions. The repository
reports the effect for that arm as **design-dependent, +5.9 to +8.7 pp**,
rather than picking the design that flatters it.

**Run T4; the checkpoint boundary, split.** The third review's P0-2 was that
A12's timers surround calls beginning with `ctx->synchronize()`, so 39.07 s
might be an attribution to the API boundary rather than to the copying.
`bench/apply_split_timers.py` drains explicitly and times the drain. The answer
is **0.002 s of 39.09 s: 0.003 % of the excess**. The queue is already empty
when the checkpoint is taken. Every component reproduces (17.336 / 16.346 /
5.412 against 17.34 / 16.33 / 5.41), the counts reproduce (785 creates, 728
restores, six times), the excess reproduces (71.49 s against 71.4), and the
share is **54.7 %**, exactly as published. The concern was legitimate and the
measurement refutes it. The llama.cpp tree was restored to stock afterwards,
verified by the library hash returning to `a0cbe4d0…`.

**And T4 cornered A16.** Six repeats inside one invocation: `spec-dflash-n2`
reads 139.36, 139.72, 139.93, then 145.04, 146.01, 144.43, a **3.9 % step**
partway through, on byte-identical output, while the two arms interleaved with
it hold 0.12 % and 0.55 %. Three arms rotating means the predecessor varies,
and it explains nothing: both predecessors produce both levels. 272 telemetry
samples explain nothing either; across the step the card is *hotter* (63.4 →
64.9 °C), the SM clock is flat, the power is flat, and `sw_power_cap` is
asserted in 17 samples while the arm is slow and 36 while it is fast. Every
recorded physical quantity either does not move or moves the wrong way. A16's
"invocation effect" is really an **arm-run-level** state that steps on a
timescale of minutes.

**What the runs cannot do**, stated because the review predicted it: both V2 and
V3 use cyclic rotations, which balance position and fix the neighbour. Only T4,
with three arms, varied the predecessor. Randomising the order is the next
experiment and it has not been run.

**The evidence for all of it is published.** A second tranche on the same
release — `raw_logs_20260827.tar.zst`, 618 logs, 2.9 GB uncompressed, sha256
`d56a7f88…` — kept separate so the first archive's digest keeps meaning what it
meant. The manifest grows to 1320 logs and 21 traces, all 620 new entries
verified against it before publishing, and the re-derivation still returns 526
of 535, 12 of 12 and 12 of 12 identical with nothing differing.

**One perturbation survived the clean mirror**, and it was a real gap: the v4
README's run-M table publishes six aggregates that the checker computed from
the data and compared with literals, never with the document. Changing a
published 127.3 to 130.3 passed everything. That table is parsed cell by cell
now, which is the same defect as the four tables the second review's pass
found, in the one place a later edit put it back.

**The pull request body is a published document, so it is in the tree now.**
The third review's P0-4 was errors in it. Fixing them once fixes nothing: it is
five numeric tables and four counts, and nothing was reading any of them.
`PULL_REQUEST.md` holds the body, `analysis/verify_claims.py` parses all five
tables cell by cell against the data, and the counts it quotes (assertions,
regressions, perturbations, run directories) are derived from the files they
describe rather than typed. `gh pr edit 2 --body-file PULL_REQUEST.md` is what
publishes it.

**That guard found one on its first run.** The checkpoint cost table appears in
[`README.md`](README.md) and in the body as four merged rows, and both
published the restore share as **30.5 %** — 22.9 + 7.6, the two component
shares each rounded to one decimal and then added. The merged row is 21.74 s of
71.4 s, which is **30.4 %**, and with it the column adds to exactly 100.0
instead of 100.1. A12's own four-row version has been parsed since 2026-08-26;
the merged version was parsed nowhere. New **B9**.

**And the count the body publishes could not hold in both places it is
checked.** Eight of the checker's assertions are git-provenance checks that do
not run where there is no `.git`, and `tests/data_mutate.py` runs the checker
in exactly such a mirror. The first version of the count check compared against
whatever had run, so it was right in a checkout and wrong in the mirror, which
the mirror reported, correctly, as "the checker fails on an unperturbed
mirror". It compares against a full checkout's count now, and a test derives
the eight by running the checker twice with `git` shadowed by a failing shim,
rather than trusting the constant.

**A perturbation anchor could match twice.** `edit_doc` checked that its anchor
was *present*, not that it was *unique*, so `| **39.09** | **54.7 %** |`, two
identical rows of run T4's split table, perturbed whichever came first. It
still fired, which is exactly the problem: an anchor can go ambiguous, or stop
meaning what its perturbation's name says, and the suite prints the same "all
detected" either way. It refuses a non-unique anchor now, and a test resolves
all 39 against their documents on every run.

**And "comparable" was decided in two files.** A16's twelve comparable runs are
selected by seven manifest fields in `analysis/verify_claims.py` and by the
same seven in `analysis/plot_v4_runs.py`. The checker also filters on the run
date, because A16 is *twelve times in one day*; the chart did not. Run T4
satisfies all seven and ran on 2026-08-27, so the moment it was committed the
checker still said twelve and the two-level chart quietly became thirteen; the
chart check caught it as "stale", which is the right alarm for the wrong
reason. Neither file was wrong about its own arithmetic. The chart carries the
date filter now, and a test asserts both files carry it *and* that some run
actually distinguishes the two rules, so the test cannot pass by there being
nothing to separate.

**And run T4's split is committed as data, not just as prose.** 39.09 s and
0.002 s existed only in the document; `checkpoint_timers_20260827_split.json`
holds all 18 arm-runs, `rederive_from_logs.py` regenerates it from the 18
attested server logs, 18 of 18 identical, and the checker reads the dump rather
than a literal.

### The third review

A third review asking for changes, at head `8954411`. Every accusation in it was checked
against the code and the data before anything moved, and every one that could be
checked was right.

**A17 does not identify what it claimed to.** Run V measured `ignore_eos` by
running the whole freerun matrix first and the whole hard-cap matrix
afterwards: `22:31:46` against `22:48:08`. Each half is position-balanced
internally and the *mode* is not randomised at all, so it is confounded with
sixteen minutes of elapsed time. That would be a footnote if A16 did not exist:
A16 establishes an unexplained, DFlash-specific invocation effect spanning
**9.4 pp** on the same drafter, and the shift A17 reports for `spec-dflash-n2`
is **9.26 pp**. A17 is now a fixed-order sensitivity analysis, the two halves
are named as two different estimands (natural-completion latency and an
equal-token-count microbenchmark, neither one the corrected version of the
other) and RETEST_TODO P1-3 is reopened for the crossover, and
`bench/run_v2_crossover.sh` is the design it needs: four sessions in
AB/BA/BA/AB order, run V's configuration otherwise verbatim, with the session
as the resampling unit. It has not been run.

**A12 measures the API boundary, not state-copy time.** The timers surround
`ckpt.update_tgt` / `load_tgt` / `load_dft`, which call
`llama_state_seq_get_data_ext` and `set_data_ext`, and at `3737e4137` those
begin with `ctx->synchronize()`, verified in the tested tree at
`llama-context.cpp:4083`. So 39.07 s is elapsed *inside* the checkpoint calls,
synchronisation included, and 54.7 % is an attribution to that boundary. Some
of the wait would be paid elsewhere on a path with no checkpoints rather than
vanishing. Splitting it needs a second timer inside the call and a re-run.

**Scope and arithmetic in the front matter.** The controlled tier is runs A–V,
not A–T3. Its findings are for `3737e4137`, not "current llama.cpp"; upstream
has open work on recurrent rollback, output row ordering and hybrid checkpoint
invalidation that touches these paths. The head-to-head figure said "nine
methods" for eight speculative configurations and a baseline. "The three
methods v1 benchmarked are the bottom three rows" was wrong twice over: v1
tested three methods and they occupy **four** rows, because the external
drafter appears twice. "Every thinking-off comparison is length-confounded"
excluded run V's own hard-cap half.

**The headline interval is within-invocation and now says so.** The column read
`95 % CI (t, over blocks)`; the blocks are nine sequential blocks of one
invocation in one fixed rotation. The primary figure for `spec-dflash-n2` is
the repeated-invocation range, **+17 % to +27 %**, with O2's point estimate and
its own interval below it, and the twelve runs are not twelve independent
measurements either, so that range is a bound on what was observed, not a
confidence interval.

**CITATION.cff claimed more than the harness measures**: "recorded batch width"
is concurrent client requests outstanding, which is not how many sequences share
a decode graph, and "preserves raw logs" is preserving their hashes.

### The harness fail-open paths the same review found

- **Only `BENCH_THINK` parsed strictly.** `BENCH_IGNORE_EOS=oen` selected off,
  `BENCH_FIT=onn` selected off, an unknown `BENCH_FLAVOR` selected master, and
  `BENCH_CONCURRENCY=0` was clamped to 1; silent treatment drift, which is the
  defect this audit was written about. Every treatment variable now goes
  through one strict parser and a typo stops the run before any GPU work.
- **`BENCH_EXPECT_LIB_SHA256` accepted a prefix** and compared one library.
  `libllama.so`, `libggml-cuda.so` and `libggml-base.so` all decide what a
  decode does. The whole map is pinned to the run's first arm-run now, and the
  expected digest must be all 64 characters.
- **The teardown guard allowed 2 GiB of residual VRAM** — most of the margin the
  fitter works in, next to a docstring saying a 120 MiB allocation killed the
  next arm, and an unreadable `nvidia-smi` returned *settled*. It is 128 MiB
  over the pre-run reading, held for three consecutive readings, and a card
  that answered before the arm-run and not after is a failure. *Not* a host
  with no reading at all: the first version of this fix missed that distinction
  and took four end-to-end harness tests down in CI, which has no `nvidia-smi`,
  a guard about memory coming back cannot apply where memory was never
  observable. Both directions are tested, and the CI condition is pinned
  locally with a failing shim on `PATH`.
- **`paired_blocks.py` and the runner disagreed on "balanced"**: a two-arm,
  four-repeat schedule the runner accepts was reported unbalanced here, and the
  analysis wrote the same JSON either way. It shares the runner's definition,
  refuses to emit an interval unless `--allow-unbalanced` is given, and records
  the balance and the interval's scope in the JSON.
- **The t table stopped at df=10 and fell back to 1.96**, the *normal* critical
  value, so any run with twelve or more blocks silently got a narrower
  interval. It is computed now, from the incomplete beta, matching published
  tables to three decimals from df=1 to df=120. `--iters N` parses as well as
  `--iters=N`. No published interval moved: the largest committed run has nine
  blocks, df=8, inside the old table, which is asserted rather than assumed.
- Regenerating all seven `paired_blocks.json` under the new gate found that
  **two of them came from schedules that are not position-balanced** and said
  nothing about it: run T rotates three arms over four repeats, and the
  head-to-head run has three blocks of a nine-arm design. No document quotes
  either interval, and every file now records its balance, whether the override
  was used, the t critical value, and that the interval covers blocks within
  one invocation and not the variation between invocations. Not one measured
  value changed in any of the seven.
- **`tests/mutate.py` still had its sidecar recovery**, which wrote a path read
  out of a file into ROOT without resolving it. The mirror made it unnecessary;
  it is gone.
- **The model path was never read back.** `server_identity()` looked for
  `llama_model_loader: loaded meta data from`, and llama.cpp prints `loaded
  meta data with N key-value pairs and M tensors from PATH`. The pattern has
  never matched, so `model_path` was absent from every arm-run identity ever
  recorded and nothing noticed, because the field was only compared when
  present, which it never was. It matches now, `/props` is read back as a
  second independent source, and both are compared against the target the
  manifest names.
- **`stage_mtp_source.py` silently picked the first of several indexes** and
  reused the stage directory, so a stage could hold two checkpoint generations.
  It requires exactly one index, checks every shard the index names exists and
  that nothing unreferenced is present, builds in a fresh directory and renames.

Thirteen net new mutations, 24 to 37, and 29 new tests, 61 to 90, cover the
above.

### What the adversarial pass over this round found

The same treatment, turned on the fixes above before they were pushed. Nine
things, six of them defects introduced in this round:

- **The library-map check cascaded off a crash.** `check_identity` seeded its
  baseline from the first arm-run it saw, and a crashed arm-run has hashed
  nothing, so one crash made every later arm-run report that the libraries had
  changed, burying the crash under a wrong finding. An empty map is skipped
  now; the crash is its own finding.
- **The strict parser refused a spelling this repository has used.**
  `matrix_L_thinkoff` records `think_env: "think_off"`, which the old parser
  accepted and the new one dropped. A parser that cannot reproduce a published
  run is a worse failure than the one it prevents. `think_off` and `think_on`
  are back, and a test pins them.
- **`bench/run_v2_crossover.sh` called `~/bench/gpu_telemetry.sh`, which the
  bench host does not have**: it lives only in this repository. Both the runner
  and the telemetry script are resolved from three candidates now and the
  script stops before any GPU time if either is missing.
- **The release notes' reproduction recipe was wrong**, in something already
  published: it omitted copying the committed arm-run JSON beside the logs and
  the four-run rename map, so anyone following it would have seen a hundred
  records "not regenerated" and concluded the archive was incomplete. Corrected
  and re-uploaded.
- **`rederive_from_logs.py` matched regenerated accounting records back by log
  basename.** `spec-draft-n8__rep0.log` exists in several runs, so the first run
  containing that name won. Each record is tagged with the run the extractor was
  called for.
- **"Run V bounds the size of that confound" claimed more than the design
  supports**, in two documents. If the shift is the invocation effect, run V
  bounds nothing. It measures a difference it cannot attribute.

Two more that the pass found in machinery this repository had already been
relying on, and which matter more than any of the above:

- **The perturbation suite was proving almost nothing.** `tests/data_mutate.py`
  restores only what each perturbation *declares* it touched, and one of them
  ("run T's telemetry runs 5 C warmer") declared nothing. It sits at index 10
  of 62, so the mirror kept a telemetry trace 5 °C warm for the rest of the
  run, the checker was **already failing** for every later perturbation, and
  `returncode != 0` was recorded as "caught" without the guard that should have
  fired ever firing. **51 of 62 results were worthless.** Declaring the restore
  fixes it; the runner now refuses to start if any perturbation omits the
  declaration, and re-verifies the mirror after the last restore.
- **Which uncovered a real hole.** With the mirror actually clean, one
  perturbation survived: a 5 % change to one request's decode time in run E.
  Every arm-run row carries `predicted_ms`, `predicted_n` and
  `predicted_per_second` **twice**, at the top level and inside `timings`, and
  different analyses read different copies: `paired_blocks.py`,
  `matrix_report.py`, `plot.py` and `plot_v4_runs.py` take the top-level one,
  `past_threshold_fit.py` the nested one. Nothing compared them, so a change to
  either was invisible to whatever read the other. All 18 344 rows of all 1805
  arm-run files are checked for agreement now, and both copies are perturbed.

- **And a third, from chasing the second.** `predicted_per_second`: llama.cpp's
  own field, and the one every **request-mean** column in this repository is
  the mean of: reports `1000 × (n − 1) / predicted_ms` in **30 300 of 30 344**
  rows. It is a rate over `n − 1` tokens divided by the time for `n`. The 44
  exceptions are the legacy `bcb5eeb64` runs, so the definition also changed
  between the two tiers. Every headline figure and every delta is a pooled rate
  computed from the raw fields and is untouched, and on a fixed-300-token run
  the bias is a uniform 0.33 % that cancels in every ratio, but where arms stop
  at different lengths it does not, which is [A17](ERRATA.md#a17-the-thinking-off-comparisons-are-not-comparisons-of-the-same-amount-of-work)'s
  mechanism riding on a different field. Written up as [B8](ERRATA.md#b8-every-request-mean-here-counts-one-token-fewer-than-it-timed),
  asserted so it cannot drift, and the recomputation left open rather than
  done: it would move several dozen figures by 0.33 % and change no conclusion.

And three where the pass improved an argument rather than fixing a bug:

- **The A16 comparator was the wrong one.** Quoting the full-day span, 9.4 pp,
  against Run V's 9.26 pp compares a whole day to sixteen minutes. The matching
  comparator is the closest pair: **U3 and U5 are six minutes apart and differ
  by 8.30 pp**. Tighter, and worse for run V.
- **The Run V ordering was inferred, and is now established.** Two `created`
  stamps do not prove sequencing on their own. The runner stamps `created`
  before the first server starts, and the freerun half's own measured work, 938
  s, fits inside the 982 s gap, so the halves cannot have overlapped.
- **The balance-agreement test checked six hand-written schedules.**
  `is_position_balanced` is duplicated because `paired_blocks.py` cannot import
  the runner, whose module body exits on a bad environment, so the copy is now
  compared against the original over every small schedule rather than a handful.

### The raw evidence is published

`raw-evidence-2026-08-27` carries `raw_logs.tar.zst`: 702 llama-server logs,
4071 MB uncompressed, sha256 `29c2401f…`, the digest this repository has
recorded since the manifest was written, and `telemetry.tar.zst` with the 19
GPU traces. Until now the repository committed the hashes of files nobody else
could see, which ties the derived JSON to something unavailable; the point of
publishing is that the extraction can be re-run instead of trusted.

`analysis/rederive_from_logs.py` re-runs the extractors over the unpacked
archive and diffs the result against what is committed. Measured, from the
archive alone:

| derived file | records | identical | not regenerated |
|---|---:|---:|---:|
| `data/spec_accounting_20260826.json` | 12 | **12** | 0 |
| `data/checkpoint_timers_20260826.json` | 12 | **12** | 0 |
| `data/acceptance_counter_comparison.json` | 535 | **526** | 9 |

**The probe would not start against a red tree and its own assertions made the
tree red.** Ten assertions read `coverage_attestations/`, so a stale set or a
changed shard count fails them and the run that would fix them refuses. The
refusal is right about the hazard it was written for, which is a FLAKY failure:
a perturbation counts as caught when the failure set grows, and a failure that
comes and goes reads as a catch. A stable one is subtracted out and masks
nothing. So `--bootstrap` lets failures named for the probe's own evidence
stand and nothing else, every attestation records which, and both the
aggregator and the checker refuse a set that declared anything wider. Two
assertions that read the aggregator's line out of this file's siblings were
named for the document rather than for what they read and sat outside an
allowance decided by the name; they are named for what they read now.
Underneath it was worse: in the worktree each shard clones there are no
attestations at all, and reading the first of an empty list raised IndexError,
so the checker died 31 assertions early and the probe took that truncated run
as its baseline. A number whose only guard is one of the 31 could not have been
caught, and would have been published as covered.

**The figures met the contrast standard for their shapes and not for their
type.** WCAG 2.2 asks 3:1 of a graphical object and 4.5:1 of text below 18 pt,
and the Okabe and Ito palette, which was adopted here for its behaviour under
colour vision deficiency, was designed for fills: against white its vermilion
measures 3.87:1, its green 3.42 and its reddish purple 3.06. Every chart prints
its values as text in the series colour, so those labels failed a criterion the
lines beside them passed. Each hue now carries a darkened twin used for type
only, at 5.34, 5.48, 5.58 and 5.85:1, and the series keep the hues they were
chosen for. The per-prompt heatmap chose its ink from the value rather than
from the cell, and the scale puts its high end in deep blue, so every 100 and
the single 104 were printed in black on the darkest cells on the figure; the
ink is now chosen per cell. The acceptance accounting figure drew each bar's
accepted portion in a different colour under a legend that named one, and
reserved a fixed band for a caption its legends already reached into, leaving
an empty third down the middle of the canvas. Both are fixed, and
`tests/test_harness_invariants.py` holds the palette, the per-cell ink and the
rule that no label is drawn in an unlightened series colour.

**The release body was published as an eighty-column file.** GitHub Flavored
Markdown preserves newlines in a release body exactly as it does in a pull
request body, and `gh release create --notes-file` hands the file over
verbatim: the published body rendered with 29 `<br>`. This repository has a
tool and a commit about the 412 that produced in the pull request body, and the
release was the one published surface with no tool.
`tools/publish_release_notes.py` reflows on the way out, takes the release name
from the notes file's own first heading so the two cannot disagree, and proves
both landed by reading them back and asking GitHub's renderer how many breaks
are in the result.

**The tag is `v4.2`.** It was `raw-evidence-2026-08-31-v4.2` for ninety
minutes, which put a date and a version in one name and matched neither of the
two families this repository already had: `v1.0` through `v3.0` for versions,
`raw-evidence-2026-08-27` for an evidence drop. The cost was not only tidiness.
`CITATION.cff` names the version `v4.2`, and the test that asks whether that
version is a tag compares the two strings; against a tag merely CONTAINING it,
the comparison failed and the test took its "no tag yet" branch. It passed
while the release existed, for the wrong reason, and the citation file went on
saying the version was not a tag, that the newest tag was v3.0, and that the
tag is cut at merge. Cutting the release had made four of its sentences false
and nothing noticed.

**The evidence workflow read a path that had not existed for weeks.** Two of
its three runs after the release started AFTER the assets were published,
fetched them, and died at `FileNotFoundError: /tmp/manifest`. The manifest is
split into `/tmp/manifest_all`, `_oob` and `_bench` so one tranche can be
verified in a root of its own, and the `preflight_tar.py` call kept the name of
the file that used to hold all of it. Nothing local could have caught it:
`ci_faithful.sh` does not reproduce `evidence.yml`, and says so, because that
job needs a published release, so the first run that could reach the step was
the one after the tag was cut. The rule is a test now: every `/tmp` path a
workflow reads has to be one the same workflow writes.

**`unit and mutation` was four minutes from its own timeout.** It ran 21
minutes on 2026-08-28 and 50 to 56 against a 60-minute limit on 2026-08-31,
because the data perturbations run the claim checker once each and the checker
has gained nine hundred assertions. The perturbations were sequential only
because they shared one mirror. They are sharded now, one per processor, each
shard with a mirror of its own so nothing it perturbs can reach another shard
at all, each checking its own mirror clean at the end.
`bench/run_data_mutations.sh` refuses more shards than the host has processors
or than the list has entries, refuses a scratch directory too small for two
mirrors per shard, collects every exit code, and requires the shards' caught
counts to add up to the whole list: a shard that silently did nothing fails the
total rather than passing quietly.

**And four defects in that launcher, found by attacking it rather than reading
it.** It counted the perturbation list before changing to the repository, so it
failed from any other directory. With one shard the shard took the same
whole-host lock the launcher was already holding and always exited. `grep`
matching nothing under `pipefail` killed the script at the line that computes
the diagnosis, so the one case those lines exist for, every shard having
failed, produced no diagnosis at all. And it had no scratch check while the
probe's launcher has one and records losing a set of attestations to a full
disk: eight shards want 3.1 GB and this host has 3.5 GB free.

**Four cross-references pointed at the wrong section, and two of them were
published that way.** A reference that names a direction rather than a target
is not checked by anything: reorder the document and it points at whatever
landed there. The README sent a reader looking for the definition of `draft_n`
to "the next section", which after the reorder was three sections short, and a
sentence about where this section used to live pointed at its neighbour
instead. In the published tree, `PULL_REQUEST.md` placed run W "in the section
above" when it is reported in the same one, and `v4_audit_2026_08_25/README.md`
attributed the `--fit-target 2048` reason to "the previous section" rather than
to the one that gives it. All four are anchor links now, which
`analysis/check_links.py` resolves, and the directional form is refused
outright: the checker allows it quoted, as this entry quotes it, and never
used.

**Zero records differ.** The nine that cannot be regenerated belong to
`matrix_G_dflash_20260826_000124`, `matrix_I_conc1_20260826_012917` and
`matrix_J_dflash_fit_20260826_014308`; three exploratory runs whose logs are in
the archive and whose arm-run JSON is not committed, because they never
finished their cell set and `check_data_integrity.py` refuses incomplete runs.
The extractor needs both halves. They are nine of the 535 rows behind A13, and
the claim there rests on the other 526. The re-derivation asserts that gap is
exactly nine rather than tolerating any shortfall, and two mutations prove it
fails on a changed record and on a vanished one.


Two external reviews of the pull request. Every specific accusation in the
second was verified against the code before anything was changed, and all four
were correct. Then the same treatment was turned on this branch's own commits,
and the second half of this entry is what that found.

### Measured

- **Run O3** — the nine-arm headline matrix repeated five hours later on the
  same stock binary, asserted per arm-run. **810 of 810 request-pairs are
  byte-identical** to run O2 and acceptance matches to a tenth of a point on
  every arm, while `spec-dflash-n2` moves −2.9 pp and the other eight move 0.2
  to 1.0.
- **Run U** — six independent invocations of one script, fifteen minutes apart,
  240 of 240 request-pairs byte-identical, spanning **8.3 pp**.
- **Run V** — the same five arms twice in one session, once as the archive did
  it and once with `ignore_eos` forcing exactly 300 tokens. Every arm moves
  6.31 to 11.90 pp and `spec-dflash-n4` changes sign. This closes RETEST_TODO
  P1-3 and turns [A17](ERRATA.md#a17-the-thinking-off-comparisons-are-not-comparisons-of-the-same-amount-of-work)
  from a subset argument into a measurement.
- Pooled over every comparable block, the headline arm spans **+17.0 % to
  +27.8 %** with `draft_n` 2441 and acceptance 72.3 % in all 43 of them: the
  speculative work identical to the token, only the time differing.

### What the adversarial pass found in this branch's own work

- **`tests/mutate.py` edited the real source files** and restored them in a
  `finally` that a kill does not run. It committed one of its own mutations:
  `bench/retest_runner.py` was published with `body.pop("ignore_eos", None)`.
  It runs in a mirror now, and `EveryPublishedFixIsStillHere`, added an hour
  earlier for exactly this, is what caught it.
- **A13 was built from `*__rep0.log`**: one arm-run per arm however many repeats
  it had, and it predated six runs. Over everything the base grows **73 → 517**
  and the claim survives with the margin narrowing **0.5 pp → 0.20 pp**.
- **The acceptance-threshold scorecard** reads the same file and was keyed by
  `(run, arm)` in a dict. Aggregated properly it is **78 / 86**, not 35 / 37.
- **`matrix_report.py --strict`** aggregated whatever files it found: a deleted
  arm-run and a renamed arm both passed.
- **The teardown guard** waited for the GPU to fall below an absolute threshold,
  so any other process on the card failed every arm-run.
- **The published checkpoint total was 39.08 by double rounding**; it is 39.07.
- **A withdrawn figure was still in the README** — 101.3 MiB, the sum A12
  retracted. Every retracted number is guarded now.
- **Four tables were computed and asserted while the tables themselves were not
  parsed.** Planting wrong numbers in the headline table, the O2/O3 replication
  table, the footnote, A12's accounting, A13's counters and C4b's thermals all
  passed. They are parsed cell by cell, and 36 perturbations are permanent tests.
- **Two committed `paired_blocks.json` came from an exploratory command** at
  `--iters=2000` rather than the documented default.
- **"No raw measurement file was edited" was not what happened**: three v2 files
  gained an audit block and had two strings corrected. Their 605 measurement
  values are fingerprinted against master and unchanged.
- **The pre-registration had no code.** Every other claim in this repository is
  produced by a script in `analysis/`; the one document written to be
  falsifiable, `v4_audit_2026_08_25/PREREGISTERED_PREDICTION.md`, was computed
  by hand. Recomputing it found that the *prediction*, written before the data
  existed, reproduces to the last decimal: the coefficients, R², leave-one-out
  error, collinearity, the power-law extrapolation, all three predicted rows.
  The *outcome section*, written after, had eight wrong figures:
  - the error column was computed from its own rounded display, which turned a
    **−0.6 %** miss at `n_max` 128 into a bolded **0.0 %**;
  - checkpoint traffic, 55.4 → 24.9 MiB per generated token, was computed at
    **101.3 MiB per checkpoint: the size A12 withdrew**. It is 44.8 → 20.2. The
    guard added earlier caught the retracted *number*; this was the retracted
    number *derived*, and nothing was looking for that;
  - 1639 checkpoints were attributed to **one 300-token request**. They are one
    arm-run of **ten**, 163.9 per request: a 10× overstatement, cited in the
    README's row for upstream issue #24055;
  - the no-speculation step was quoted as **7.87 ms** until this pass: a
    figure that is repeat 0 alone. Pooled over the five repeats the model is
    fitted on it is **8.11**, so the law's intercept is 3.3× it, not 3.4×;
  - nine **pooled decode rates** were published as "end-to-end throughput,
    which is what a user actually gets". End to end they are 30.2 … 8.8 tok/s
    against a **110.8** baseline, not 123.4;
  - and the earlier **+11.9 % and +15.9 %**, two of the three figures for how
    much the model over-predicts when fed measured inputs, were **repeat 0 of
    three**.
    Pooled they are +12.3 % and +18.1 %. The third, +24.8 %, matches no repeat
    and no pooling; it is +25.2 %;
  - the run-to-run scatter was given, here and in A7, as a range that no
    earlier figure produces. The three standard deviations are 0.05, 0.16
    and 0.11 tok/s;
  - and the per-round term's contribution, 91 % of the error draft volume
    leaves, was truncated rather than rounded. It is 92 %.

  Two figures that *looked* wrong were not. The residual step at the coverage
  threshold, **−0.39 pp**, and the **24–34 %** amortisation deviation both
  reproduce exactly; the first in `(measured − predicted) / predicted`, the
  convention [A10](ERRATA.md#a10-the-single-regressor-law-is-falsified-out-of-sample-and-p_min-is-the-lever-that-matters)
  publishes and this document did not state, and the second by fitting the six
  points the model was fitted on rather than everything below the threshold.
  Reading the step in the opposite convention gives −0.13 pp, which is what
  this pass first "corrected" it to before checking A10. Both conventions are
  now written down where the numbers are, and the conclusion never depended on
  either: the step is two orders of magnitude below the ±11 % scatter it would
  have to rise out of.

  `analysis/past_threshold_fit.py` now derives all of it, and 117 new assertions
  parse the document back against the script.

### Added

- **Run T3** — run T repeated at three **balanced** blocks on the same
  instrumented build, with the library hash asserted per arm-run rather than
  once per run. The checkpoint attribution replicates: 785 creates and 728
  restores in every arm-run of both runs, 39.16 s against 39.07 s, 54.6 % of
  the excess against 54.7 %.
- **ERRATA A16** — and run T3 also produced **byte-identical output** to run T
  on all three arms and all ten prompts, with identical acceptance, identical
  fit and identical GPU thermal and clock state, while running the DFlash arm
  **3.4 % slower on every prompt**. The cause is not isolated. The consequence
  is that the 1.6 pp paired-block interval on the headline arm describes the
  run, not the configuration: five runs of it span +20.7 % to +26.7 %, and the
  README now says so beside the table.
- `tests/fake_llama_server.py`: enough of the llama.cpp HTTP surface to drive
  the runner end to end in a second without a GPU. The guards that fire after
  the arm loop had no test because reaching them needed a 20 GiB model, and a
  mutation that deleted the completeness check survived the whole suite.
- `bench/collect_evidence.sh`, `requirements-plot.lock`, `analysis/plot_data.json`,
  and the SHA-256 manifest of the twelve run T logs.

### Fixed

- `BENCH_ORDER=latin` generated a cyclic rotation for any (arms, repeats) pair
  and labelled the result balanced. Three arms over four repeats rotates 0, 1,
  2, 0, which is what run T did, while its manifest recorded `latin`. The
  schedule is now built and validated before the first server starts, the
  unbalanced rotation is named `cyclic`, and the manifest carries the schedule
  and the verified balance. `analysis/paired_blocks.py` re-derives the schedule
  from each arm-run's monotonic timestamps and warns when the design does not
  hold, rather than trusting the label.
- `RUN_COMPLETE.json` was written unconditionally when the arm loop returned, so
  a run with a crashed arm still carried the marker every consumer reads as
  "this is a whole run". Gated on validation; failures write `RUN_FAILED.json`.
- `analysis/extract_checkpoint_timers.py` stripped only the literal
  `__rep0.log`, so repeats 1–3 were filed under separate arm names and the
  controls rested on one log each. Fixing it exposed the same class of defect in
  `analysis/verify_claims.py`, where the timer records were keyed by arm and
  four arm-runs collapsed into one.
- `analysis/check_data_integrity.py` counted arm-runs instead of checking the
  exact (arm, repeat) product, so rep0 twice and no rep3 passed. It now also
  checks each filename against the `arm`/`repeat` inside it, and validates
  `RUN_COMPLETE.json` against the directory it attests to.
- The headline footnote quoted run O's **+24.6 %** as though it were run O2's
  own figure. It was written when run O *was* the headline table and was not
  updated when O2 replaced it. The footnote is now derived from the data and
  asserted against the README text.
- The headline table's "95 % CI" column is the Student-t interval while the
  prose above it described a block bootstrap. Both are computed and both are in
  `paired_blocks.json`; the column is now labelled, and t is the wider of the
  two on every row.
- `stop_server` printed a warning when the driver had not handed the memory back
  and carried on, so the failure landed on the *next* arm. It is a run-level
  failure now.
- `bench/convert_dflash.sh` promoted the converted drafter to its final path
  before the load check, leaving a file the loader refuses where the next run
  would pick it up.
- `DATA_LICENSE` did not name `v4_audit_2026_08_25/data/**`, leaving 601
  committed measurement files without a stated licence, and `CITATION.cff`
  listed MIT alone for a dataset whose measurements are CC0.
- Provenance: `runner_sha256`, `harness_tree_sha`, `prompt_set_sha256` (over the
  prompts, not their label), and per-arm-run `server_lib_sha256`,
  `server_loaded_commit` and `server_log_sha256`. `BENCH_EXPECT_COMMIT` and
  `BENCH_EXPECT_LIB_SHA256` fail the run on the first arm that disagrees.
- CI pins every action to a commit SHA, installs the chart dependencies from a
  hash-pinned lockfile, checks the committed charts against the committed data,
  and aggregates every attested run under `--strict`.

---

## [v4.1] — 2026-08-26 · the controlled tier, and a reversal

The title loses the word "Historical": it was added by the v4.0 audit to stop
readers taking v1 as current, and it stopped being true the moment this
repository grew a controlled tier on post-merge master.

v4.0 audited what this repository had published. v4.1 measures what it had
never run. Nine of master's eleven `--spec-type` methods, on one card, against
a matched no-speculation baseline inside every run; the two left out are
`draft-eagle3`, which needs three extract layers this model does not expose,
and `draft-dspark`, which the runtime accepts only for DeepseekV4. The headline
reverses.

### Added

- **Runs I–O**, 110 arm-runs on post-merge master `3737e4137`, each with its own
  baseline, ABBA ordering, 3–5 repeats, per-request text and token ids,
  recorded batch width, and a continuous GPU trace:

  | run | question | answer |
  |---|---|---|
  | I | does batching rescue speculation, as upstream says? | no — no-speculation gains +64 % at eight in flight, the drafter −8 % |
  | J | DFlash off vs on, one binary — the A/B v3 never had | **+18.7 %**, and it reverses v3's sign |
  | K | where is the optimum, and does it survive batching? | a plateau at `n_max` 2–4, a cliff after; batching erases it |
  | L | does the win survive the workload changing? | it halves with thinking off, and `n_max 4` goes negative — **see [ERRATA A17](ERRATA.md#a17-the-thinking-off-comparisons-are-not-comparisons-of-the-same-amount-of-work): that comparison is confounded by output length and `n_max 4` is +14.1 % on the length-matched prompts** |
  | M | `draft-mtp` — the method the vLLM sibling uses | **+18.6 %** on aggregate at `n_max` 2, its best MTP arm; nothing was blocking it, it had never been run |
  | N | `ngram-map-k` / `k4v`, never run here | they never engage: 0.0 % acceptance, three lookup hits in thirty requests |
  | O | every method, one baseline, one matrix, one policy | a factor of five, split by drafter architecture |

- **ERRATA A11** — speculation is not output-preserving on this build. Against a
  determinism control that holds in every run (170/170 self-reproducible), token
  streams differ from no-speculation in 27–30 of 30 request-pairs at the
  300-token cap, and the divergence rate tracks output length.
- **ERRATA A12** — the mechanism, measured. An external drafter forces the
  hybrid target to checkpoint and restore 82.079 MiB of target state plus
  19.266 MiB of draft state on every partially accepted round: 772 saves and
  709 restores per arm-run, a nominal ≈118.7 GiB by event count × the size the
  server reports. DFlash logs none at draft lengths 1–16 and MTP none at 1–8.
  **Both figures were corrected after review**: the volume double-counted the
  draft component that `size()` already includes, and a "≈19.5 % of the wall
  clock" claim was withdrawn because the create and restore messages sit on
  opposite sides of the work they name.
- **Runs P, Q and R**, 60 further arm-runs. P and R repeat the key arms on a
  second set of **twenty** prompts sharing none with the v1 ten: long inputs,
  structured output, four languages, arithmetic, and two genuinely multi-turn
  exchanges, gated on the model recalling four-turn-old context. The decode
  speed-up moves by at most 4.3 pp, so it is not a property of the original
  prompt mix. Q settles the one anomaly v4.1 could not explain.
- **ERRATA A14** — within-run repeats are not an error bar. Ten (arm,
  configuration) pairs measured in two or three independent runs have a median
  between-run spread of 0.56 pp, and one pair differs by 8.6 pp with the argv,
  the drafter hash, the fitter's choices, the per-prompt draft counts, the
  acceptance, the temperature and the clocks all identical. A three-repeat
  delta is accurate to about a point, not to its printed SD.
- **ERRATA A13** — llama.cpp keeps *two* acceptance counters and this repository
  had only ever quoted one. Across 517 single-request arm-runs they agree to
  within 0.5 pp on every path that takes no speculative checkpoint (31 runs)
  and disagree by at least 1.0 pp on every path that does (42 runs), with no
  overlap. The worst case is 53.3 pp. This narrows A1's account of the upstream
  fix (the denominator moved, the early return did not) and it discredits the
  drafter's counter too: `spec-draft-n1` reports 1639 of 1639 accepted on an
  arm running at a quarter of baseline.
- `analysis/plot_v4_runs.py`, `analysis/extract_spec_accounting.py`,
  `analysis/compare_acceptance_counters.py`, `bench/stage_mtp_source.py`, and
  four new charts.

### Changed

- **The headline.** "Every tested condition that recorded speculative activity
  was slower" is still true of the archival tier, and the archival tier tested
  the losing third of the methods. Self-speculation wins here by a fifth;
  external speculation loses by three quarters. The README leads with the
  nine-arm table rather than with v1's negative.
- README now describes an archival tier and a controlled tier separately, and
  says which to cite about current llama.cpp. Its IMPORTANT block had claimed
  this repository holds only 2026-04-21…05-07 data and "is not a benchmark of
  current llama.cpp master"; both were false once runs A–O existed.
- `BENCHMARK_ENV.md` gains the v4 artefact hashes and, more usefully, the
  memory-policy table: `-ngl`, `-c` and `--fit-target` are variables across
  these runs, which is exactly why absolute rates do not transfer between them.
- `bench/retest_runner.py`: `BENCH_CONCURRENCY` now actually dispatches
  concurrently and reads the achieved batch width back out of request
  timestamps; `wait_health` notices a dead server instead of burning a
  300-second timeout; `stop_server` waits for the driver to release VRAM;
  `BENCH_CTX` and `BENCH_FIT_TARGET` make the memory policy explicit.

### Retracted

- **"A configuration is worth running when it clears ~48 % draft acceptance."**
  Published in this repository on 2026-08-26 and falsified the same day. It is
  12/12 inside the DFlash family and fails on the external drafter exactly
  where it should matter: `spec-draft-n1` reaches **68.7 % acceptance and is
  74.8 % slower**. Scored per family it is 21/23, and both failures are at the
  high-acceptance end. The threshold is a property of the drafter, not of
  acceptance: see A12.
- **"A Q4_K_M MTP head moves 6.8 pp against Q8_0 at unchanged acceptance."**
  Reported in this branch as measured-and-unexplained. Five repeats of each
  drafter dissolve it: the Q8_0 arm at `n_max 4` was one measurement that did
  not replicate. With it re-measured the Q4_K_M head is ahead at both draft
  lengths, which is the simpler and correct story.
- **Run N's "0.0 % acceptance".** Written and corrected the same day. The server
  counter reads 0.0 %; the speculator's own counter reads up to 70.0 %. What is
  measurable without either is that `generate()` was called 3271 times and
  returned a draft twice.
- **v3's DFlash direction.** What v3 measured at short draft windows was a
  binary change, not a DFlash penalty.
- **"vLLM MTP remains the only positive-yield speculative decoding path on this
  hardware"** (`pr_comment.md`). MTP now runs under llama.cpp here, and so does
  DFlash, and both are positive.

### Fixed

- The data map described `bench/retest_runner.py` as "never executed"; it
  produced every v4 measurement.
- `RETEST_TODO.md`'s MTP entry was wrong twice: first "no head weights", then
  "a converter gap". The weights are present, and the stock converter and stock
  runtime both support the architecture. A patch attempt refuted the second
  claim in one step with `TypeError: Cannot create a consistent MRO`.
- Two independent waiter loops raced and ran two benchmarks on one card for
  ninety seconds. Everything they touched was discarded and the runs re-driven
  from a single sequential process.

## [v4.0] — 2026-08-25 · audit and retraction

> Never tagged or published on its own. Its changes ship inside **v4.1**, whose
> tag both entries link to. The two entries are kept separate because they
> document different work: v4.0 corrected what had been published, v4.1 measured
> what had not been run.

A full adversarial re-examination of everything this repository has published.
No raw measurement file was edited; `results/`, `v2_3090_followup/v2_*/`,
`exp2_codejson_n3/master.log`, and `v3_dflash_2026_05_07/data/` are
byte-identical to the archived releases, and every historical aggregate was
re-derived from them and reproduced exactly. What changed is the wording, the
statistics, and the causal claims. Full itemised list with evidence:
[`ERRATA.md`](ERRATA.md).

### Retracted

- **"100 % draft acceptance."** The ratio is 1.0 by construction on this model.
  Qwen3.6-35B-A3B is a hybrid Gated-DeltaNet/MoE model, so the context reports
  `COMMON_CONTEXT_SEQ_RM_TYPE_FULL`; a partially accepted round then takes an
  early `continue` in `server-context.cpp` that skips both acceptance counters
  and re-verifies the truncated prefix. The drafter's own counters, printed one
  line below the quoted `1.00000 (115 / 115)` in the same committed
  `verbose.log`, report **115 of 214 generated draft tokens accepted (53.7 %)**
  and **33 of 81 drafts (40.7 %)**. `analysis/plot_accept_vs_speed.png`, whose
  entire x-axis was that artefact, is deleted.
- **"Vocab-matched draft model."** llama.cpp rejected the pair on its
  special-token gate and ran the token-translation fallback for every
  classic-draft measurement ever published here.
- **"The regression is structural / engine-independent / hardware-independent."**
- **"Q4 collapses the technique"**, **"the mechanism generalises to DFlash"**,
  **"co-trained heads are the only positive yield path"** — all removed from
  v3.
- **"Exp 2 refutes the workload-shape hypothesis."** Downgraded to exploratory:
  `-no-cnv` was rejected, `/no_think` did not disable thinking, and per-request
  outputs were never committed, so the intended treatment is unverifiable.
- **All "first public benchmark / first public datapoint" claims**, from the
  title, the v3 banner, `pr_comment.md`, and this changelog's v3.0 entry. No
  novelty search was ever performed.

### Corrected

- Pooled decode throughput now reported alongside the request-mean. The gap is
  material: classic draft `--draft-max 8` is −10.8 % by request-mean but
  **−19.0 % pooled**; `ngcache-1000tok` is −25.7 % pooled against its matched
  long-output baseline.
- `±` columns relabelled as across-prompt spread. Each v1 cell was measured
  once, so they were never run-to-run uncertainty. Exp 2's `± 7.57` decomposes
  into a trial-mean SD of **0.058** and prompt-to-prompt spread.
- "19-config matrix" → 19 run labels, of which 14 recorded a draft round and 5
  did not.
- "Every configuration hits 59–67 tok/s" → that band belongs to specific
  ngram-cache and classic-draft requests; the ngram-mod family bottoms out at
  119.8–129.6.
- "Entirely bimodal by prompt class" → three configurations produce three
  different prompt partitions; ngram-mod fires on exactly the chat prompts the
  old text said it could not.
- "all completions reach the cap" → true for the 300-token group, false for
  the 1000-token group (`baseline-1000tok` returned 354–1000 tokens).
- Target artefact named consistently as `UD-Q4_K_XL`; `Q4_K_M` is the draft.
- `T_thres ≈ 94` → `ceil(94.36) = 95`, restated as an expected-coverage
  heuristic; `k_e = 8` and `γ` replace the overloaded `K`.
- Qwen3.5-122B-A10B has the **same** 256 experts and top-8 routing, so the
  "larger active footprint gives a lower `T_thres`" sentence is deleted.
- PR #20075 reattributed to **eauchs**; it is an external-draft-model plus SSM
  checkpoint/restore fix, not ngram-mod.
- Upstream statuses re-checked: #20075 **closed unmerged** 2026-04-25 (was
  listed OPEN); #22105 **merged** 2026-06-28 (v3 said open draft).
- Licence conflict resolved: MIT for code and docs, CC0-1.0 for data with
  `DATA_LICENSE` and `LICENSES/CC0-1.0.txt` present. v3's "Apache 2.0" removed.

### Found during the audit

- **v1 measured truncated thinking, not answers.** 144 of 190 v1 requests
  (75.8 %) returned an empty `message.content`; `reasoning` and `code_small`
  are 19/19. The 300-token cap was reached inside the thinking block and
  `reasoning_content` was never captured.
- **Root cause of the vocabulary failure.** `Qwen/Qwen3.5-0.8B` has no
  `generation_config.json` upstream (HTTP 404), so its GGUF carries no
  `tokenizer.ggml.bos_token_id` and llama.cpp substitutes the hard-coded GPT-2
  legacy default `11` against the target's `248044`. Both models declare
  `add_bos_token = false`, so the field that gates speculation is one neither
  model uses. `--override-kv tokenizer.ggml.bos_token_id=int:248044` flips
  `vocab_cmpt` from 0 to 1.
- **The fix does not explain the slowdown.** A same-binary A/B on 2026-08-25
  moved `long_explain` from 48.4 tok/s to 50–51 tok/s, against a baseline of
  ~126. The negative finding survives, now measured on a matched-vocabulary
  path.
- **`llama-server` plus a draft model aborts at `bcb5eeb64`** with
  `CUDA error: an unsupported value or parameter` in
  `ggml_cuda_op_mul_mat_cublas`, reproducibly, immediately after a
  partial-accept checkpoint restore. 3/3 on `code_small`, in both the
  translation and matched arms.
- **The committed v2 script does not match the committed v2 data** — different
  config directory names, and no v2/v3 log records its own argv.
- **`--spec-type` is server-only**, not "missing from master" as v3 claimed.

### Added

- [`ERRATA.md`](ERRATA.md): every corrected claim with its evidence.
- [`RETEST_TODO.md`](RETEST_TODO.md): the runnable work queue that closes the
  open items, with the bench host's verified state.
- [`analysis/verbose_accounting.py`](analysis/verbose_accounting.py):
  reconstructs the acceptance-counter artefact from any `-v` log.
- [`bench/retest_runner.py`](bench/retest_runner.py): one pinned binary,
  ABBA ordering, N repeats, hashed manifest, and per-request capture of text,
  reasoning channel, stop reason, timings, and token IDs via `logprobs` (near-complete: `probs_output` drops trailing stop-word tokens, `server-context.cpp:2036-2039`, so the list can run a few short of `predicted_n`, which stays the authority for token counts).
  Thinking suppression is verified per request, not assumed.
- `analysis/plot_acceptance_accounting.png`, replacing the retracted scatter.
- `SHA256SUMS`, `CITATION.cff`, `DATA_LICENSE`, `LICENSES/CC0-1.0.txt`.
- Errata headers on every historical script; all host-specific paths are now
  environment variables.

### What survives

Under the exact conditions archived here, no tested condition that recorded
speculative activity beat its matched no-speculation reference in aggregate.
That statement is narrower than what was published, and it is what the data
support.

## Sibling-repo notice — vllm-2x3090 v5.0 (2026-05-17)

A natural follow-up to this repo's MTP findings was published in the sibling [`qwen3.6-vllm-2x3090` v5.0](https://github.com/thc1006/qwen3.6-vllm-2x3090/releases/tag/v5.0): a same-hardware A/B between the production MoE Qwen3.6-35B-A3B-AWQ + MTP k=3 + TP=2 stack and the new dense sibling **Qwen3.6-27B-AWQ** on a voice-agent workload (10 prompts × 3 trials × 2 models = 60 samples).

Result: **MoE+MTP production stack wins decisively** — TTFT 178 ms vs Dense 771 ms (**4.34×**), tok/s 88 vs 16 (**5.42×**), e2e 274 ms vs 1684 ms (**6.13×**). The "Dense 27B fits TP=1, should be cheaper to serve" intuition is falsified on this hardware × workload, and the MTP k=3 production recommendation from this repo's v3.0 is corroborated by the absence of any cheaper Dense-no-spec alternative.

Caveat: see the [vllm-2x3090 v5 README scope section](https://github.com/thc1006/qwen3.6-vllm-2x3090/blob/master/v5_2026_05_17/README.md#scope-and-known-caveats), N=3, single hardware, plus several vLLM-config and prompt-specification confounds documented in the ERRATA. **Latency findings are compute-bound and robust; tool-accuracy findings need a prompt-matched retest.** This repo (`qwen3.6-speculative-decoding-rtx3090`) does not get a version bump or new release for this event, the work belongs in the sibling repo's lineage.

## [v3.0] — 2026-05-07

### Added

- **DFlash speculative-decoding bench** via llama.cpp PR #22105 on the same
  hardware as v2.x. Full content in [`v3_dflash_2026_05_07/`](v3_dflash_2026_05_07/).
  - Setup: same RTX 3090 (single card) + `Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf` target.
    Drafter is `z-lab/Qwen3.6-35B-A3B-DFlash` (HF safetensors), converted to
    GGUF via PR #22105's modified `convert_hf_to_gguf.py` with `--target-model-dir`.
  - 5 prompts x 1 trial x 3 draft-max configs (4, 8, 16) = 15 measurements.
  - Result: **NET LOSS -44.6 % at best (DFlash --draft-max=8: 77.0 tok/s vs
    138.9 tok/s no-spec baseline)**. Slightly less bad than v2.x's Oleg
    draft-spec NET LOSS (-52 %), but still net negative.
  - ~~First public RTX 3090 + DFlash + Q4 quantized target datapoint.~~
    **RETRACTED 2026-08-25** — no novelty search was performed (ERRATA F3).
  - Reproduction: [`v3_dflash_2026_05_07/bench/bench_dflash.sh`](v3_dflash_2026_05_07/bench/bench_dflash.sh)
    (note: do NOT use `set -euo pipefail` + `grep | tail` combo; empty grep
    matches kill the whole script via pipefail).

### Cross-method ranking (single 3090, Qwen3.6-35B-A3B Q4_K_XL target)

| method | tok/s (mean) | vs baseline |
|---|---:|---:|
| no spec (baseline) | 138.9 | reference |
| Oleg draft-spec max=32 | 65.5 | -52.8 % |
| Oleg draft-spec max=16 | 66.6 | -52.1 % |
| DFlash --draft-max 16 | 65.8 | -52.6 % |
| **DFlash --draft-max 8** | **77.0** | **-44.6 %** (best DFlash) |
| DFlash --draft-max 4 | 74.9 | -46.1 % |

### Mechanism note

Per llama.cpp PR #22105 author: *"for Qwen3.5/3.6 MoE, performance is currently
not optimal due to MoE + hybrid structure not well supported."* The wider
mechanism is **MoE expert-routing x consumer-Ampere bandwidth**:
Qwen3.6-35B-A3B routes 8-of-256 experts per token, expert-saturation threshold
~94 tokens. At single-stream batch=1 with `--draft-max ≤ 32`, drafted tokens
stay below saturation, so verification has to load the union of expert slices,
exceeding savings even at 100 % acceptance. This is **not Q4-specific or
consumer-GPU-specific in isolation** — it's the joint MoE-routing × bandwidth ×
Q4-coupling effect. Cross-reference: the [sister
repo](https://github.com/thc1006/qwen3.6-vllm-2x3090) v3.0/v4.0 shows vLLM MTP
(co-trained head, no separate KV cache) is +27 % NET WIN on the same hardware.


## [v2.3] — 2026-04-26 (afternoon)

### Added
- [`v2_3090_followup/exp2_codejson_n3/`](v2_3090_followup/exp2_codejson_n3/):
  workload-shape probe. 5 code/JSON prompts (Python class with RLock, REST API
  spec JSON, Rust merge_sort, Nginx reverse proxy config, PostgreSQL top-10
  customers query) × 3 trials × 3 configs (baseline, Oleg `--draft-min 2
  --draft-max 32`, srogmann `--draft-min 48 --draft-max 64`) on standalone
  single 3090. **Result confirms v2 negative direction**: baseline 139.22 ±
  0.46 tok/s, Oleg 66.57 ± 7.57 tok/s (−52 %), srogmann 83.84 ± 1.80 tok/s (−40
  %). The "structured prompts win" hypothesis is refuted on llama.cpp + Q4 +
  RTX 3090.
- README banner at the top documenting Exp 2 + the cross-engine status
  correction.

### Changed
- **MAJOR scope correction** — earlier wording (v2.2) claimed the spec-decode
  negative finding is "hardware-class-independent" across consumer Ampere +
  datacenter Ampere NVLink + Hopper. A v3 clean A/B retest in the sibling repo [`thc1006/qwen3.6-vllm-2x3090`](https://github.com/thc1006/qwen3.6-vllm-2x3090)
  (matched flags + `--no-enable-prefix-caching`) found vLLM MTP k=1 on the
  **same 2× RTX 3090 PCIe hardware** is **+27.5 % faster decode rate**, not −12
  %. So the negative direction is **engine + spec-method specific**: it holds
  for `llama.cpp` draft-then-verify (this repo's data) but does not hold for
  `vLLM` MTP k=1 with prefix-cache disabled. README "Cross-engine confirmation"
  paragraph and the "Why" paragraph have been corrected.
- README banner updated; `v2_3090_followup` directory expanded.

## [v2.2] — 2026-04-26 · documented, never tagged

> This section is dated like a release and there is no `v2.2` tag and no
> GitHub release for it; `git tag` goes v2.1, v2.3. It was found on
> 2026-08-30 by a check written for v4.0 and v4.1, which are untagged for
> a different and deliberate reason (the tag is cut at merge). Cite v2.1
> or v2.3 by tag, or this section by commit.

### Changed
- README cross-engine paragraph: disclose 3090 vLLM published config
  confound (baseline `--gpu-memory-utilization 0.90 --max-num-seqs 8`
  vs MTP run `0.80 / 2`); cite **2× A100-80GB SXM4 NVLink** Modal clean
  A/B as the canonical cross-hardware MTP datapoint (prompt-4
  decode-only delta **−11.4 %**, TTFT-robust).
- README "Why" paragraph: remove "memory-bandwidth-bound" attribution. The
  mechanism is **bandwidth-independent** — the same negative direction is now
  observed on (a) consumer 3090 GDDR6X 936 GB/s, (b) datacenter A100 SXM4 HBM2e
  2 TB/s + NVLink, (c) Hopper H20-3e per [vllm #38182](https://github.com/vllm-project/vllm/issues/38182).
- Replace "engine-independent on Ampere" with "**hardware-class-
  independent at single-stream batch=1** across consumer Ampere +
  datacenter Ampere (and Hopper per #38182)".

### Added
- README image hot-link to cross-hardware comparison chart hosted in
  the sibling `qwen3.6-vllm-2x3090` repo (`analysis/plot_cross_hardware.png`).
- N=3 trial replication on a fresh standalone 3090 host of the v2 numbers
  (baseline 139.19, Oleg `--draft-min 2 --draft-max 32` 65.24, srogmann
  `--draft-min 48 --draft-max 64` 85.50). Run-to-run stdev <0.11 tok/s,
  matching v2 published 139.9/65.0/85.6 within <0.5 pp. Addresses the
  N=1 caveat in v2 limitations.

## [v2.1] — 2026-04-25

### Added
- README **Validation timeline (post-publication)** section consolidating six
  independent corroborations and one academic theoretical framing that have
  appeared since v1 / v2:
  - [MoE-Spec (arXiv 2602.16052)](https://arxiv.org/abs/2602.16052) names
    the phenomenon ("expert budgeting") and proposes a training-free
    verification-time budget cap.
  - [Alloc-MoE (arXiv 2604.08133)](https://arxiv.org/abs/2604.08133) and
    [XShare (arXiv 2602.07265)](https://arxiv.org/pdf/2602.07265) frame
    the same expert-saturation pressure.
  - [vllm #35387](https://github.com/vllm-project/vllm/issues/35387), H100 +
    FP8 + Qwen3-Next-80B-A3B with `method=qwen3_next_mtp` reports −76.5 %
    latency regression (different hardware/quant/arch from this bench,
    suspected `mamba_postprocess` CPU sync; same direction).
  - [vllm #38182](https://github.com/vllm-project/vllm/issues/38182), H20-3e +
    Qwen3.5-35B-A3B + MTP drops prefix-cache hit rate ≈92 % → ≈71 %; @Angazenn
    pinpoints the cause to `single_type_kv_cache_manager.py:L457`.
  - [vLLM Qwen3.5/3.6 official Recipes](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html)
    now state up-front that "MTP-1 reduces per-token latency but degrades
    text throughput under high concurrency".
- README cross-engine confirmation note in TL;DR + Related reading pointer
  to sibling repo
  [`thc1006/qwen3.6-vllm-2x3090`](https://github.com/thc1006/qwen3.6-vllm-2x3090).
- README applicability note (iv): batched multi-user serving caveat, and (v):
  explicit scope: this bench tests `ngram-cache`, `ngram-mod`, classic
  `--model-draft` in llama.cpp and `mtp k=1` in vLLM. **EAGLE-3** with CUDA
  graphs (vLLM Model Runner V2) is not evaluated here.
- README counter-example block: corrected attribution for the +15–45 %
  Qwen3.5-122B-A10B speedup on PR #20075; that data is from the PR author's M3
  Max bench plus @0xSero's AMD Strix Halo follow-up, not srogmann's bench.
  Strix Halo also reports up to **+119 %** with the REAP-pruned variant on
  gfx1151.

### Changed
- README applicability note (ii): `[PR #20075](https://github.com/ggml-org/llama.cpp/pull/20075)`
  was open at v1 publication; on 2026-04-25 a community comment suggested
  it can be closed because its functionality is superseded elsewhere. The
  note now reflects that the hybrid-SSM/MoE checkpoint situation is fluid.

### Older Unreleased entries (carried over from earlier in 2026-04-22 → 2026-04-25)
- `v2_3090_followup/results_v2.json`: machine-readable summary of all
  v2 runs (per-prompt `llama-cli` stats + per-config mean / min / max /
  std), extracted from the `.log` artefacts.
- `v2_3090_followup/extract_results.py`: the extraction script used to
  produce the above JSON.
- `v2_3090_followup/plot_v2.py` + `plot_v2_configs.png`: horizontal
  bar chart comparing all 9 original-commit configs against baseline,
  with master-commit cross-check annotation.
- `CHANGELOG.md`: this file.

## [v2.0] — 2026-04-22

Follow-up bench addressing [Oleg-dM's comment](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF/discussions/14)
on the HF-discussion thread for `unsloth/Qwen3.6-35B-A3B-GGUF`.

### Added
- `v2_3090_followup/`: fresh single-3090 bench on a different
  physical 3090 covering:
  - baseline (no spec-decode)
  - `--draft-min 2 --draft-max 32` (Oleg's suggestion)
  - `--draft-min 2 --draft-max 16` and `--draft-max 64` (bracketing sweep)
  - default `--draft-min=5` variants (`--draft-max 8/16/32` + bare `-md`)
  - srogmann-style `--draft-min 48 --draft-max 64`
  - `--verbose` per-token acceptance dump
- **Cross-check on current master** `bcb5eeb64`
  (post PR #22227 `speculative-simple: add checkpoint support`), 3 key configs
  re-run on master hardware to rule out stale-commit artefact.
- 45 per-prompt `llama-cli` logs + `verbose.log`.
- `v2_3090_followup/SUMMARY.md` with methodology + full result table.
- `v2_3090_followup/bench_3090_oleg.sh` reproducible script.
- v2 section appended to `BENCHMARK_ENV.md` documenting the v2
  hardware, toolchain, and both commits tested.
- `.gitignore` negation for `v2_3090_followup/**/*.log` so bench
  evidence survives `git clean` and `git add -A` round-trips.

### Changed
- `README.md`: UPDATE banner summarising v2 findings and linking to
  the follow-up artefacts.
- `pr_comment.md`: UPDATE note so the historical llama.cpp PR-comment
  draft stays consistent with the repo state.

### Key findings
- Oleg's `--draft-min 2 --draft-max 32` beats the `--draft-min=5`
  defaults by +18 % (65.0 vs 55.3 tok/s) but is still −54 % vs
  baseline 139.9 tok/s.
- Aggressive `--draft-min 48 --draft-max 64` is the **least bad**
  recipe at 85.6 tok/s (−39 %): counter-intuitively, the "wasteful"
  large-window config amortises verify + KV-management overhead better than
  tight windows.
- ~~100 % draft acceptance is genuine: source read of
  `common/speculative.cpp` (`impl->n_acc_tokens += n_accepted;` in
  `common_speculative_accept()`) + `--verbose` run emitting
  `draft acceptance rate = 1.00000 (115 accepted / 115 generated)`.~~
  **Corrected 2026-08-25:** ERRATA A1 quotes this sentence as the claim it
  refutes. The ratio is 1.0 by construction on this model; the server counter
  counts accepted against accepted.
- Master cross-check gives identical numbers within ±0.3 % noise, so
  the regression is architectural rather than a stale-commit artefact.

### Conclusion of v1 stands
~~On a consumer RTX 3090 with Qwen3.6-35B-A3B at Q4_K_M, **no speculative
decoding configuration is a net win** — regardless of commit, regardless of
draft-min / draft-max, regardless of which measurement regime. H100 / H200 or
NVLinked pairs may flip the sign.~~ **Corrected 2026-08-26:** the v4 audit
falsifies this on the same card and model. DFlash and the target's own
multi-token-prediction head are net wins; v1 measured three methods and none of
them was either.

## [v1.0] — 2026-04-21

Initial public release of the spec-decode benchmark matrix.

### Added
- 19-configuration bench matrix on a single RTX 3090 (of the two on s1)
  via `llama-server` at commit `9789512` (post PR #19493 merge).
- `Qwen3.6-35B-A3B-UD-Q4_K_XL` main + `Qwen3.5-0.8B-Q4_K_M` draft
  (~~vocab-matched~~, 248320 vocab size). **Corrected 2026-08-25:** the vocab
  *sizes* match, but llama.cpp rejects the pair on its special-token gate and
  ran every classic-draft measurement through the token-translation fallback.
  See [`ERRATA.md`](ERRATA.md) A2.
- llama-server-based Python bench runner (`bench_runner.py`) plus
  three shell driver scripts (`run_matrix.sh`, `run_p0_matrix.sh`,
  `run_verify_matrix.sh`).
- Analysis plots: bar chart, per-prompt heatmap, accept-vs-speed
  scatter (in `analysis/`).
- Per-config JSON results in `results/` + `results/verify/`.
- `pr_comment.md` draft for llama.cpp PR #19493 / Issue #20039.
- `BENCHMARK_ENV.md` environment snapshot (s1, 2× RTX 3090,
  i7-11700, Ubuntu 24.04).
- `collect_env.sh` helper to regenerate the env snapshot.

### Key finding
No speculative-decode configuration achieves a net speedup over the
non-speculative baseline of 135.7 tok/s. Mean decode drops 3–12 %
across ngram-cache, ngram-mod, and classic draft-model variants, with
a bimodal tail reaching 59–67 tok/s on reasoning / code prompts
~~despite 100 % draft acceptance. Interpretation aligned with
MoESD (arXiv 2505.19645) and Utility-Driven SD (arXiv 2506.20675):
for a 3B-active MoE, draft batch K stays below the expert-saturation
threshold (~94 tokens for this sparsity), so each drafted token pulls
new experts through the memory hierarchy and verification pays for
the union.~~ **Corrected 2026-08-25:** the 100 % figure is a counter
artefact and not a measurement (ERRATA A1), and the expert-union
mechanism built on it was never supported by evidence collected here
(ERRATA A7, A9). The slowdown is accounted for without any
MoE-specific pathology.

[v4.1]: https://github.com/thc1006/qwen3.6-speculative-decoding-rtx3090/releases/tag/v4.1
[v4.0]: https://github.com/thc1006/qwen3.6-speculative-decoding-rtx3090/releases/tag/v4.1
[v3.0]: https://github.com/thc1006/qwen3.6-speculative-decoding-rtx3090/releases/tag/v3.0
[v2.3]: https://github.com/thc1006/qwen3.6-speculative-decoding-rtx3090/releases/tag/v2.3
[v2.2]: https://github.com/thc1006/qwen3.6-speculative-decoding-rtx3090/releases/tag/v2.3
[v2.1]: https://github.com/thc1006/qwen3.6-speculative-decoding-rtx3090/releases/tag/v2.1
[v2.0]: https://github.com/thc1006/qwen3.6-speculative-decoding-rtx3090/releases/tag/v2.0
[v1.0]: https://github.com/thc1006/qwen3.6-speculative-decoding-rtx3090/releases/tag/v1.0
