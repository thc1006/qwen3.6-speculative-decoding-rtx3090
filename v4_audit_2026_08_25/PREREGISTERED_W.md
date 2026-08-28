# Pre-registered analysis plan for run W

Committed **before** run W's data existed. `git log` on this file is the
timestamp; W's arm-runs land in a later commit, and
`analysis/verify_claims.py` asserts that ordering.

## Why this file exists at all

Run W was designed to settle a disagreement this repository has already been
wrong about twice. `spec-dflash-n2`'s hard-cap shift reads **+5.92 pp** in run
V2's crossover and **+8.65 pp** in run V3's within-invocation square, and the
fourth review's P0-4 was that neither design can attribute the difference,
because both are cyclic rotations that balance treatment *position* and leave
first-order carryover alone. Measured from V3's own data: `baseline` is
preceded by one and the same arm in **9 of 9** within-repeat adjacencies.

An analysis chosen after seeing which answer W gives is not evidence about
which design was right. So the questions, the estimators and the thresholds are
written down here first, and `analysis/carryover.py` and
`analysis/length_mode.py` already exist, are tested, and will not be edited to
suit the result.

## What W is

`bench/run_w_williams.sh`: five sessions, each a 10 × 10 Williams square over
the same ten arms as V3 — five configurations in free and hard-capped form —
with the row order shuffled from a per-session seed recorded in the manifest.
Every arm visits every position exactly once **and** is preceded by every other
arm exactly once within a repeat. It is run V3 verbatim except for
`BENCH_ORDER`; the exported treatment variables were diffed against
`run_v3_within.sh` and differ in three places, all of them the schedule.

## The three questions, and what each answer would mean

**Q1. Does the mode effect survive a carryover-balanced schedule?**

Estimator: the same `shift_pp` A17 publishes — an absolute change in
percentage points of the arm-versus-baseline figure — with the session as the
resampling unit and a two-sided 95 % Student-t interval over the five sessions.
The log contrast is reported beside it as a sensitivity, as it now is for V2
and V3.

- If W's interval for `spec-dflash-n4` contains V2's +12.03 and V3's +12.17,
  that arm's result is robust to the schedule and the sign flip stands.
- If W's `spec-dflash-n2` lands inside V2's [+4.86, +6.99], the crossover was
  measuring the effect and V3's +8.65 carried carryover.
- If it lands near +8.65, the within-invocation reading was right and V2's
  between-invocation design was diluting it.
- If it lands outside both, neither previous design measured it and this
  repository will say so rather than pick the nearest.

**Q2. Is `spec-dflash-n2` sensitive to what ran before it?**

This is the question A17 says "points the same way each time" and cannot test,
and the question RETEST_TODO opened for A16. Estimator: for each arm, the mean
decode rate after a capped predecessor minus the mean after a free one, using
**within-repeat adjacencies only** — the ones the square balances — with a
t interval over the five sessions.

The design gives each uncapped arm exactly five capped and four free
predecessors, and each capped arm the reverse. `analysis/carryover.py` refuses
to report this for any schedule where that split is not exact, and refuses V3
and O2 today for exactly that reason.

- An interval excluding zero for `spec-dflash-n2` and containing zero for the
  other arms would make first-order carryover a live explanation for the
  V2/V3 disagreement, and A16's next experiment would be about state that
  survives a server restart.
- Intervals containing zero for every arm would eliminate the predecessor as
  the explanation, which is a real result: it is the one candidate this
  repository has been able to name and could never test.

**Q3. Does the within-invocation instability A16 reports appear again?**

Descriptive, not inferential: the per-repeat coefficient of variation for each
arm inside each session, as reported for V3 (0.27 % for no speculation against
1.82 % for `spec-dflash-n2`) and for T4 (a 3.9 % step partway through six
repeats). W has ten repeats a session and five sessions, which is the largest
dataset this repository has for it.

## What will not be done

- No arm will be dropped, and no session will be dropped, except by the rules
  the analysers already enforce: a run without `RUN_COMPLETE.json` is refused,
  a crashed arm-run does not enter a pooled rate, and a run whose order cannot
  be recovered from `t_start` is refused whole.
- The estimand will not be switched after seeing which one is tidier. `pp` is
  primary because it is what A17's tables and `length_matching.py` publish; the
  log contrast is reported for every arm whether or not it flatters the result.
- If W disagrees with both V2 and V3, that will be published as a three-way
  disagreement. This repository has said the wrong thing about
  `spec-dflash-n2` twice and the correct response to a third reading is not to
  choose a favourite.
- No claim will be made that W identifies the A16 mechanism. It balances one
  nuisance factor. Thermal state, driver state, allocator history and the
  GDDR6X memory-junction temperature that NVML does not expose on Linux are all
  still unmeasured, and A16 will keep saying so.

## The one thing W cannot do

Five sessions is four degrees of freedom, so `t = 2.776` and the intervals will
be wider than V2's eight-session ones. A null result on Q2 is therefore weaker
evidence than a positive one, and will be reported as "no detectable
predecessor effect at this power", with the interval, not as "there is none".
