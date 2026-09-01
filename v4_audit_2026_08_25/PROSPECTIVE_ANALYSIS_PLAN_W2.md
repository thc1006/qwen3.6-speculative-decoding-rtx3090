# Prospective analysis plan, run W2

Written and committed **before run W2 was started**, which is the difference
between this file and the one beside it: `PROSPECTIVE_ANALYSIS_PLAN_W.md` was
finalised at 360 of 500 arm-runs, and says so. Git ancestry can show that this
commit precedes the commit that adds W2's data, and this time it also precedes
the run: the driver had not been invoked when this was written.

## What is being asked

Run W left one thing open, and its own *Not closed* entry says what it is:

> no detectable predecessor-mode association, with the widest matched interval
> spanning [-2.97, +0.86] %. That is a bound, not an answer: an effect of the
> size that would matter sits inside it. What is still missing is the power to
> exclude one, which is more sessions rather than a different design.

The size that would matter is **2.4 pp**, the hard-cap mode effect the
predecessor was offered as an explanation for. `-2.4` lies inside the run W
interval, so run W cannot exclude it.

## Design, unchanged from W

The same 10 × 10 Williams square, randomised row order, one request at a time,
thinking off, 300-token cap, the same ten arms, `BENCH_EXPECT_COMMIT`
`3737e4137`, on the same card. Only the number of sessions changes.

## Sessions: twelve, fixed in advance

Twelve, not five, and not "until the interval is narrow enough". From run W's
matched interval the implied session-level SD is 1.543 pp, so with the Student
t multiplier at eleven degrees of freedom the expected half-width is

    t(0.975, 11) * 1.543 / sqrt(12) = 2.201 * 1.543 / 3.464 = 0.98 pp

which puts the expected interval at about [-2.04, -0.07] and excludes -2.4 with
margin. Eight sessions would clear it on paper, at [-2.34, +0.23], and that is
too close to survive the point estimate moving at all.

**Twelve is the number, and it is the number whatever the answer looks like.**
Stopping early because the interval already excludes -2.4, or adding sessions
because it does not, is optional stopping and would invalidate the interval it
was meant to produce. If the run is interrupted, the completed sessions are
analysed and reported as *k of 12*, never as *k*.

## Why a new invocation rather than five more sessions

Run W's five sessions are one invocation. Appending sessions later would pool
across invocations, and the difference between invocations is exactly what A16
is about and what nothing here explains. W2 is a separate invocation with its
own label, so no analyser can pool the two by accident: the driver's output
prefix is `BENCH_RUN_LABEL`, hardcoded `W` until 2026-08-30, and every analyser
globs `matrix_W_*` with no invocation qualifier.

## The estimand, and how it is computed

The primary quantity is the **matched capped-vs-free predecessor contrast**,
`carryover.py`'s `capped_contrast_matched`: for a current arm `X`, each of the
other four configurations `Y` contributes the paired difference between the
rate after `Y-cap` and the rate after `Y`, with `X`'s own twin dropped, and the
session is the resampling unit. Row-boundary adjacencies are excluded, because
the Williams square balances within rows only.

Reported alongside, as sensitivity and not as the answer:

- the grouped contrast run W published, so the two are comparable;
- the same estimate including row boundaries;
- the per-session values, so the between-session spread is visible.

## What each outcome means

- **Interval excludes -2.4.** The predecessor cannot account for the hard-cap
  mode effect at this power. That is what the run is for.
- **Interval still contains -2.4.** The bound is tighter and the question is
  still open. It is reported as that, and the number of sessions that would
  close it is computed from W2's own SD rather than W's.
- **Interval excludes zero.** A predecessor-mode association exists and the
  W-versus-V2 gap needs re-examining. This is not the expected outcome and is
  named here so it cannot be treated as noise afterwards.

No outcome licenses dropping W2 and keeping run W.

## What this plan does not claim

That no outcome summary will be looked at while the run is going. The driver
prints per-session progress and the operator will see it. What ancestry can
show is that this file, its session count and its estimand were fixed before
the first arm-run existed; nothing more is asserted.
