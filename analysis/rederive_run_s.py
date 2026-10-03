#!/usr/bin/env python3
"""Derive plan S's two contrasts from the arm-runs, with the estimator it pledged.

    python3 analysis/rederive_run_s.py            # derive and print
    python3 analysis/rederive_run_s.py --json     # the same, as JSON

WRITTEN BEFORE THE DATA. The run was launched at 2026-10-03T19:19:47+08:00 and
this file was committed while it was still in its third block, so nothing in it
could have been chosen to suit a number. That is the only reason a
pre-registration is worth writing: the analysis has to be fixed before the
result, or the bands below are a description of what happened rather than a test
of what was predicted.

The estimator is RUN Y'S, reused rather than reimplemented. `paired_blocks`
holds it: within-block log ratios, their mean, a Student t interval on the mean,
and the point back-transformed with `exp`. Run Y's own table was computed by hand
and three of its columns were a mean log ratio printed as a percentage, which is
why `analysis/rederive_run_y.py` exists and why this file imports the same
functions instead of writing the arithmetic again.

The two contrasts, in the direction the plan states them:

    packed against distinct      log(packed / distinct)      what packing costs
    nopoll against packed        log(nopoll / packed)        what --poll 0 recovers

The bands are the ones
`v4_audit_2026_08_25/PROSPECTIVE_PLAN_SIBLINGS.md` pre-registered for the
primary contrast, and this prints which one the interval lands in without
deciding anything else:

    inside +-1 %        hyperthread sharing is not the mechanism
    1 % to 6 %          the size a yielding spin predicts; the candidate stands
    above 10 %          too large for a yielding spin; something takes the core

What it refuses rather than reports:

  * a block count other than twelve, for either contrast. The interval's degrees
    of freedom are published, so a different count is a different number.
  * an arm missing from a block, which `load_blocks` leaves out and this counts.
  * a run directory holding an arm this plan does not name, which would mean the
    directory is not plan S's.
  * more than one run directory matching, because then the figures would depend
    on which one sorted first.
"""
from __future__ import annotations

import json
import math
import pathlib
import statistics as st
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from paired_blocks import (                                      # noqa: E402
    load_blocks, observed_schedule, t_critical_975)

ROOT = pathlib.Path(__file__).resolve().parents[1]
DISTINCT = "spec-dflash-n2"
PACKED = "spec-dflash-n2-packed"
NOPOLL = "spec-dflash-n2-nopoll-packed"
ARMS = (DISTINCT, PACKED, NOPOLL)
CONTRASTS = ((DISTINCT, PACKED), (PACKED, NOPOLL))
BLOCKS = 12
MINUS = "−"          # U+2212, which is what the documents carry

# The pre-registered reading of the primary contrast, as (low, high, verdict).
# Written from the plan, not from the data.
BANDS = (
    (-1.0, 1.0, "inside the null band: hyperthread sharing is not the mechanism"),
    (1.0, 6.0, "the size a yielding spin predicts: the candidate stands"),
    (6.0, 10.0, "between the bands the plan names, which it did not reserve"),
    (10.0, float("inf"), "too large for a yielding spin: something takes the core"),
)


def _run_dir() -> pathlib.Path:
    """Globbed, never written down. A hard-coded run directory is a figure that
    silently keeps describing the wrong run after the next one lands."""
    found = sorted(ROOT.glob("v*/data/matrix_S_siblings_*"))
    if not found:
        raise SystemExit(
            "no v*/data/matrix_S_siblings_* directory; plan S's run is not "
            "committed, so there is nothing to derive")
    if len(found) > 1:
        raise SystemExit(
            f"{len(found)} plan S run directories: "
            f"{', '.join(p.name for p in found)}. Which one the figures describe "
            f"would depend on which sorted first, so this refuses rather than "
            f"choosing.")
    return found[0]


def _pct(v: float) -> str:
    return f"{MINUS if v < 0 else ''}{abs(v):.2f} %"


def verdict(change_pct: float) -> str:
    mag = abs(change_pct)
    for lo, hi, text in BANDS:
        if lo <= mag < hi or (lo < 0 and mag < hi):
            return text
    return "outside every band named, which is itself a finding"


def derive(run: pathlib.Path | None = None) -> dict:
    run = run or _run_dir()
    blocks = load_blocks(run)
    seen = {a for v in blocks.values() for a in v}
    extra = sorted(seen - set(ARMS))
    if extra:
        raise SystemExit(
            f"{run.name} holds arms this plan does not name: {extra}. That is "
            f"not plan S's run directory.")
    missing = sorted(set(ARMS) - seen)
    if missing:
        raise SystemExit(f"{run.name} has no arm-runs for {missing}")

    rows = []
    for base, other in CONTRASTS:
        have = sorted(b for b, v in blocks.items() if base in v and other in v)
        if len(have) != BLOCKS:
            raise SystemExit(
                f"{other} against {base}: {len(have)} complete blocks, not "
                f"{BLOCKS}. The interval's degrees of freedom are published, so "
                f"a different count is a different number, not a near miss.")
        b = [blocks[k][base] for k in have]
        o = [blocks[k][other] for k in have]
        lr = [math.log(o[i] / b[i]) for i in range(len(have))]
        m = st.fmean(lr)
        se = st.stdev(lr) / math.sqrt(len(lr))
        df = len(lr) - 1
        half = t_critical_975(df) * se
        rows.append({
            "base": base, "other": other,
            "base_mean": st.fmean(b), "other_mean": st.fmean(o),
            "change_pct": (math.exp(m) - 1) * 100,
            "ci_lo_pct": (math.exp(m - half) - 1) * 100,
            "ci_hi_pct": (math.exp(m + half) - 1) * 100,
            "blocks": len(lr), "df": df, "mean_log_ratio": m,
            "sd_log_ratio": st.stdev(lr),
        })

    schedule, n_pos = observed_schedule(run)
    return {"run": run.name, "rows": rows,
            "schedule": {a: sorted(v) for a, v in schedule.items()},
            "positions": n_pos}


def main() -> int:
    as_json = "--json" in sys.argv[1:]
    for a in sys.argv[1:]:
        if a != "--json":
            print(f"unknown argument {a!r}; the only one is --json",
                  file=sys.stderr)
            return 2
    out = derive()
    if as_json:
        print(json.dumps(out, indent=2))
        return 0

    print(f"plan S, {out['run']}")
    print(f"  {'contrast':34s} {'base':>9} {'other':>9} {'change':>9} "
          f"{'95 % interval':>22}")
    for r in out["rows"]:
        name = f"{r['other'].replace('spec-dflash-n2', '')or 'distinct'} "\
               f"vs {r['base'].replace('spec-dflash-n2', '') or 'distinct'}"
        print(f"  {name:34s} {r['base_mean']:9.2f} {r['other_mean']:9.2f} "
              f"{_pct(r['change_pct']):>9} "
              f"[{_pct(r['ci_lo_pct'])}, {_pct(r['ci_hi_pct'])}]".rjust(22))
    primary = out["rows"][0]
    print(f"\n  primary contrast, packed against distinct: "
          f"{_pct(primary['change_pct'])}")
    print(f"  pre-registered reading: {verdict(primary['change_pct'])}")
    print(f"  {primary['blocks']} blocks, {primary['df']} degrees of freedom, "
          f"log-ratio sd {primary['sd_log_ratio']:.5f}")
    # The design's own control: every arm must have occupied every position
    # across the blocks, or position is confounded with arm.
    print("\n  observed positions within a block, rebuilt from the data:")
    for arm, pos in sorted(out["schedule"].items()):
        print(f"    {arm:34s} {pos}")
    if any(len(p) < 2 for p in out["schedule"].values()):
        print("  NOTE: an arm held only one position, so position within a block "
              "is confounded with the arm and the mirrored order did not happen.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
