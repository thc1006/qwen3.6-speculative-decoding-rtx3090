"""Paired block analysis of a Latin-square matrix.

Every earlier table in this repository reported a delta between two pooled
figures, each averaged over repeats, with a "run-to-run SD" beside it computed
from repeats inside one run. An external review pointed out that this is not an
estimator precision: the repeats share a position in the arm order, and
ERRATA A14 shows one arm whose between-run spread was 8.5 pp against a
within-run SD of 0.53.

This does it properly for a design where each repeat is a block and every arm
appears once per block:

  ratio      pooled(arm, block) / pooled(baseline, block), so the baseline used
             is the one measured inside the same block, not a grand mean
  point      geometric mean of those ratios, i.e. exp(mean(log ratio))
  interval   percentile bootstrap over BLOCKS, resampling whole blocks with
             replacement. The block is the unit of resampling because it is
             the unit of replication: arms are assigned to positions by a fixed
             cyclic rotation, not at random, so "unit of randomisation" is the
             wrong name for it and this said so until 2026-08-26. Resampling
             requests or arm-runs instead would understate the interval in
             exactly the way A14 documents
  t interval reported alongside as a check that the bootstrap is not doing
             something strange on nine blocks

Run: python analysis/paired_blocks.py <run-dir> [--baseline ARM] [--iters N]
"""
from __future__ import annotations

import glob
import json
import math
import random
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path


def pooled(rows: list[dict]) -> float:
    n = sum(x["predicted_n"] for x in rows)
    ms = sum(x["predicted_ms"] for x in rows)
    return 1000 * n / ms if ms else float("nan")


def load_blocks(run_dir: Path) -> dict[int, dict[str, float]]:
    """block index -> {arm: pooled decode rate}"""
    blocks: dict[int, dict[str, float]] = defaultdict(dict)
    for f in sorted(glob.glob(str(run_dir / "*__rep*.json"))):
        r = json.loads(Path(f).read_text(encoding="utf-8"))
        if not r.get("rows") or r.get("crashed"):
            continue
        if r["arm"] in blocks[r["repeat"]]:
            # two files claiming the same cell: whichever sorted last would win
            # silently, and the block would look complete either way
            sys.exit(f"{Path(f).name}: {r['arm']} rep{r['repeat']} is already "
                     f"loaded. Two files claim the same cell; run "
                     f"analysis/check_data_integrity.py on this directory.")
        blocks[r["repeat"]][r["arm"]] = pooled(r["rows"])
    return dict(blocks)


def observed_schedule(run_dir: Path) -> tuple[dict[str, list[int]], int]:
    """Rebuild the order the arms actually ran in, from the data.

    `t_start` on each request is `time.perf_counter()` inside the one driver
    process, so it is monotonic across the whole run and orders the arm-runs
    within a block without needing the driver log. This is the difference
    between verifying the design and reading the label the manifest recorded:
    run T's manifest says `latin` and its arms sit at positions [1,3,2,1].
    """
    per_block: dict[int, list[tuple[float, str]]] = defaultdict(list)
    for f in glob.glob(str(run_dir / "*__rep*.json")):
        r = json.loads(Path(f).read_text(encoding="utf-8"))
        if not r.get("rows"):
            continue
        per_block[r["repeat"]].append((min(x["t_start"] for x in r["rows"]), r["arm"]))
    pos: dict[str, list[int]] = defaultdict(list)
    for rep in sorted(per_block):
        for i, (_, arm) in enumerate(sorted(per_block[rep])):
            pos[arm].append(i + 1)
    return dict(pos), len(per_block)


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta function (Lentz)."""
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        c = 1.0 + aa / c
        if abs(d) < tiny:
            d = tiny
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        c = 1.0 + aa / c
        if abs(d) < tiny:
            d = tiny
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < 3e-16:
            break
    return h


def _betainc(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
             + a * math.log(x) + b * math.log(1.0 - x))
    front = math.exp(lbeta)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def t_critical_975(df: int) -> float:
    """Two-sided 0.975 Student-t critical value, computed rather than tabulated.

    The table this replaces stopped at df=10 and fell back to 1.96, the NORMAL
    critical value, for everything above it. Run O2 has nine blocks and was
    inside the table; anything with twelve or more - and small samples routinely
    have - silently got a narrower interval than Student's t allows. No scipy:
    this file has to run in the claims job, which installs nothing.
    """
    if df < 1:
        return float("inf")
    lo, hi = 0.0, 200.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        # two-sided tail above +mid
        tail = _betainc(df / 2.0, 0.5, df / (df + mid * mid))
        if tail > 0.05:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def f_critical_95(d1: int, d2: int) -> float:
    """Upper 5 % point of F(d1, d2), by the same bisection as the t above.

    A17 compares two session-level SD estimates, one on four degrees of freedom
    and one on eleven, and says the difference is not significant. That is a
    published number like any other and a table lookup is not re-derivable, so
    it is computed here from `_betainc` rather than quoted.
    """
    if d1 < 1 or d2 < 1:
        return float("inf")
    lo, hi = 0.0, 1000.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        # P(F > mid) = I_{d2/(d2+d1*mid)}(d2/2, d1/2)
        tail = _betainc(d2 / 2.0, d1 / 2.0, d2 / (d2 + d1 * mid))
        if tail > 0.05:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def bootstrap_ci(log_ratios: list[float], iters: int, seed: int) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(log_ratios)
    means = []
    for _ in range(iters):
        means.append(st.mean(rng.choices(log_ratios, k=n)))
    means.sort()
    lo = means[int(0.025 * iters)]
    hi = means[min(iters - 1, int(0.975 * iters))]
    return lo, hi


def is_position_balanced(pos: dict) -> bool:
    """The runner's definition, verbatim: every arm visits every position the
    SAME NUMBER OF TIMES. Requiring exactly once is only the special case where
    the repeat count equals the arm count, and this file used to require it -
    so a two-arm, four-repeat schedule that the runner accepts as balanced was
    reported here as unbalanced."""
    if not pos:
        return False
    n = len(pos)
    per = len(next(iter(pos.values()))) / n
    if per != int(per) or per < 1:
        return False
    want = sorted(list(range(1, n + 1)) * int(per))
    return all(sorted(v) == want for v in pos.values())


def _parse_argv(argv: list[str]) -> tuple[list[str], dict]:
    """Accept `--opt=value` and `--opt value`. The usage line documented the
    second spelling and only the first was parsed, so `--iters 2000` silently
    ran 20000 resamples."""
    args, opts, i = [], {}, 0
    while i < len(argv):
        a = argv[i]
        if a.startswith("--"):
            if "=" in a:
                k, _, v = a.partition("=")
                opts[k] = v
            elif a in ("--allow-unbalanced",):
                opts[a] = "1"
            elif i + 1 < len(argv) and not argv[i + 1].startswith("--"):
                opts[a] = argv[i + 1]
                i += 1
            else:
                opts[a] = "1"
        else:
            args.append(a)
        i += 1
    return args, opts


def main() -> None:
    args, opts = _parse_argv(sys.argv[1:])
    if not args:
        sys.exit("usage: python analysis/paired_blocks.py <run-dir> "
                 "[--baseline ARM] [--iters N] [--allow-unbalanced]")
    run_dir = Path(args[0])
    base = opts.get("--baseline", "baseline")
    iters = int(opts.get("--iters", "20000"))
    allow_unbalanced = "--allow-unbalanced" in opts

    blocks = load_blocks(run_dir)
    if not blocks:
        sys.exit(f"no usable arm-runs in {run_dir}")

    man = {}
    if (run_dir / "manifest.json").exists():
        man = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    arms = [a for a in (man.get("arms") or {}) if a != base] or sorted(
        {a for b in blocks.values() for a in b} - {base})

    complete = [b for b, v in blocks.items() if base in v and all(a in v for a in arms)]
    print(f"{run_dir.name}")
    # verified from the arm-runs themselves, not read off the manifest
    pos, n_blocks = observed_schedule(run_dir)
    balanced = is_position_balanced(pos)
    declared = man.get("schedule_is_position_balanced")
    print(f"  ordering: {man.get('order_mode', '?')}"
          f"{' (position-balanced)' if balanced else ' (NOT position-balanced)'}"
          f"   blocks on disk: {len(blocks)}   complete blocks used: {len(complete)}")
    if declared is not None and declared != balanced:
        print(f"  ! the manifest records schedule_is_position_balanced="
              f"{declared} and the arm-runs say {balanced}")
    if not balanced:
        # A block interval is still a block interval, but arm position is
        # confounded with time when the schedule does not balance it, and the
        # name `latin` in a manifest is not evidence that it does: run T
        # recorded `latin` for three arms over four repeats, which rotates
        # 0,1,2,0.
        print("  ! this run's schedule does not put every arm in every position "
              "an equal number of times, so arm position is confounded with "
              "time within the interval below")
        for a, v in sorted(pos.items())[:4]:
            print(f"      {a}: positions {v}")
        # Printing a warning and writing the same JSON is how a warning gets
        # lost: `paired_blocks.json` looks identical whether the schedule
        # balanced or not, and it is the file the documents quote. Refuse,
        # unless the caller says in the command line that they know.
        if not allow_unbalanced:
            sys.exit("  refusing to write an interval for an unbalanced "
                     "schedule; pass --allow-unbalanced to override, and say so "
                     "wherever the number is published")
    if len(complete) < len(blocks):
        print(f"  ! {len(blocks) - len(complete)} block(s) dropped for missing arms")
    if len(complete) < 3:
        sys.exit("  too few complete blocks for a block-level interval")
    if len(complete) < 6:
        print(f"  ! {len(complete)} blocks. A percentile bootstrap under-covers badly "
              f"at this size - it can only ever resample the {len(complete)} values it "
              f"has - so read the t interval as the honest one here. The bootstrap "
              f"column is printed for continuity, not because it is trustworthy at "
              f"n={len(complete)}.")

    print(f"\n  paired against `{base}` measured in the same block; "
          f"{iters} block bootstrap resamples\n")
    print(f"  {'arm':20s} {'change':>9s}  {'95 % CI (bootstrap)':>22s}  "
          f"{'95 % CI (t)':>19s}  {'blocks':>6s}")
    print("  " + "-" * 84)
    out = []
    for a in sorted(arms, key=lambda k: -st.mean(
            [blocks[b][k] / blocks[b][base] for b in complete])):
        lr = [math.log(blocks[b][a] / blocks[b][base]) for b in complete]
        point = math.exp(st.mean(lr))
        lo, hi = bootstrap_ci(lr, iters, seed=1234)
        sd = st.stdev(lr) if len(lr) > 1 else 0.0
        # Student t, two-sided 0.975, for 8 df (nine blocks). Table value, since
        # this file must not pull in scipy.
        tcrit = t_critical_975(len(lr) - 1)
        half = tcrit * sd / math.sqrt(len(lr))
        t_lo, t_hi = st.mean(lr) - half, st.mean(lr) + half
        print(f"  {a:20s} {100*(point-1):+8.1f}%  "
              f"[{100*(math.exp(lo)-1):+7.1f}%, {100*(math.exp(hi)-1):+7.1f}%]  "
              f"[{100*(math.exp(t_lo)-1):+7.1f}%, {100*(math.exp(t_hi)-1):+7.1f}%]  "
              f"{len(lr):6d}")
        out.append({"arm": a, "point_pct": round(100 * (point - 1), 2),
                    "ci95_boot_pct": [round(100 * (math.exp(lo) - 1), 2),
                                      round(100 * (math.exp(hi) - 1), 2)],
                    "ci95_t_pct": [round(100 * (math.exp(t_lo) - 1), 2),
                                   round(100 * (math.exp(t_hi) - 1), 2)],
                    "blocks": len(lr)})

    # the baseline against itself across blocks: how much does the reference move?
    bvals = [blocks[b][base] for b in complete]
    print(f"\n  baseline across blocks: {min(bvals):.1f}-{max(bvals):.1f} tok/s, "
          f"CV {100*st.stdev(bvals)/st.mean(bvals):.2f} %")
    dest = run_dir / "paired_blocks.json"
    dest.write_text(json.dumps({"baseline": base, "blocks": len(complete),
                                "bootstrap_iters": iters,
                                "schedule_position_balanced": balanced,
                                "unbalanced_override": allow_unbalanced,
                                "t_critical_975": round(t_critical_975(len(complete) - 1), 4),
                                "interval_scope": "blocks within one invocation "
                                                  "of the driver; it does not cover "
                                                  "invocation-to-invocation variation "
                                                  "(ERRATA A16)",
                                "arms": out}, indent=2) + "\n",
                    encoding="utf-8")
    print(f"  wrote {dest.relative_to(Path.cwd()) if dest.is_relative_to(Path.cwd()) else dest}")


if __name__ == "__main__":
    main()
