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
        blocks[r["repeat"]][r["arm"]] = pooled(r["rows"])
    return dict(blocks)


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


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opts = {a.split("=")[0]: a.split("=")[1] for a in sys.argv[1:] if "=" in a and a.startswith("--")}
    if not args:
        sys.exit("usage: python analysis/paired_blocks.py <run-dir> [--baseline=ARM] [--iters=N]")
    run_dir = Path(args[0])
    base = opts.get("--baseline", "baseline")
    iters = int(opts.get("--iters", "20000"))

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
    print(f"  ordering: {man.get('order_mode', '?')}   blocks on disk: {len(blocks)}   "
          f"complete blocks used: {len(complete)}")
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
        tcrit = {2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
                 8: 2.306, 9: 2.262, 10: 2.228}.get(len(lr) - 1, 1.96)
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
                                "bootstrap_iters": iters, "arms": out}, indent=2) + "\n",
                    encoding="utf-8")
    print(f"  wrote {dest.relative_to(Path.cwd()) if dest.is_relative_to(Path.cwd()) else dest}")


if __name__ == "__main__":
    main()
