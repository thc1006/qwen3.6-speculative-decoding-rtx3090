"""Summarise a bench/gpu_telemetry.sh trace: throttling, clocks, and drift.

A multi-hour matrix can be biased by the GPU heating and downclocking, and no
v1, v2 or v3 run captured anything but a single pre-load snapshot (ERRATA C4b).
This turns the continuous trace into the three statements that matter:

  overclock   power.limit vs power.default_limit vs power.max_limit
  throttling  how many samples each clocks_throttle_reasons flag was active on,
              and at what clock - a thermal flag raised while the clock sits at
              its maximum has no performance consequence
  drift       first half vs second half of the run, for clock and temperature.
              If the second half is not slower, thermal bias did not occur.

Run: python analysis/thermal_report.py <telemetry.csv> [...]
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

FLAGS = ("thr_sw_thermal", "thr_hw_thermal", "thr_hw_power_brake", "thr_sw_power_cap")


def num(s: str) -> float:
    s = (s or "").strip().split()[0] if (s or "").strip() else ""
    try:
        return float(s)
    except ValueError:
        return float("nan")


def report(path: Path) -> None:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    if not rows:
        print(f"  {path}: empty")
        return
    g = lambda r, k: (r.get(k) or "").strip()  # noqa: E731
    temps = [num(g(r, "temp_c")) for r in rows]
    gfx = [num(g(r, "gfx_mhz")) for r in rows]
    temps = [t for t in temps if t == t]
    gfx = [c for c in gfx if c == c]

    print("=" * 78)
    print(f"{path.name}  -  {len(rows)} samples")
    print("=" * 78)
    lim, dflt, mx = (g(rows[0], k) for k in ("power_limit_w", "power_default_w", "gfx_max_mhz"))
    print(f"  power limit / default : {lim} / {dflt}"
          f"   {'-> NOT overclocked' if lim == dflt else '-> LIMIT DIFFERS FROM DEFAULT, check for OC'}")
    print(f"  temperature           : {min(temps):.0f}-{max(temps):.0f} C, "
          f"mean {sum(temps)/len(temps):.1f}")
    print(f"  graphics clock        : {min(gfx):.0f}-{max(gfx):.0f} MHz of {mx}, "
          f"mean {sum(gfx)/len(gfx):.0f}")
    print()
    for f in FLAGS:
        act = [r for r in rows if g(r, f) == "Active"]
        print(f"  {f:22s} active on {len(act):5d} / {len(rows)}")
        for r in act[:3]:
            print(f"      {g(r,'wall_iso')}  temp={g(r,'temp_c')}C  "
                  f"gfx={g(r,'gfx_mhz')}  power={g(r,'power_w')}")
        if act and f in ("thr_sw_thermal", "thr_hw_thermal"):
            at = [num(g(r, "gfx_mhz")) for r in act]
            if max(at) >= max(gfx):
                print("      note: raised while the clock was at the run maximum, "
                      "so no downclock accompanied it")
    print()
    h = len(gfx) // 2
    g1, g2 = sum(gfx[:h]) / h, sum(gfx[h:]) / (len(gfx) - h)
    ht = len(temps) // 2
    t1, t2 = sum(temps[:ht]) / ht, sum(temps[ht:]) / (len(temps) - ht)
    print(f"  drift, first half -> second half:")
    print(f"    clock       {g1:.0f} -> {g2:.0f} MHz  ({100*(g2/g1-1):+.2f} %)")
    print(f"    temperature {t1:.1f} -> {t2:.1f} C   ({t2-t1:+.1f} C)")
    verdict = ("no thermal bias: the second half is not slower"
               if g2 >= g1 * 0.995 else "SECOND HALF SLOWER - investigate thermal bias")
    print(f"    verdict: {verdict}")
    print()


def main() -> None:
    args = [Path(a) for a in sys.argv[1:]]
    if not args:
        sys.exit("usage: python analysis/thermal_report.py <telemetry.csv> [...]")
    for p in args:
        report(p)


if __name__ == "__main__":
    main()
