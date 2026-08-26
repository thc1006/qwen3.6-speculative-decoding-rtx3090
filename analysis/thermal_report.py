"""Summarise a bench/gpu_telemetry.sh trace: throttling, clocks, and drift.

Reads all three schemas this repository recorded. It read only the first until
2026-08-26, which meant twelve of the seventeen committed traces - including
run T's, which ERRATA A16's thermal comparison depends on - could not be
analysed with the tool that names them. The two narrower schemas carry no
power-limit, max-clock or per-flag throttle columns, so the statements that need
those are reported as unavailable rather than silently skipped.

A multi-hour matrix can be biased by the GPU heating and downclocking, and no
v1, v2 or v3 run captured anything but a single pre-load snapshot (ERRATA C4b).
This turns the continuous trace into the three statements that matter:

  overclock   power.limit vs power.default_limit vs power.max_limit
  throttling  how many samples each clocks_throttle_reasons flag was active on,
              and at what clock - a thermal flag raised while the clock sits at
              its maximum has no performance consequence
  drift       first half vs second half of the run, for clock and temperature.
              If the second half is not slower, thermal bias did not occur.
              Computed over LOADED samples only: a trace spanning gaps between
              arms would otherwise be diluted by idle-clock samples, and how
              many gaps land in each half is arbitrary. The script reports how
              many samples it excluded so that filtering is visible rather than
              silent.

Run: python analysis/thermal_report.py <telemetry.csv> [...]
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

FLAGS = ("thr_sw_thermal", "thr_hw_thermal", "thr_hw_power_brake", "thr_sw_power_cap")

# nvidia-smi's clocks_throttle_reasons.active bitmask, for the traces that
# record it instead of the per-reason columns
THROTTLE_BITS = {
    0x0000000000000001: "gpu_idle",
    0x0000000000000002: "applications_clocks_setting",
    0x0000000000000004: "sw_power_cap",
    0x0000000000000008: "hw_slowdown",
    0x0000000000000010: "sync_boost",
    0x0000000000000020: "sw_thermal_slowdown",
    0x0000000000000040: "hw_thermal_slowdown",
    0x0000000000000080: "hw_power_brake_slowdown",
    0x0000000000000100: "display_clock_setting",
}


def num(s: str) -> float:
    s = (s or "").strip().split()[0] if (s or "").strip() else ""
    try:
        return float(s)
    except ValueError:
        return float("nan")


IDLE_UTIL_PCT = 50.0   # below this a sample is treated as a gap between arms


# canonical name -> the column each schema calls it
SCHEMAS = {
    "full": {},          # already canonical
    "compact": {
        "util_pct": "util", "mem_used_mib": "mem_used", "temp_c": "temp",
        "sm_mhz": "clk_sm", "gfx_mhz": "clk_sm", "mem_mhz": "clk_mem",
        "power_w": "pwr", "throttle_active": "throttle", "ts": "ts",
    },
    "raw": {
        "util_pct": " utilization.gpu [%]", "mem_used_mib": " memory.used [MiB]",
        "temp_c": " temperature.gpu", "sm_mhz": " clocks.current.sm [MHz]",
        "gfx_mhz": " clocks.current.graphics [MHz]", "power_w": " power.draw [W]",
        "pstate": " pstate", "throttle_active": " clocks_event_reasons.active",
        "ts": "timestamp",
    },
}


def detect(fieldnames) -> str:
    names = set(fieldnames or ())
    if "gfx_max_mhz" in names:
        return "full"
    if "clk_sm" in names:
        return "compact"
    if any(n.strip() == "clocks.current.sm [MHz]" for n in names):
        return "raw"
    return "full"


def normalise(rows: list[dict], schema: str) -> list[dict]:
    """Rename a trace's columns to the `full` schema's names.

    `compact` has no separate graphics clock, so `clk_sm` stands in for both -
    on this card they are the same domain and the `full` traces agree to the
    megahertz.
    """
    m = SCHEMAS[schema]
    if not m:
        return rows
    out = []
    for r in rows:
        n = {k: v for k, v in r.items()}
        for canon, actual in m.items():
            if actual in r:
                n[canon] = r[actual]
        out.append(n)
    return out


def sample_interval(rows: list[dict]) -> float:
    """Median gap between consecutive timestamps, in seconds."""
    import datetime as _dt
    ts = []
    for r in rows:
        raw = (r.get("ts") or r.get("timestamp") or "").strip()
        for fmt in ("%Y/%m/%d %H:%M:%S.%f", "%Y/%m/%d%H:%M:%S.%f"):
            try:
                ts.append(_dt.datetime.strptime(raw, fmt))
                break
            except ValueError:
                continue
    if len(ts) < 3:
        return float("nan")
    gaps = sorted((b - a).total_seconds() for a, b in zip(ts, ts[1:]))
    return gaps[len(gaps) // 2]


def report(path: Path) -> None:
    reader = csv.DictReader(path.open(encoding="utf-8"))
    rows = list(reader)
    if not rows:
        print(f"  {path}: empty")
        return
    schema = detect(reader.fieldnames)
    rows = normalise(rows, schema)
    g = lambda r, k: (r.get(k) or "").strip()  # noqa: E731

    # Loaded samples only. `thr_gpu_idle` is authoritative when present;
    # utilisation is the fallback for traces that lack the column.
    def loaded(r) -> bool:
        if g(r, "thr_gpu_idle") == "Active":
            return False
        u = num(g(r, "util_pct"))
        return not (u == u and u < IDLE_UTIL_PCT)

    busy = [r for r in rows if loaded(r)]
    dropped = len(rows) - len(busy)
    if not busy:
        print(f"  {path.name}: no loaded samples")
        return

    temps = [t for t in (num(g(r, "temp_c")) for r in busy) if t == t]
    gfx = [c for c in (num(g(r, "gfx_mhz")) for r in busy) if c == c]
    if not temps or not gfx:
        print(f"  {path.name}: no usable clock/temperature values")
        return

    print("=" * 78)
    # The sampling interval decides what a flag count means: at 5 s each
    # flagged sample covers five seconds, at 1 s one. Two traces at different
    # intervals have comparable means and NOT comparable throttle fractions.
    iv = sample_interval(rows)
    print(f"{path.name}  -  {len(rows)} samples, {len(busy)} under load "
          f"({dropped} idle/low-utilisation samples excluded)   schema={schema}"
          + (f"   sampling {iv:.0f}s" if iv == iv else ""))
    span = [g(r, "wall_iso") for r in rows if g(r, "wall_iso")]
    if len(span) > 1:
        print(f"  trace span: {span[0]}  ->  {span[-1]}")
    print("=" * 78)
    first = next((r for r in busy if g(r, "power_limit_w")), busy[0])
    lim, dflt, mx = (g(first, k) for k in ("power_limit_w", "power_default_w", "gfx_max_mhz"))
    if lim and dflt:
        print(f"  power limit / default : {lim} / {dflt}"
              f"   {'-> NOT overclocked' if lim == dflt else '-> LIMIT DIFFERS FROM DEFAULT, check for OC'}")
    else:
        print(f"  power limit / default : not recorded in the `{schema}` schema; "
              f"the overclock check needs a `full` trace")
    print(f"  temperature           : {min(temps):.0f}-{max(temps):.0f} C, "
          f"mean {sum(temps)/len(temps):.1f}")
    print(f"  graphics clock        : {min(gfx):.0f}-{max(gfx):.0f} MHz"
          f"{f' of {mx}' if mx else ''}, mean {sum(gfx)/len(gfx):.0f}")
    print()

    if not any(f in (rows[0] or {}) for f in FLAGS):
        # The narrower schemas carry one hexadecimal bitmask instead of the
        # per-reason columns. Printing "0 / N" for each flag from a trace that
        # cannot record them would be a false all-clear, which is exactly the
        # failure this file exists to prevent, so decode the mask instead.
        counts: dict = {}
        for r in busy:
            m = g(r, "throttle_active")
            if not m:
                continue
            try:
                v = int(m, 16)
            except ValueError:
                continue
            for bit, name in THROTTLE_BITS.items():
                if v & bit:
                    counts[name] = counts.get(name, 0) + 1
        print(f"  the `{schema}` schema records one bitmask, not per-reason "
              f"columns; decoded from `clocks_throttle_reasons.active`:")
        if counts:
            for name, n in sorted(counts.items(), key=lambda kv: -kv[1]):
                print(f"    {name:34s} active on {n:5d} / {len(busy)}")
        else:
            print("    no throttle bit set on any loaded sample")
        print()
    else:
        for f in FLAGS:
            # loaded samples, matching the denominator in the header and in
            # every statement below. Counting over every row instead - which is
            # what this did - reports a different fraction from the one the
            # ERRATA C4b table quotes, out of the same file.
            act = [r for r in busy if g(r, f) == "Active"]
            print(f"  {f:22s} active on {len(act):5d} / {len(busy)}")
            for r in act[:3]:
                print(f"      {g(r,'wall_iso') or g(r,'ts')}  temp={g(r,'temp_c')}C  "
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
    # 0.5 % is the tolerance: boost clocks jitter by more than that between
    # consecutive samples, so anything smaller is not a trend.
    verdict = ("no thermal bias: the second half is not slower by more than 0.5 %"
               if g2 >= g1 * 0.995 else "SECOND HALF SLOWER BY >0.5 % - investigate")
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
