#!/usr/bin/env python3
"""Did the clocks a run locked actually HOLD for the whole of it.

    check_clocks_held.py TRACE.csv CORE_MHZ MEM_MHZ

`nvidia-smi -lgc` and `-lmc` are requests. The card drops below them under the
power cap or a thermal limit, and the drop is reported in the trace and nowhere
else: a run whose clock moved is not a run at locked clocks, and every figure it
produced is a figure at some other clock. Plan Z's layer A reads bandwidth against
thermal state AT FIXED CLOCKS, so this is the difference between its measurement
and a confounded one.

Verified against the trace rather than at the start, because a readback at the
start says only that the request was accepted.

Columns are looked up by NAME through the aliases the three schemas use, and a
missing one is a refusal rather than a pass: `bench/gpu_telemetry.sh`'s `raw`
schema does not record the memory clock at all, so a run that used it cannot
answer this question and must not appear to.
"""
from __future__ import annotations

import csv
import pathlib
import statistics as st
import sys

# The spellings the three schemas use for one quantity. One place that knows them,
# for the same reason `bench/check_telemetry_cover.py` is one place that knows the
# three timestamp spellings.
ALIASES = {
    "sm": ("sm_mhz", "clk_sm", "clocks.current.sm [MHz]", "clocks.current.sm"),
    "mem": ("mem_mhz", "clk_mem", "clocks.current.memory [MHz]",
            "clocks.current.memory"),
    "temp": ("temp_c", "temp", "temperature.gpu"),
    "power": ("power_w", "pwr", "power.draw [W]", "power.draw"),
    "throttle": ("throttle_active", "throttle", "clocks_event_reasons.active",
                 "clocks_throttle_reasons.active"),
}


def _column(header: list[str], which: str) -> int | None:
    for name in ALIASES[which]:
        if name in header:
            return header.index(name)
    return None


def _num(cell: str):
    try:
        return float(cell.strip().split()[0])
    except (ValueError, IndexError):
        return None


def main() -> int:
    if len(sys.argv) != 4:
        print(__doc__, file=sys.stderr)
        return 2
    path = pathlib.Path(sys.argv[1])
    try:
        want_core, want_mem = int(sys.argv[2]), int(sys.argv[3])
    except ValueError:
        print(f"CORE_MHZ and MEM_MHZ must be integers, got "
              f"{sys.argv[2]!r} {sys.argv[3]!r}", file=sys.stderr)
        return 2
    if not path.is_file():
        print(f"FAIL: no trace at {path}", file=sys.stderr)
        return 1
    with path.open(encoding="utf-8") as fh:
        rd = csv.reader(fh)
        try:
            header = [h.strip() for h in next(rd)]
        except StopIteration:
            print(f"FAIL: {path.name} is empty", file=sys.stderr)
            return 1
        rows = [r for r in rd if r]
    if not rows:
        print(f"FAIL: {path.name} has a header and no samples", file=sys.stderr)
        return 1

    missing = [k for k in ("sm", "mem") if _column(header, k) is None]
    if missing:
        print(f"FAIL: {path.name} records no {missing}. The `raw` schema does not "
              f"record the memory clock, so a run that used it cannot show its "
              f"clocks held; use `full` or `compact`.", file=sys.stderr)
        return 1

    bad = 0
    seen = {"sm": [], "mem": [], "temp": [], "power": []}
    throttled = {}
    ti = _column(header, "throttle")
    for r in rows:
        for k in seen:
            i = _column(header, k)
            if i is not None and i < len(r):
                v = _num(r[i])
                if v is not None:
                    seen[k].append(v)
        if ti is not None and ti < len(r):
            t = r[ti].strip()
            if t and t not in ("0x0000000000000000",):
                throttled[t] = throttled.get(t, 0) + 1

    def spread(k):
        v = seen[k]
        if not v:
            return "no samples"
        return (f"min {min(v):.0f} max {max(v):.0f} mean {st.fmean(v):.1f}"
                + (f" sd {st.stdev(v):.2f}" if len(v) > 1 else ""))

    print(f"{path.name}: {len(rows)} samples")
    for k in ("sm", "mem", "temp", "power"):
        print(f"  {k:8s} {spread(k)}")
    for k, want in (("sm", want_core), ("mem", want_mem)):
        v = seen[k]
        off = [x for x in v if int(round(x)) != want]
        if not v:
            print(f"FAIL: no {k} clock samples", file=sys.stderr)
            bad += 1
        elif off:
            print(f"FAIL: the {k} clock was asked to hold {want} MHz and "
                  f"{len(off)} of {len(v)} samples are not it, down to "
                  f"{min(off):.0f}. A run whose clock moved is not a run at "
                  f"locked clocks.", file=sys.stderr)
            bad += 1
        else:
            print(f"  {k} held at {want} MHz for all {len(v)} samples")
    if throttled:
        # printed, not failed: a throttle bit with the clock still at its lock is
        # the card telling you which cap it was against, which is information. A
        # bit WITH a clock drop is the failure above.
        for t, n in sorted(throttled.items(), key=lambda x: -x[1])[:4]:
            print(f"  throttle {t} in {n} of {len(rows)} samples")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
