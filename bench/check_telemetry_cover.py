#!/usr/bin/env python3
"""Does this telemetry trace actually cover the run it is named for?

`run_w_williams.sh` started the sampler in the background and never looked at it
again: no check that it was alive, no `wait` for its exit status, no requirement
that the trace span the run, no minimum sample count. A sampler that died in the
first second left a two-line CSV and the driver still reported success, so the
only record of what the card was doing would be missing exactly where it was
wanted.

    check_telemetry_cover.py TRACE.csv RUN_T0 RUN_T1 INTERVAL_S [+HHMM]

Exits non-zero, naming what is wrong, if the trace starts late, ends early, or
holds fewer than half the samples the interval implies.
"""
from __future__ import annotations

import csv
import datetime as dt
import sys

TOLERANCE_S = 5.0
MIN_FRACTION = 0.5


# `nvidia-smi --format=csv` writes `2026/08/30 22:05:54.200`, with slashes,
# and `fromisoformat` cannot read it. The first version of this file accepted
# only ISO, so it failed on EVERY trace this repository holds, including the
# one for the run it was written to guard -- and nothing noticed, because it
# had no test and the driver looked for a filename that no longer existed.
_FORMATS = ("%Y/%m/%d %H:%M:%S.%f", "%Y/%m/%d %H:%M:%S")


def _stamp(row: dict, tz: dt.timezone) -> float | None:
    """Epoch seconds for one row, with the naive case given an explicit zone.

    `nvidia-smi` writes no offset, so `strptime(...).timestamp()` reads the
    trace in whatever zone the machine running this happens to be in. That made
    the answer depend on the runner: the same trace and the same run bounds
    passed on a +0800 host and failed on a UTC one by exactly eight hours, and
    the failure was in CI on 2026-09-01, the first time anything had ever run
    this file. A verifier whose verdict depends on `TZ` is not a verifier, so
    the zone is a parameter and the caller takes it from the run's own
    manifest, which does carry an offset.
    """
    for k in ("timestamp", "time", "ts", "wall_iso"):
        v = (row.get(k) or "").strip()
        if not v:
            continue
        try:
            parsed = dt.datetime.fromisoformat(v.replace("Z", "+00:00"))
            # an ISO stamp without an offset is naive too, and gets the same
            # treatment rather than falling back to the local zone
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=tz)
            return parsed.timestamp()
        except ValueError:
            pass
        for fmt in _FORMATS:
            try:
                return dt.datetime.strptime(v, fmt).replace(tzinfo=tz).timestamp()
            except ValueError:
                continue
    return None


def main() -> int:
    if not 5 <= len(sys.argv) <= 6:
        print(__doc__, file=sys.stderr)
        return 2
    path, t0, t1, interval = (sys.argv[1], float(sys.argv[2]),
                              float(sys.argv[3]), float(sys.argv[4]))
    # The zone the trace's naive stamps are in, as +HHMM or -HHMM. Refused
    # rather than defaulted: the default was the local zone and that is the
    # defect this argument exists for.
    if len(sys.argv) == 6:
        raw = sys.argv[5]
        try:
            sign = -1 if raw[0] == "-" else 1
            tz = dt.timezone(sign * dt.timedelta(hours=int(raw[1:3]),
                                                 minutes=int(raw[3:5])))
        except (ValueError, IndexError):
            print(f"FAIL: {raw!r} is not a UTC offset like +0800", file=sys.stderr)
            return 2
    else:
        tz = None
    with open(path, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if tz is None:
        # only naive stamps need the zone, so an all-ISO trace is still fine
        if any((r.get(k) or "").strip() and "+" not in (r.get(k) or "")
               and "Z" not in (r.get(k) or "")
               for r in rows[:1] for k in ("timestamp", "time", "ts", "wall_iso")):
            print(f"FAIL: {path}: its stamps carry no offset and none was "
                  f"given, so the answer would depend on this machine's TZ",
                  file=sys.stderr)
            return 2
        tz = dt.timezone.utc
    bad = []
    if len(rows) < 2:
        print(f"FAIL: {path} holds {len(rows)} sample(s)", file=sys.stderr)
        return 1
    stamps = [x for x in (_stamp(r, tz) for r in rows) if x is not None]
    if not stamps:
        print(f"FAIL: {path} has no parsable timestamp column; columns are "
              f"{list(rows[0])}", file=sys.stderr)
        return 1
    lo, hi = min(stamps), max(stamps)
    if lo > t0 + TOLERANCE_S:
        bad.append(f"it starts {lo - t0:.0f}s after the run did")
    if hi < t1 - TOLERANCE_S:
        bad.append(f"it ends {t1 - hi:.0f}s before the run did")
    expected = (t1 - t0) / max(interval, 1e-3)
    if len(rows) < expected * MIN_FRACTION:
        bad.append(f"{len(rows)} samples over {t1 - t0:.0f}s at {interval}s, "
                   f"fewer than half the {expected:.0f} the interval implies")
    for b in bad:
        print(f"FAIL: {path}: {b}", file=sys.stderr)
    if not bad:
        print(f"telemetry covers the run: {len(rows)} samples, "
              f"{hi - lo:.0f}s span, run was {t1 - t0:.0f}s")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
