#!/usr/bin/env python3
"""Does this telemetry trace actually cover the run it is named for?

`run_w_williams.sh` started the sampler in the background and never looked at it
again: no check that it was alive, no `wait` for its exit status, no requirement
that the trace span the run, no minimum sample count. A sampler that died in the
first second left a two-line CSV and the driver still reported success, so the
only record of what the card was doing would be missing exactly where it was
wanted.

    check_telemetry_cover.py TRACE.csv RUN_T0 RUN_T1 INTERVAL_S

Exits non-zero, naming what is wrong, if the trace starts late, ends early, or
holds fewer than half the samples the interval implies.
"""
from __future__ import annotations

import csv
import datetime as dt
import sys

TOLERANCE_S = 5.0
MIN_FRACTION = 0.5


def _stamp(row: dict) -> float | None:
    for k in ("timestamp", "time", "ts"):
        v = row.get(k)
        if not v:
            continue
        try:
            return dt.datetime.fromisoformat(v.replace("Z", "+00:00")).timestamp()
        except ValueError:
            continue
    return None


def main() -> int:
    if len(sys.argv) != 5:
        print(__doc__, file=sys.stderr)
        return 2
    path, t0, t1, interval = (sys.argv[1], float(sys.argv[2]),
                              float(sys.argv[3]), float(sys.argv[4]))
    with open(path, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    bad = []
    if len(rows) < 2:
        print(f"FAIL: {path} holds {len(rows)} sample(s)", file=sys.stderr)
        return 1
    stamps = [x for x in (_stamp(r) for r in rows) if x is not None]
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
