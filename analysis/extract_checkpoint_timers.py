"""Sum the source-level checkpoint timers from an instrumented `-v` log.

ERRATA A12 withdrew a wall-clock attribution derived from the gap between a
checkpoint log line and the next line, because the create message is emitted
after `update_tgt()` and the restore message before `load_tgt()`: the same rule
missed one direction and captured the other. This reads timers placed around the
calls themselves, so both directions are measured the same way.

The binary that produces these lines is NOT stock llama.cpp. Its sha256 is
recorded in the run manifest, and the run exists alongside a throughput control
against the stock build.

Run: python analysis/extract_checkpoint_timers.py <log> [<log> ...]
"""
from __future__ import annotations

import json
import os
import re
import statistics as st
import sys

# One `AUDIT_US` record per line, as `key=value` pairs. Positional regexes tied
# the reader to the field ORDER, so adding the synchronisation split - which
# inserts `sync_tgt=` after `update_tgt=` and `sync_lt=` between `load_tgt=` and
# `load_dft=` - would have silently stopped matching the restore line and
# reported zero restores rather than failing.
RE_AUDIT = re.compile(r"AUDIT_US ((?:\w+=\d+ ?)+)")
RE_PAIR = re.compile(r"(\w+)=(\d+)")


def _records(text: str) -> list[dict]:
    return [{k: int(v) for k, v in RE_PAIR.findall(body)}
            for body in RE_AUDIT.findall(text)]


def analyse(path: str) -> dict:
    with open(path, errors="replace") as fh:
        text = fh.read()
    recs = _records(text)
    tgt = [r["update_tgt"] for r in recs if "update_tgt" in r]
    dft = [r["update_dft"] for r in recs if "update_dft" in r]
    loads = [r for r in recs if "load_tgt" in r]
    lt = [r["load_tgt"] for r in loads]
    ld = [r.get("load_dft", 0) for r in loads]
    # The split, when the binary emits it. `update_tgt` and friends still mean
    # THE WHOLE CALL, so everything below is unchanged on an unsplit log; the
    # state work is the call minus its wait.
    sync = {k: [r[k] for r in recs if k in r]
            for k in ("sync_tgt", "sync_dft", "sync_lt", "sync_ld")}
    out = {
        "log": os.path.basename(path),
        # repeat-independent: the first version stripped only `__rep0.log`,
        # so rep1..N kept the suffix in `arm` and any per-arm assertion silently
        # covered rep0 alone.
        "arm": re.sub(r"__rep\d+\.log$", "", os.path.basename(path)).replace(".log", ""),
        "repeat": (int(m.group(1)) if (m := re.search(r"__rep(\d+)\.log$",
                                                      os.path.basename(path))) else None),
        "creates": len(tgt), "restores": len(loads),
        "update_tgt_s": round(sum(tgt) / 1e6, 3),
        "update_dft_s": round(sum(dft) / 1e6, 3),
        "load_tgt_s": round(sum(lt) / 1e6, 3),
        "load_dft_s": round(sum(ld) / 1e6, 3),
    }
    # rounded once, from the raw microseconds. Summing the four already-rounded
    # component fields instead can differ by up to 2 ms, which is nothing here
    # and is still the wrong arithmetic.
    out["checkpoint_total_s"] = round(
        (sum(tgt) + sum(dft) + sum(lt) + sum(ld)) / 1e6, 3)
    for name, vals in (("update_tgt", tgt), ("load_tgt", lt)):
        if vals:
            out[f"{name}_median_ms"] = round(st.median(vals) / 1000, 3)
            out[f"{name}_max_ms"] = round(max(vals) / 1000, 3)
    # additive: absent entirely on a log from the unsplit build, so the
    # committed extractions reproduce byte for byte
    if any(sync.values()):
        pairs = (("sync_tgt", tgt), ("sync_dft", dft),
                 ("sync_lt", lt), ("sync_ld", ld))
        for key, whole in pairs:
            vals = sync[key]
            if not vals:
                continue
            out[f"{key}_s"] = round(sum(vals) / 1e6, 3)
            out[f"{key.replace('sync', 'state')}_s"] = round(
                (sum(whole) - sum(vals)) / 1e6, 3)
        out["sync_total_s"] = round(
            sum(sum(v) for v in sync.values()) / 1e6, 3)
        out["state_total_s"] = round(
            ((sum(tgt) + sum(dft) + sum(lt) + sum(ld))
             - sum(sum(v) for v in sync.values())) / 1e6, 3)
        out["sync_share_of_checkpoint_pct"] = round(
            100.0 * out["sync_total_s"] / out["checkpoint_total_s"], 1
        ) if out["checkpoint_total_s"] else None
    return out


def main() -> None:
    argv = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not argv:
        sys.exit("usage: python analysis/extract_checkpoint_timers.py [--repeats N] <log> [...]")
    want = None
    if "--repeats" in sys.argv:
        want = int(sys.argv[sys.argv.index("--repeats") + 1])
        argv = [a for a in argv if a != str(want)]
    recs = [analyse(p) for p in argv]

    # A control that reports zero checkpoint operations is only as strong as the
    # number of arm-runs behind it. The first extraction covered rep0 alone for
    # both controls while covering all four repeats of the treatment, so the
    # comparison was four logs against one and nothing said so.
    by_arm: dict[str, set] = {}
    for r in recs:
        by_arm.setdefault(r["arm"], set()).add(r["repeat"])
    if want is not None:
        short = {a: sorted(v) for a, v in by_arm.items()
                 if v != set(range(want))}
        if short:
            sys.exit(f"--repeats {want} but these arms are not fully covered: "
                     + "; ".join(f"{a} has {v}" for a, v in sorted(short.items())))
    else:
        n = {len(v) for v in by_arm.values()}
        if len(n) > 1:
            print(f"# WARNING: uneven repeat coverage across arms: "
                  f"{ {a: sorted(v) for a, v in sorted(by_arm.items())} }",
                  file=sys.stderr)
    print(json.dumps(recs, indent=2))


if __name__ == "__main__":
    main()
