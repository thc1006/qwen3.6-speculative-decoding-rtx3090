"""Extract llama.cpp's own speculative instrumentation from `-v` server logs.

The logs are ~100 MB per matrix and are not committed. Every claim in ERRATA
A12 rests on them, so this pulls the numbers into one small JSON that IS
committed and that `verify_claims.py` can check, and records which log each
number came from.

What it reads, per arm-run:
  - `statistics <type>: #gen drafts / #acc drafts / #gen tokens / #acc tokens /
    #mean acc len / #acc rate/pos / dur(b,g,a)` — the drafter's own counters,
    including the per-draft-position acceptance curve and the time spent inside
    generate()
  - `created speculative checkpoint (... size = X MiB, draft = Y MiB)` and
    `restoring speculative checkpoint (... size = N)` — the recurrent-state
    save/restore traffic that only the external-drafter path incurs
  - the wall time that elapses immediately after each checkpoint line, against
    the median gap after any other line, as an estimate of that path's share

Timestamps are `minutes.seconds.milliseconds.microseconds`. Parsing them as
hours.minutes.seconds.ms makes a 124-second arm-run look like two hours.

Run: python analysis/extract_spec_accounting.py <log> [<log> ...] > out.json
"""
from __future__ import annotations

import json
import os
import re
import statistics as st
import sys

RE_STATS = re.compile(
    r"statistics\s+([a-z0-9-]+): #calls\(b,g,a\) =\s*\d+\s+\d+\s+\d+, "
    r"#gen drafts =\s*(\d+), #acc drafts =\s*(\d+), "
    r"#gen tokens =\s*(\d+), #acc tokens =\s*(\d+), "
    r"#mean acc len = ([\d.]+), #acc rate/pos = \(([^)]*)\)[^\n]*?"
    r"dur\(b,g,a\) = ([\d.]+), ([\d.]+), ([\d.]+) ms", re.S)
RE_CREATE = re.compile(
    r"created speculative checkpoint \([^)]*size = ([\d.]+) MiB(?:, draft = ([\d.]+) MiB)?\)")
RE_RESTORE = re.compile(r"restoring speculative checkpoint \([^)]*size = (\d+)\)")
RE_TS = re.compile(r"^\s*(\d+)\.(\d+)\.(\d+)\.(\d+)")


def stamp(line: str) -> float | None:
    m = RE_TS.match(line)
    if not m:
        return None
    mi, s, ms, us = (int(x) for x in m.groups())
    return mi * 60 + s + ms / 1000.0 + us / 1e6


def analyse(path: str) -> dict:
    text = open(path, errors="replace").read()
    out: dict = {
        "log": os.path.basename(path),
        "arm": os.path.basename(path).replace("__rep0.log", "").replace(".log", ""),
        "run": os.path.basename(os.path.dirname(os.path.dirname(path))),
    }
    created = RE_CREATE.findall(text)
    restored = [int(x) for x in RE_RESTORE.findall(text)]
    out["checkpoints_created"] = len(created)
    out["checkpoints_restored"] = len(restored)
    if created:
        out["checkpoint_target_mib"] = float(created[0][0])
        out["checkpoint_draft_mib"] = float(created[0][1]) if created[0][1] else None
        out["state_written_gib"] = round(
            sum(float(a) + (float(b) if b else 0.0) for a, b in created) / 1024, 2)
    if restored:
        out["restore_bytes_each"] = restored[0]
        out["state_read_back_gib"] = round(sum(restored) / 2 ** 30, 2)

    if (m := RE_STATS.search(text[::1]) or None) is None:
        ms_ = list(RE_STATS.finditer(text))
        m = ms_[-1] if ms_ else None
    else:
        ms_ = list(RE_STATS.finditer(text))
        m = ms_[-1] if ms_ else None
    if m:
        gen_t, acc_t = int(m.group(4)), int(m.group(5))
        out["spec_type"] = m.group(1)
        out["drafts_generated"] = int(m.group(2))
        out["draft_tokens_generated"] = gen_t
        out["draft_tokens_accepted"] = acc_t
        out["draft_token_acceptance_pct"] = round(100 * acc_t / gen_t, 1) if gen_t else None
        out["mean_accepted_length"] = float(m.group(6))
        out["acceptance_by_position"] = [
            round(float(x), 3) for x in m.group(7).replace("\n", " ").split(",")]
        out["drafter_generate_s"] = round(float(m.group(9)) / 1000, 2)

    stamped = [(stamp(l), l) for l in text.splitlines()]
    stamped = [(t, l) for t, l in stamped if t is not None]
    if len(stamped) > 2:
        ck, other = [], []
        for i in range(len(stamped) - 1):
            d = stamped[i + 1][0] - stamped[i][0]
            if d < 0 or d > 2:
                continue
            (ck if "speculative checkpoint" in stamped[i][1] else other).append(d)
        span = stamped[-1][0] - stamped[0][0]
        out["log_span_s"] = round(span, 1)
        if ck and other:
            excess = sum(ck) - len(ck) * st.median(other)
            out["checkpoint_lines"] = len(ck)
            out["checkpoint_excess_s"] = round(excess, 2)
            out["checkpoint_share_pct"] = round(100 * excess / span, 1)
    return out


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit("usage: python analysis/extract_spec_accounting.py <log> [<log> ...]")
    print(json.dumps([analyse(p) for p in sys.argv[1:]], indent=2))


if __name__ == "__main__":
    main()
