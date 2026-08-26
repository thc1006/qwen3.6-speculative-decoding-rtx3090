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

RE_TGT = re.compile(r"AUDIT_US update_tgt=(\d+)")
RE_DFT = re.compile(r"AUDIT_US update_dft=(\d+)")
RE_LOAD = re.compile(r"AUDIT_US load_tgt=(\d+) load_dft=(\d+)")


def analyse(path: str) -> dict:
    text = open(path, errors="replace").read()
    tgt = [int(x) for x in RE_TGT.findall(text)]
    dft = [int(x) for x in RE_DFT.findall(text)]
    loads = RE_LOAD.findall(text)
    lt = [int(a) for a, _ in loads]
    ld = [int(b) for _, b in loads]
    out = {
        "log": os.path.basename(path),
        "arm": os.path.basename(path).replace("__rep0.log", "").replace(".log", ""),
        "creates": len(tgt), "restores": len(loads),
        "update_tgt_s": round(sum(tgt) / 1e6, 3),
        "update_dft_s": round(sum(dft) / 1e6, 3),
        "load_tgt_s": round(sum(lt) / 1e6, 3),
        "load_dft_s": round(sum(ld) / 1e6, 3),
    }
    out["checkpoint_total_s"] = round(
        out["update_tgt_s"] + out["update_dft_s"] + out["load_tgt_s"] + out["load_dft_s"], 3)
    for name, vals in (("update_tgt", tgt), ("load_tgt", lt)):
        if vals:
            out[f"{name}_median_ms"] = round(st.median(vals) / 1000, 3)
            out[f"{name}_max_ms"] = round(max(vals) / 1000, 3)
    return out


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit("usage: python analysis/extract_checkpoint_timers.py <log> [...]")
    print(json.dumps([analyse(p) for p in sys.argv[1:]], indent=2))


if __name__ == "__main__":
    main()
