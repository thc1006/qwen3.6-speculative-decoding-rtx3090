"""Compare llama.cpp's two acceptance counters for the same arm-run.

There are two, and this repository has always quoted only the first:

  server   `timings.draft_n` / `timings.draft_n_accepted`, which is what every
           table here reports and what `analysis/*.py` aggregates
  drafter  `statistics <type>: #gen tokens / #acc tokens`, printed by the
           speculator itself in the `-v` log

ERRATA A1 documented the server counter being a tautology at `97895129e`, and
recorded that post-merge master moved the denominator so it is not. That is true
for arm-runs that never take the speculative-checkpoint path. For arm-runs that
do, the two counters still disagree, and this script is what establishes the
correspondence: across every arm-run whose log survives, `checkpoints > 0`
predicts disagreement and `checkpoints == 0` predicts agreement.

It does not decide which counter is right. Where they disagree, at least one is
wrong and the log alone cannot say which - see ERRATA A13, which argues from
throughput that neither can be taken at face value on those paths.

Run: python analysis/compare_acceptance_counters.py <bench-root> > out.json
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys

RE_STATS = re.compile(
    r"statistics\s+([a-z0-9-]+): #calls\(b,g,a\) =\s*\d+\s+(\d+)\s+(\d+), "
    r"#gen drafts =\s*(\d+), #acc drafts =\s*(\d+), "
    r"#gen tokens =\s*(\d+), #acc tokens =\s*(\d+),", re.S)


def main() -> None:
    root = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/bench")
    out = []
    for run in sorted(glob.glob(os.path.join(root, "matrix_*"))):
        logs = os.path.join(run, "server_logs")
        if not os.path.isdir(logs):
            continue
        # every repeat, not just rep0. The first version globbed `*__rep0.log`
        # and stripped that literal suffix, so each arm contributed one arm-run
        # however many it had - the same restriction that made
        # `analysis/extract_checkpoint_timers.py` compare four logs of one arm
        # against one log of each control.
        for lg in sorted(glob.glob(f"{logs}/*__rep*.log")):
            stem = os.path.basename(lg)[:-len(".log")]
            arm, _, rep = stem.rpartition("__rep")
            js = os.path.join(run, f"{stem}.json")
            if not os.path.exists(js):
                continue
            d = json.load(open(js))
            if not d.get("rows"):
                continue
            s_dn = sum(x["draft_n"] for x in d["rows"])
            s_da = sum(x["draft_n_accepted"] for x in d["rows"])
            if not s_dn:
                continue
            with open(lg, errors="replace") as fh:
                text = fh.read()
            m = list(RE_STATS.finditer(text))
            if not m:
                continue
            g = m[-1]
            g_gen, g_acc = int(g.group(6)), int(g.group(7))
            out.append({
                "run": os.path.basename(run),
                "arm": arm,
                "repeat": int(rep) if rep.isdigit() else None,
                "spec_type": g.group(1),
                "server_drafted": s_dn,
                "server_accepted": s_da,
                "server_pct": round(100 * s_da / s_dn, 1),
                "drafter_calls_generate": int(g.group(2)),
                "drafter_drafts": int(g.group(4)),
                "drafter_drafted": g_gen,
                "drafter_accepted": g_acc,
                "drafter_pct": round(100 * g_acc / g_gen, 1) if g_gen else None,
                "checkpoints_created": text.count("created speculative checkpoint"),
                "checkpoints_restored": text.count("restoring speculative checkpoint"),
            })
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
