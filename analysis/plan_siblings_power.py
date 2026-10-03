#!/usr/bin/env python3
"""Derive what `PROSPECTIVE_PLAN_SIBLINGS.md` publishes, and check it.

    python analysis/plan_siblings_power.py          # derive, check the document
    python analysis/plan_siblings_power.py --show   # derive and print, check nothing

Run Y measured the host channel: moving llama.cpp's eight threads from eight
distinct performance cores to eight efficiency cores costs 14.42 % of the decode
rate, so ERRATA A16's 3.93 % step is 0.27 of a full displacement. An earlier
reading turned that fraction into a count of displaced threads. It cannot be one,
for two reasons this file states as numbers where it can:

  1. every target and drafter layer is on the card, so the eight CPU threads do no
     arithmetic while decoding. What run Y displaced was the MAIN thread's work.
  2. a barrier over statically split work takes its slowest thread's time, so
     displacing one thread costs what displacing eight does. There is no linear
     interpolation to put 0.27 on.

So the successor is not a sweep over how many threads are slow. Three arms differ
in ONE thing each, and the design table here is derived from what run Y actually
measured rather than estimated: its within-invocation spread, its per-arm-run cost,
and its own two levels.
"""
from __future__ import annotations

import math
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
from paired_blocks import t_critical_975                           # noqa: E402

DOC = "v4_audit_2026_08_25/PROSPECTIVE_PLAN_SIBLINGS.md"
BLOCK_COUNTS = (6, 12, 18)
ARMS = 3
# A16's step, which this run is about, and which `analysis/plan_z_power.py` derives
# from run T4's own arm-runs.
A16_STEP_PCT = 3.93


def inputs() -> dict:
    """What run Y measured, reused rather than re-estimated."""
    import load_run_power                                          # noqa: E402
    import rederive_run_y                                          # noqa: E402
    sd = load_run_power.measured()["adjacent_diff_sd"]
    y = rederive_run_y.derive()
    pp = rederive_run_y.plan_power()
    # run Y ran four arm-runs to a block; this runs three, and the cost is per
    # arm-run rather than per block
    # run Y's own span over its own arm-runs: twelve blocks of four
    blocks = len([b for b in rederive_run_y.load_blocks(rederive_run_y.RUN)])
    per_arm_run_h = pp["observed_span_hours"] / (blocks * 4)
    full = abs(y[0]["change_pct"])
    return {"sd_pct": sd, "per_arm_run_h": per_arm_run_h,
            "full_displacement_pct": full,
            "share": A16_STEP_PCT / full,
            "blocks_measured": blocks}


def design(inp: dict) -> list[dict]:
    out = []
    for n in BLOCK_COUNTS:
        df = n - 1
        half = t_critical_975(df) * inp["sd_pct"] / math.sqrt(n)
        out.append({"blocks": n,
                    "arm_runs": n * ARMS,
                    "hours": n * ARMS * inp["per_arm_run_h"],
                    "half_width_pct": half,
                    "step_in_half_widths": A16_STEP_PCT / half})
    return out


def _split(line: str) -> list[str]:
    return [c.strip().replace("**", "") for c in line.strip().strip("|").split("|")]


def cells(r: dict) -> dict:
    return {"blocks": str(r["blocks"]),
            "arm-runs": str(r["arm_runs"]),
            "hours": f"{r['hours']:.2f}",
            "95 % half width": f"{r['half_width_pct']:.2f} %",
            "the step in half widths": f"{r['step_in_half_widths']:.0f}"}


def check_doc(inp: dict, rows: list[dict]) -> list[str]:
    """The design table by header, and the prose figures with their neighbours."""
    path = ROOT / DOC
    if not path.is_file():
        return [f"{DOC} does not exist"]
    text = path.read_text(encoding="utf-8").replace("**", "").replace("°C", "C")
    lines = text.splitlines()
    bad = []
    heads = [k for k, l in enumerate(lines) if l.startswith("| blocks |")]
    if len(heads) != 1:
        return [f"{DOC}: {len(heads)} design tables, expected one"]
    cols = _split(lines[heads[0]])
    known = cells(rows[0])
    unknown = [c for c in cols if c not in known]
    if unknown:
        return [f"{DOC}: the design table has column(s) {unknown} that nothing derives"]
    for r in rows:
        want = cells(r)
        got = [ln for ln in lines[heads[0]:] if _split(ln)[:1] == [str(r["blocks"])]]
        if len(got) != 1:
            bad.append(f"{DOC}: {len(got)} rows for {r['blocks']} blocks")
            continue
        for col, cell in zip(cols, _split(got[0])):
            if cell != want[col]:
                bad.append(f"{DOC}: {r['blocks']} blocks, column {col!r} is {cell!r} "
                           f"and the data gives {want[col]!r}")
    # the prose, each figure with the values one step away forbidden: a presence
    # check is weak where a figure is quoted twice, which this plan's own
    # displacement is
    figures = {
        "the full displacement": f"{inp['full_displacement_pct']:.2f} %",
        "A16's step": f"{A16_STEP_PCT} %",
        "the share": f"{inp['share']:.2f}",
        "the within-invocation spread": f"{inp['sd_pct']:.3f} %",
    }
    for what, v in figures.items():
        if v not in text:
            bad.append(f"{DOC}: does not carry {v} as {what}")
            continue
        head = v.split(" ", 1)[0]
        tail = v[len(head):]
        dec = len(head.split(".", 1)[1]) if "." in head else 0
        step = 10.0 ** -dec
        for y in (float(head) - step, float(head) + step):
            near = f"{y:.{dec}f}{tail}"
            if near != v and near in text:
                bad.append(f"{DOC}: carries {near} where {what} is {v}")
    return bad


# --- the topology runs, and the figures the document quotes from them ---------
# Added because the document quoted seven of them and nothing derived any. Four
# were wrong, every one by a single unit in the last place, because the prose was
# written from an interactive run and a LATER run was committed as the evidence
# beside it. That is the shape of defect this repository exists to remove: a
# published number whose cited artefact says something else, small enough that
# nobody reading either would notice.
TOPOLOGY = {
    "bench": "v4_audit_2026_08_25/topology/topology_3090_20261003.txt",
    "devbox": "v4_audit_2026_08_25/topology/topology_devbox_20261003.txt",
}
# Measured, and recorded in the bench host's own run: processors 8 to 11 are the
# two cores Turbo Boost Max favours. Named here rather than inferred from the
# rates, so that a run where they stopped being faster FAILS instead of quietly
# redefining which ones they are.
FAVOURED = frozenset({8, 9, 10, 11})


def read_topology(rel: str) -> dict:
    """Parse one recorded run. Refuses rather than returning a partial reading."""
    path = ROOT / rel
    if not path.is_file():
        raise SystemExit(f"{rel} is missing; the document cites it")
    txt = path.read_text(encoding="utf-8")
    solo = {int(m[1]): float(m[2])
            for m in re.finditer(r"cpu (\d+)\s+([\d.]+) blocks/s", txt)}
    pairs = [(int(a), int(b), float(r), v) for a, b, r, v in
             re.findall(r"(\d+) \+ (\d+)\s+([\d.]+)\s+(shared|distinct)", txt)]
    ctl = re.search(r"two threads on cpu (\d+) alone: ([\d.]+)", txt)
    agree = re.search(r"agrees with /sys on all (\d+) pair\(s\)", txt)
    if not solo or not pairs or not ctl:
        raise SystemExit(f"{rel}: parsed {len(solo)} solo rates, {len(pairs)} "
                         f"pairs and {'a' if ctl else 'no'} control; a reading "
                         f"this incomplete would compare against nothing")
    if not agree:
        raise SystemExit(f"{rel} does not report agreement with /sys, so the "
                         f"document cannot cite it for the masks")
    if int(agree[1]) != len(pairs):
        raise SystemExit(f"{rel} says it agreed on {agree[1]} pairs and lists "
                         f"{len(pairs)}")
    return {"rel": rel, "solo": solo, "pairs": pairs,
            "control_cpu": int(ctl[1]), "control": float(ctl[2]),
            "agreed": int(agree[1])}


def topology_figures(rec: dict) -> dict:
    """Every quantity the document quotes, derived from one recorded run."""
    solo, pairs = rec["solo"], rec["pairs"]
    fav = [v for c, v in solo.items() if c in FAVOURED]
    other_p = [v for c, v in solo.items() if c < 16 and c not in FAVOURED]
    shared = [r for _a, _b, r, v in pairs if v == "shared"]
    distinct = [r for _a, _b, r, v in pairs if v == "distinct"]
    out = {"processors": len(solo), "control": rec["control"],
           "agreed": rec["agreed"], "n_shared": len(shared),
           "n_distinct": len(distinct)}
    if shared:
        out["shared_lo"], out["shared_hi"] = min(shared), max(shared)
    if distinct:
        out["distinct_lo"], out["distinct_hi"] = min(distinct), max(distinct)
    if fav and other_p:
        # The separation, not just the maximum. `max(fav)` is insensitive to one
        # of the four dropping out: cpu 8 was dropped to a non-favoured rate in a
        # deliberate corruption and the quoted figure still matched, because three
        # others were still above it. What the masks rest on is that the SET is
        # separated, so that is what is asserted. The comment above FAVOURED
        # promised this failure and the code did not have it.
        if min(fav) <= max(other_p):
            slow_fav = min((c for c in solo if c in FAVOURED), key=lambda c: solo[c])
            fast_other = max((c for c in solo if c < 16 and c not in FAVOURED),
                             key=lambda c: solo[c])
            raise SystemExit(
                f"{rec['rel']}: the two performance-core tiers are not separated "
                f"in this run. cpu {slow_fav} is in FAVOURED and measured "
                f"{solo[slow_fav]:.1f}; cpu {fast_other} is outside it and "
                f"measured {solo[fast_other]:.1f}. FAVOURED names "
                f"{sorted(FAVOURED)} and the masks in the document balance on that "
                f"naming, so a run where it is false has to be read rather than "
                f"quoted.")
        out["fav_k"] = max(fav) / 1000.0
        out["other_k"] = max(other_p) / 1000.0
        out["fav_ratio"] = (sum(fav) / len(fav)) / (sum(other_p) / len(other_p))
        out["fav_margin"] = min(fav) - max(other_p)
    return out


def check_topology(doc: str) -> list:
    """Anchored, one pattern per figure. Presence would not be a check: 0.496 is
    both a sibling ratio and, one unit away, something else, and the document
    carries several numbers in the same band.

    The document spells small numbers out in narrative prose and writes DIGITS
    where the number is the claim being checked. That is deliberate rather than
    inconsistent: comparing against a spelled number needs a spelling table inside
    the checker, and a table of number words is one more thing that can drift away
    from what it describes. A figure under check is written in the form the check
    can read.
    """
    bad = []
    b = topology_figures(read_topology(TOPOLOGY["bench"]))
    d = topology_figures(read_topology(TOPOLOGY["devbox"]))
    wants = [
        (rf"{b['fav_k']:.1f} against {b['other_k']:.1f} thousand blocks per",
         "the bench host's favoured and other performance-core rates"),
        (rf"sibling pairs measuring {b['shared_lo']:.3f} to {b['shared_hi']:.3f}",
         "the bench host's sibling band"),
        (rf"distinct pairs {b['distinct_lo']:.3f} to {b['distinct_hi']:.3f}",
         "the bench host's distinct band"),
        (rf"same-processor control\s+at {b['control']:.3f}",
         "the bench host's same-processor control"),
        (rf"all {d['n_distinct']}\s+of its pairs landing between "
         rf"{d['distinct_lo']:.3f} and {d['distinct_hi']:.3f}",
         "the development box's pairs"),
        (rf"{b['agreed']} pairs? agree with `/sys`|"
         rf"All {b['agreed']} pairs agree with `/sys`",
         "how many pairs agreed on the bench host"),
    ]
    flat = " ".join(doc.replace("**", " ").split())
    for pat, what in wants:
        if not re.search(pat.replace(" ", r"\s+"), flat):
            bad.append(f"md: {what} is not what "
                       f"{TOPOLOGY['bench'] if 'bench host' in what else TOPOLOGY['devbox']} "
                       f"says; expected to find /{pat}/")
    return bad


def main() -> int:
    show = "--show" in sys.argv[1:]
    for a in sys.argv[1:]:
        if a != "--show":
            print(f"unknown argument {a!r}; the only one is --show", file=sys.stderr)
            return 2
    inp = inputs()
    rows = design(inp)
    print(f"run Y measured: a full displacement costs "
          f"{inp['full_displacement_pct']:.2f} %, so A16's {A16_STEP_PCT} % step is "
          f"{inp['share']:.2f} of one")
    print(f"  within-invocation spread {inp['sd_pct']:.3f} %, "
          f"per arm-run {inp['per_arm_run_h'] * 3600:.0f} s over "
          f"{inp['blocks_measured']} blocks of four")
    print(f"  {'blocks':>6} {'arm-runs':>9} {'hours':>6} {'+-':>8} {'step in +-':>11}")
    for r in rows:
        c = cells(r)
        print(f"  {c['blocks']:>6} {c['arm-runs']:>9} {c['hours']:>6} "
              f"{c['95 % half width']:>8} {c['the step in half widths']:>11}")
    if show:
        return 0
    bad = check_doc(inp, rows) + check_topology(
        (ROOT / DOC).read_text(encoding="utf-8"))
    if bad:
        print("\nFAILED")
        for b in bad:
            print("  " + b)
        return 1
    print(f"\n{DOC} matches what run Y measured and what the topology runs say")
    return 0


if __name__ == "__main__":
    sys.exit(main())
