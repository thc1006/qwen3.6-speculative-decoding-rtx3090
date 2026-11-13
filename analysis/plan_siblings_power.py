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


def a16_step_pct() -> float:
    """A16's step, which this run is about, from `analysis/plan_z_power.py`.

    It was a literal here, 3.93, beside a comment saying that file derives it.
    """
    import plan_z_power                                            # noqa: E402
    return plan_z_power.a16_quoted()["step_pct"]


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
    step = a16_step_pct()
    # Run Y's own within-block spread, the one plan Z carries beside the null
    # spread because a real effect widens it. Taken from plan Z's derivation so
    # the two plans quote one number, and this plan carries it because it did
    # not, which left its table stating only the optimistic precision.
    import plan_z_power                                            # noqa: E402
    sd_effect = plan_z_power.design()[1]["sd_effect_pct"]
    return {"sd_pct": sd, "sd_effect_pct": sd_effect,
            "per_arm_run_h": per_arm_run_h,
            "full_displacement_pct": full, "a16_step_pct": step,
            "share": step / full,
            "blocks_measured": blocks}


def design(inp: dict) -> list[dict]:
    out = []
    for n in BLOCK_COUNTS:
        df = n - 1
        half = t_critical_975(df) * inp["sd_pct"] / math.sqrt(n)
        half_e = t_critical_975(df) * inp["sd_effect_pct"] / math.sqrt(n)
        out.append({"blocks": n,
                    "arm_runs": n * ARMS,
                    "hours": n * ARMS * inp["per_arm_run_h"],
                    "half_width_pct": half,
                    "step_in_half_widths": inp["a16_step_pct"] / half,
                    "half_width_effect_pct": half_e,
                    "step_in_effect_half_widths": inp["a16_step_pct"] / half_e})
    return out


def _split(line: str) -> list[str]:
    return [c.strip().replace("**", "") for c in line.strip().strip("|").split("|")]


EFFECT_COL = "under run Y's spread"


def cells(r: dict) -> dict:
    return {"blocks": str(r["blocks"]),
            "arm-runs": str(r["arm_runs"]),
            "hours": f"{r['hours']:.2f}",
            "95 % half width": f"{r['half_width_pct']:.2f} %",
            "the step in half widths": f"{r['step_in_half_widths']:.0f}",
            EFFECT_COL: f"{r['half_width_effect_pct']:.2f} %",
            "the step in those": f"{r['step_in_effect_half_widths']:.0f}"}


def prose_figures(inp: dict) -> dict:
    """The prose figures `check_doc` compares, as the document writes them.

    One function with two callers: `check_doc`, and `checked_figures`, which is
    what `analysis/figure_registry.py` asks. A figure added here is checked and
    counted as covered; one removed stops being both.
    """
    return {
        "the full displacement": f"{inp['full_displacement_pct']:.2f} %",
        "A16's step": f"{inp['a16_step_pct']:.2f} %",
        "the share": f"{inp['share']:.2f}",
        "the within-invocation spread": f"{inp['sd_pct']:.3f} %",
        "run Y's own block spread": f"{inp['sd_effect_pct']:.3f} %",
    }


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
    # Every row the table HAS must be one the data derives. Looking up only the
    # derived rows let a row nobody derived sit in the table and pass every check.
    table = []
    for ln in lines[heads[0] + 1:]:
        if not ln.strip().startswith("|"):
            break
        c = _split(ln)
        if c and not c[0].startswith("---"):
            table.append(c[0])
    extra = sorted(set(table) - {str(r["blocks"]) for r in rows})
    if extra:
        bad.append(f"{DOC}: the design table has row(s) for {extra} blocks, which "
                   f"nothing derives")
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
    for what, v in prose_figures(inp).items():
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
    # The two spreads beside what each is. Presence passed with them swapped:
    # "five times the within-invocation spread of 2.089 %" carried both values.
    flat = " ".join(text.split())
    pf = prose_figures(inp)
    null_sd, effect_sd = pf["the within-invocation spread"], pf["run Y's own block spread"]
    for ctx in (f"within-invocation spread of {null_sd}",
                f"within-block log ratios spread {effect_sd}"):
        if ctx not in flat:
            bad.append(f"{DOC}: does not say {ctx!r}")
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


def topology_quoted() -> dict:
    """The topology figures the document quotes, formatted as it writes them.

    `check_topology` builds its patterns from these and `checked_figures` reports
    them, so the set compared and the set reported cannot differ.
    """
    b = topology_figures(read_topology(TOPOLOGY["bench"]))
    d = topology_figures(read_topology(TOPOLOGY["devbox"]))
    return {"fav_k": f"{b['fav_k']:.1f}", "other_k": f"{b['other_k']:.1f}",
            "shared_lo": f"{b['shared_lo']:.3f}", "shared_hi": f"{b['shared_hi']:.3f}",
            "distinct_lo": f"{b['distinct_lo']:.3f}",
            "distinct_hi": f"{b['distinct_hi']:.3f}",
            "control": f"{b['control']:.3f}", "agreed": f"{b['agreed']}",
            "dev_n_distinct": f"{d['n_distinct']}",
            "dev_distinct_lo": f"{d['distinct_lo']:.3f}",
            "dev_distinct_hi": f"{d['distinct_hi']:.3f}"}


def checked_figures() -> dict:
    """Every prose figure this file compares, keyed by document, for the registry."""
    inp = inputs()
    table = {f"{r['blocks']} blocks {k}": v for r in design(inp)
             for k, v in cells(r).items()}
    return {DOC: {**table, **prose_figures(inp),
                  **{f"topology {k}": v for k, v in topology_quoted().items()}}}


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
    q = topology_quoted()
    e = {k: re.escape(v) for k, v in q.items()}
    wants = [
        (rf"{e['fav_k']} against {e['other_k']} thousand blocks per",
         "the bench host's favoured and other performance-core rates"),
        (rf"sibling pairs measuring {e['shared_lo']} to {e['shared_hi']}",
         "the bench host's sibling band"),
        (rf"distinct pairs {e['distinct_lo']} to {e['distinct_hi']}",
         "the bench host's distinct band"),
        (rf"same-processor control\s+at {e['control']}",
         "the bench host's same-processor control"),
        (rf"all {e['dev_n_distinct']}\s+of its pairs landing between "
         rf"{e['dev_distinct_lo']} and {e['dev_distinct_hi']}",
         "the development box's pairs"),
        (rf"{e['agreed']} pairs? agree with `/sys`|"
         rf"All {e['agreed']} pairs agree with `/sys`",
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
          f"{inp['full_displacement_pct']:.2f} %, so A16's {inp['a16_step_pct']:.2f} % step is "
          f"{inp['share']:.2f} of one")
    print(f"  within-invocation spread {inp['sd_pct']:.3f} %, "
          f"per arm-run {inp['per_arm_run_h'] * 3600:.0f} s over "
          f"{inp['blocks_measured']} blocks of four")
    print(f"  and run Y's own block spread {inp['sd_effect_pct']:.3f} %, which a "
          f"real effect would bring")
    print(f"  {'blocks':>6} {'arm-runs':>9} {'hours':>6} {'+-':>8} {'step in +-':>11}"
          f" {'+- under Y':>11} {'step in it':>11}")
    for r in rows:
        c = cells(r)
        print(f"  {c['blocks']:>6} {c['arm-runs']:>9} {c['hours']:>6} "
              f"{c['95 % half width']:>8} {c['the step in half widths']:>11}"
              f" {c[EFFECT_COL]:>11} {c['the step in those']:>11}")
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
