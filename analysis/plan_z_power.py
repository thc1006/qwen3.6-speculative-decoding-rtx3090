#!/usr/bin/env python3
"""Derive what `PROSPECTIVE_PLAN_Z_MEMORY_STATE.md` publishes, and check it.

    python analysis/plan_z_power.py          # derive, check the document, exit 1 on a mismatch
    python analysis/plan_z_power.py --show   # derive and print, check nothing

Run Y left ERRATA A16 with two hypotheses: memory-subsystem thermal state, and
page-cache or allocator state. Plan Z is about the first, and it does not hunt
A16's step: that step is a telegraph between invocations, it cannot be summoned,
and run T4 caught it once in six repeats. It measures the SENSITIVITY of the
decode rate to memory state instead, and asks whether that sensitivity times the
thermal divergence this card can actually produce could reach A16's step at all.

Three things this file derives from data already committed, because they are the
plan's premises and a premise in prose is a premise nothing checks:

  1. the recorded GPU state on both sides of A16's step, per repeat of the arm it
     is about. The step is between T4's first three repeats and its last three.
  2. that T4's committed telemetry trace cannot be attributed to arm-runs, which
     is why the plan's first prerequisite is a timestamp the run does not record.
  3. the design table: blocks against hours and against the precision each gives,
     from the within-invocation spread `analysis/load_run_power.py` measures and
     from the span run Y actually took.
"""
from __future__ import annotations

import csv
import datetime as dt
import json
import math
import pathlib
import statistics as st
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from paired_blocks import t_critical_95_one_sided, t_critical_975   # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
DOC = "v4_audit_2026_08_25/PROSPECTIVE_PLAN_Z_MEMORY_STATE.md"
# the arm A16's step is about, and the repeats each level is the mean of
STEP_ARM = "spec-dflash-n2"
LOW_REPS, HIGH_REPS = (0, 1, 2), (3, 4, 5)
# the two memory clocks `nvidia-smi -q -d SUPPORTED_CLOCKS` offers on this card,
# read on 2026-10-01 and applied and reverted once to confirm they take
MEM_CLOCKS = (9501, 5001)
BLOCK_COUNTS = (6, 12, 18)
# four arm-runs to a block: two arms at each of the two memory clocks
ARM_RUNS_PER_BLOCK = 4


def _t4():
    import load_run_power                                           # noqa: E402
    return load_run_power


def step_thermal_state() -> dict:
    """The recorded GPU state on each side of A16's step, from the snapshots.

    Per arm-run, not from the trace: the trace cannot be attributed to an arm-run
    (see `trace_is_not_attributable`), and these snapshots can, because the runner
    takes them at each arm-run's own boundaries.
    """
    L = _t4()
    man = json.loads((L.T4 / "manifest.json").read_text(encoding="utf-8"))
    fields = [f.strip() for f in str(man["gpu_fields"]).split(",")]
    idx = {n: i for i, n in enumerate(fields)}
    out = {}
    for name, reps in (("low", LOW_REPS), ("high", HIGH_REPS)):
        acc = {"rate": [], "before_c": [], "after_c": [], "sm": [], "power_w": []}
        for i in reps:
            j = json.loads((L.T4 / f"{STEP_ARM}__rep{i}.json").read_text(
                encoding="utf-8"))

            def cell(which, field):
                return float([c.strip() for c in j[which].split(",")]
                             [idx[field]].split()[0])

            acc["rate"].append(L.pooled(STEP_ARM, i))
            acc["before_c"].append(cell("gpu_before", "temperature.gpu"))
            acc["after_c"].append(cell("gpu_after", "temperature.gpu"))
            acc["sm"].append(cell("gpu_after", "clocks.current.sm"))
            acc["power_w"].append(cell("gpu_after", "power.draw"))
        out[name] = {k: st.fmean(v) for k, v in acc.items()}
    out["gap_pct"] = 100.0 * (out["high"]["rate"] / out["low"]["rate"] - 1.0)
    out["after_delta_c"] = out["high"]["after_c"] - out["low"]["after_c"]
    out["before_delta_c"] = out["high"]["before_c"] - out["low"]["before_c"]
    return out


def trace_is_not_attributable() -> dict:
    """Why the plan's first prerequisite is a timestamp.

    The arm-runs stamp requests with `CLOCK_MONOTONIC` and the sampler writes wall
    clock, so attributing a sample to an arm-run needs an anchor. The only one the
    run offers is the manifest's `created`, against the first request's monotonic
    time, and that assumes no server launch precedes the first request. One does.

    Checked on a signal the outcome does not use: the three arms hold different
    drafters, so their VRAM footprints differ, and a correct alignment would put
    each arm's own footprint inside its own windows. The reconstruction does not:
    the window LENGTHS come back to within a few seconds of the recorded request
    spans, and the footprints are scrambled across arms.
    """
    L = _t4()
    man = json.loads((L.T4 / "manifest.json").read_text(encoding="utf-8"))
    fields = [f.strip() for f in str(man["gpu_fields"]).split(",")]
    created = dt.datetime.fromisoformat(man["created"])
    runs = []
    for q in sorted(L.T4.glob("*__rep*.json")):
        j = json.loads(q.read_text(encoding="utf-8"))
        if not j.get("rows"):
            continue
        runs.append((min(r["t_start"] for r in j["rows"]),
                     max(r["t_end"] for r in j["rows"]), j["arm"]))
    runs.sort()
    offset = created.timestamp() - runs[0][0]
    traces = sorted(L.T4.parent.glob("gpu_telemetry_T4_*.csv"))
    if len(traces) != 1:
        sys.exit(f"{len(traces)} T4 traces beside the run directory, expected one")
    rows = []
    with traces[0].open(encoding="utf-8") as fh:
        rd = csv.reader(fh)
        hdr = [h.strip() for h in next(rd)]
        for r in rd:
            if len(r) < len(hdr):
                continue
            t = dt.datetime.strptime(r[0].strip(), "%Y/%m/%d %H:%M:%S.%f").replace(
                tzinfo=created.tzinfo).timestamp()
            rows.append((t, [c.strip() for c in r]))
    col = {h: i for i, h in enumerate(hdr)}
    snap = {}
    for arm in {a for _, _, a in runs}:
        j = json.loads((L.T4 / f"{arm}__rep0.json").read_text(encoding="utf-8"))
        snap[arm] = float([c.strip() for c in j["gpu_after"].split(",")]
                          [fields.index("memory.used")].split()[0])
    inside = {}
    for s, e, arm in runs:
        for t, r in rows:
            if s + offset <= t <= e + offset:
                inside.setdefault(arm, []).append(
                    float(r[col["mem_used"]].split()[0]))
    wrong = sorted(a for a, v in inside.items()
                   if max(v) > snap[a] + 1.0 or min(v) < 1000.0)
    return {
        "samples": len(rows),
        "interval_s": (rows[-1][0] - rows[0][0]) / (len(rows) - 1),
        "request_span_s": sum(e - s for s, e, _ in runs),
        "samples_inside": sum(len(v) for v in inside.values()),
        "arms_whose_footprint_is_wrong": wrong,
        "arms": sorted(snap),
    }


def design() -> list[dict]:
    """Blocks against hours and against the precision each gives.

    The spread is the one `load_run_power.measured()` takes from T4's adjacent
    differences, which is the WITHIN-invocation figure and so the right one for a
    contrast measured inside one invocation. Run Y's own observed within-block
    log-ratio spread is carried beside it, because a real effect widens the
    spread and the precision under an effect is the one that matters here.

    The hours are not an estimate: run Y ran this many arm-runs to a block and
    the span it took is recorded, so the per-block cost is measured.
    """
    L = _t4()
    sd_null = L.measured()["adjacent_diff_sd"]
    import rederive_run_y                                           # noqa: E402
    # the spread run Y measured on its own twelve within-block log ratios
    blocks = rederive_run_y.load_blocks(rederive_run_y.RUN)
    fast, slow = rederive_run_y.CONTRASTS[0]
    have = sorted(b for b, v in blocks.items() if fast in v and slow in v)
    sd_effect = 100.0 * st.stdev(
        [math.log(blocks[b][slow] / blocks[b][fast]) for b in have])
    per_block_h = rederive_run_y.plan_power()["observed_span_hours"] / len(have)
    out = []
    for n in BLOCK_COUNTS:
        df = n - 1
        out.append({
            "blocks": n,
            "hours": per_block_h * n,
            "bound_null_pct": t_critical_95_one_sided(df) * sd_null / math.sqrt(n),
            "half_width_null_pct": t_critical_975(df) * sd_null / math.sqrt(n),
            "half_width_effect_pct": t_critical_975(df) * sd_effect / math.sqrt(n),
        })
    hi, lo = max(MEM_CLOCKS), min(MEM_CLOCKS)
    return out, {"sd_null_pct": sd_null, "sd_effect_pct": sd_effect,
                 "per_block_hours": per_block_h, "blocks_measured": len(have),
                 "arm_runs_per_block": ARM_RUNS_PER_BLOCK,
                 "mem_clock_hi": hi, "mem_clock_lo": lo,
                 # the treatment stated as what it does to bandwidth, which is
                 # proportional to the clock
                 "mem_clock_cut_pct": 100.0 * (hi - lo) / hi}


# This repository spells a small number in prose, so a check that looked for the
# digit would have forced the document to read worse to satisfy it.
_WORDS = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
          "nine", "ten", "eleven", "twelve")


def _word(n: int) -> str:
    if not 0 <= n < len(_WORDS):
        raise SystemExit(f"{n} is outside the range this spells")
    return _WORDS[n]


def _neighbours(v: str) -> list[str]:
    """The spellings next to a figure, which must not appear beside it.

    A number gets the two values one step away in its own last decimal place; a
    spelled count gets the words either side of it. Nothing else: a neighbour list
    that guessed would fail on prose it has no business reading.
    """
    out = []
    head = v.split(" ", 1)[0]
    tail = v[len(head):]
    if head in _WORDS:
        i = _WORDS.index(head)
        out += [_WORDS[j] + tail for j in (i - 1, i + 1) if 0 <= j < len(_WORDS)]
        return out
    try:
        dec = len(head.split(".", 1)[1]) if "." in head else 0
        x = float(head)
    except ValueError:
        return out
    step = 10.0 ** -dec
    for y in (x - step, x + step):
        out.append(f"{y:.{dec}f}{tail}")
    return out


def _split(line: str) -> list[str]:
    return [c.strip().replace("**", "") for c in line.strip().strip("|").split("|")]


def step_cells(s: dict) -> dict:
    """What each row of the step table should hold, by its own header."""
    out = {}
    for name, label in (("low", "slow, repeats 0 to 2"),
                        ("high", "fast, repeats 3 to 5")):
        d = s[name]
        out[label] = {
            "level": label,
            "decode": f"{d['rate']:.3f} tok/s",
            "core before": f"{d['before_c']:.1f} °C",
            "core after": f"{d['after_c']:.1f} °C",
            "SM clock": f"{d['sm']:.0f} MHz",
            "power": f"{d['power_w']:.1f} W",
        }
    return out


def design_cells(rows: list[dict]) -> dict:
    out = {}
    for r in rows:
        out[str(r["blocks"])] = {
            "blocks": str(r["blocks"]),
            "hours": f"{r['hours']:.2f}",
            "one-sided bound under a null": f"{r['bound_null_pct']:.2f} %",
            "± under a null": f"{r['half_width_null_pct']:.2f} %",
            "± under an effect": f"{r['half_width_effect_pct']:.2f} %",
        }
    return out


def _check_table(lines, head_starts, want, bad, doc, what):
    """Every cell of the rows the document has, against the header it declares."""
    heads = [k for k, ln in enumerate(lines) if ln.startswith(head_starts)]
    if len(heads) != 1:
        bad.append(f"{doc}: {len(heads)} {what} tables, expected exactly one")
        return
    cols = _split(lines[heads[0]])
    first = cols[0]
    known = next(iter(want.values()))
    unknown = [c for c in cols if c not in known]
    if unknown:
        bad.append(f"{doc}: the {what} table has column(s) {unknown} that nothing "
                   f"derives")
        return
    seen = set()
    for ln in lines[heads[0] + 1:]:
        cells = _split(ln)
        if len(cells) != len(cols) or cells[0] in ("---", ""):
            if not ln.strip().startswith("|"):
                break
            continue
        if cells[0] not in want:
            continue
        seen.add(cells[0])
        for col, cell in zip(cols, cells):
            if cell != want[cells[0]][col]:
                bad.append(f"{doc}: {what} row {cells[0]!r} column {col!r} is "
                           f"{cell!r} and the data gives {want[cells[0]][col]!r}")
    missing = sorted(set(want) - seen)
    if missing:
        bad.append(f"{doc}: the {what} table has no row for {missing}, keyed on "
                   f"{first!r}")


def check_doc() -> list[str]:
    """The two tables cell by cell, and every figure the prose derives.

    The levers table is NOT checked: its rows are what the card answered on one
    day, which is host state and not in this tree. It says so in the document.
    """
    s, (rows, inp) = step_thermal_state(), design()
    t = trace_is_not_attributable()
    text = (ROOT / DOC).read_text(encoding="utf-8")
    lines = text.splitlines()
    bad: list[str] = []
    _check_table(lines, ("| level |",), step_cells(s), bad, DOC, "step")
    _check_table(lines, ("| blocks |",), design_cells(rows), bad, DOC, "design")
    flat = " ".join(text.replace("**", " ").replace("`", " ").split())
    figures = {
        "A16's step": f"{s['gap_pct']:.2f} %",
        "how much hotter the faster level ends": f"{s['after_delta_c']:.2f} °C",
        "the trace's samples": f"{t['samples']} samples",
        "its interval": f"{t['interval_s']:.2f} s",
        "the request spans it is checked against": f"{t['request_span_s']:.0f} s",
        "the memory clock cut": f"{inp['mem_clock_cut_pct']:.2f} %",
        "the within-invocation spread": f"{inp['sd_null_pct']:.3f} %",
        "run Y's own block spread": f"{inp['sd_effect_pct']:.3f} %",
        "the per-block cost": f"{inp['per_block_hours'] * 60:.1f} min",
        "the arm-runs to a block": f"{_word(inp['arm_runs_per_block'])} arm-runs",
    }
    # Each figure comes with the values that must NOT appear. A presence check
    # alone is weak where a figure is quoted twice, which the arm-runs per block
    # is: one occurrence can be edited and the other still satisfies it.
    for what, v in figures.items():
        if v not in flat:
            bad.append(f"{DOC}: does not carry {v} as {what}")
        for w in _neighbours(v):
            if w in flat:
                bad.append(f"{DOC}: carries {w} where {what} is {v}")
    if len(t["arms_whose_footprint_is_wrong"]) != len(t["arms"]):
        bad.append(f"{DOC} says all three arms' windows hold a footprint that is "
                   f"not theirs, and "
                   f"{len(t['arms_whose_footprint_is_wrong'])} of {len(t['arms'])} "
                   f"do, so the sentence has to change with the data")
    return bad


def main() -> None:
    show = "--show" in sys.argv[1:]
    for a in sys.argv[1:]:
        if a != "--show":
            sys.exit(f"unknown argument {a!r}; the only one is --show")
    s = step_thermal_state()
    print(f"A16's step, from run T4's own snapshots of {STEP_ARM}:")
    print(f"  slow level  {s['low']['rate']:.3f} tok/s   core "
          f"{s['low']['before_c']:.1f} -> {s['low']['after_c']:.1f} C   "
          f"sm {s['low']['sm']:.0f} MHz   {s['low']['power_w']:.1f} W")
    print(f"  fast level  {s['high']['rate']:.3f} tok/s   core "
          f"{s['high']['before_c']:.1f} -> {s['high']['after_c']:.1f} C   "
          f"sm {s['high']['sm']:.0f} MHz   {s['high']['power_w']:.1f} W")
    print(f"  the gap is {s['gap_pct']:+.2f} % and the faster level ends "
          f"{s['after_delta_c']:+.2f} C, which is the WRONG SIGN for a thermal "
          f"explanation")
    t = trace_is_not_attributable()
    print(f"T4's trace: {t['samples']} samples at {t['interval_s']:.2f} s, "
          f"{t['samples_inside']} inside the reconstructed windows against "
          f"{t['request_span_s']:.0f} s of recorded request spans")
    print(f"  arms whose own VRAM footprint is not what their windows hold: "
          f"{t['arms_whose_footprint_is_wrong']} of {t['arms']}")
    rows, inp = design()
    print(f"design: {inp['arm_runs_per_block']} arm-runs to a block, "
          f"{inp['per_block_hours'] * 60:.1f} min each as run Y measured over "
          f"{inp['blocks_measured']} blocks")
    print(f"  spreads: {inp['sd_null_pct']:.3f} % within an invocation, "
          f"{inp['sd_effect_pct']:.3f} % on run Y's own block ratios")
    print(f"  the treatment: {inp['mem_clock_hi']} against {inp['mem_clock_lo']} MHz, "
          f"a {inp['mem_clock_cut_pct']:.2f} % cut in memory clock and so in bandwidth")
    print(f"  {'blocks':>6} {'hours':>6} {'bound':>7} {'+-null':>8} {'+-effect':>9}")
    for r in rows:
        print(f"  {r['blocks']:>6} {r['hours']:>6.2f} {r['bound_null_pct']:>6.2f} % "
              f"{r['half_width_null_pct']:>6.2f} % {r['half_width_effect_pct']:>7.2f} %")
    if show:
        return
    bad = check_doc()
    if bad:
        print("\nFAILED")
        for b in bad:
            print("  " + b)
        sys.exit(1)
    print(f"\n{DOC} matches the data it derives from")


if __name__ == "__main__":
    main()
