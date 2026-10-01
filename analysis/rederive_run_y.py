#!/usr/bin/env python3
"""Re-derive every number run Y's result table publishes, and check the documents.

    python analysis/rederive_run_y.py          # derive, check both copies, exit 1 on a mismatch
    python analysis/rederive_run_y.py --show   # derive and print, check nothing

Why this file exists. Run Y's result table was computed by hand, published in
two documents, and every inferential column in it was the MEAN LOG RATIO printed
as a percentage: a mean log ratio of -0.15573 reached both documents as
"-15.57 %" when the change it describes is -14.42 %. `analysis/paired_blocks.py`
has done this correctly since run O2 -- `point = exp(mean(log ratio))` -- and it
was never run on this round, because it pairs every arm against one baseline ARM
and the contrast here is each arm against its own unpinned self.

Nothing read the table. `analysis/verify_claims.py` holds every other published
number and is one of the six files `bench/check_release_binding.py` compares
between the `v4.2` tag and HEAD, so it cannot grow an assertion for a round
published after that tag. The exclusion keeping run Y's README out of the census
said its tables were guarded by `check_data_integrity.py` instead; that walks
directory structure and reads no published value. This is what that sentence
should have named.

What it covers: the seven columns of the result table, for both arms, in both
documents, compared as whole rows -- a table in two documents with one copy
wired to an assertion is ERRATA A51 -- and the generated, drafted and accepted
token totals the mechanism paragraph quotes.

It also covers the four telemetry figures that paragraph quotes. They are not
means over the sampler's trace -- segmenting that by condition would need a
mapping between CLOCK_MONOTONIC and wall clock that the run did not record -- but
means over the per-arm-run `gpu_after` snapshots, of the SPECULATIVE arm only.
Both documents introduced them with "its", reading as the run rather than one arm
of it, and the baseline arm's utilisation barely moves where the speculative
arm's falls by fifteen points, so which arm it is was load-bearing and unstated.
"""
from __future__ import annotations

import json
import math
import pathlib
import statistics as st
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from paired_blocks import (                                      # noqa: E402
    load_blocks, t_critical_95_one_sided, t_critical_975)

ROOT = pathlib.Path(__file__).resolve().parents[1]
RUN = ROOT / "v5_pinning_2026_09_26/data/matrix_Y_pinning_20260926_131110"
# fast (performance cores) against slow (efficiency cores), within the block
CONTRASTS = (("spec-dflash-n2", "spec-dflash-n2-ecore"),
             ("baseline", "baseline-ecore"))
DOCS = ("v5_pinning_2026_09_26/README.md",
        "v4_audit_2026_08_25/PROSPECTIVE_PLAN_Y_CORE_PINNING.md")
BLOCKS = 12
# the pre-registered equivalence margin: the arm holds if the interval is inside it
MARGIN_PCT = 1.0
MINUS = "−"          # U+2212, which is what the documents carry


def _pct(v: float) -> str:
    """`-14.42 %` the way the tables spell it, with a real minus sign."""
    return f"{MINUS if v < 0 else ''}{abs(v):.2f} %"


def derive() -> list[dict]:
    blocks = load_blocks(RUN)
    rows = []
    for fast, slow in CONTRASTS:
        have = sorted(b for b, v in blocks.items() if fast in v and slow in v)
        if len(have) != BLOCKS:
            sys.exit(f"{fast}: {len(have)} complete blocks, not {BLOCKS}. "
                     f"The interval's degrees of freedom are published, so a "
                     f"different count is a different number, not a near miss.")
        f = [blocks[b][fast] for b in have]
        s = [blocks[b][slow] for b in have]
        lr = [math.log(s[k] / f[k]) for k in range(len(have))]
        m = st.fmean(lr)
        se = st.stdev(lr) / math.sqrt(len(lr))
        df = len(lr) - 1
        half = t_critical_975(df) * se
        # back-transformed, which is the whole point of this file
        change = (math.exp(m) - 1) * 100
        lo = (math.exp(m - half) - 1) * 100
        hi = (math.exp(m + half) - 1) * 100
        bound = (1 - math.exp(m - t_critical_95_one_sided(df) * se)) * 100
        rows.append({
            "arm": fast, "fast": st.fmean(f), "slow": st.fmean(s),
            "change_pct": change, "ci_lo_pct": lo, "ci_hi_pct": hi,
            "one_sided_bound_pct": bound, "blocks": len(lr), "df": df,
            "mean_log_ratio": m,
            # the equivalence reading: holds only if the whole interval is inside
            # the margin, so a wide interval that straddles it does not pass
            "verdict": ("holds" if -MARGIN_PCT <= lo and hi <= MARGIN_PCT
                        else "moves"),
        })
    return rows


def cells(r: dict) -> dict:
    """What each column of the result table should hold, by its header.

    Keyed by header rather than position, because the two documents carry the
    same table with different column SETS: the README adds the equivalence
    verdict and the plan does not. A checker that compared one canonical row
    string would have had to skip the plan's copy, and skipping a copy is what
    ERRATA A51 is about.
    """
    return {
        "arm": f"`{r['arm']}`",
        "fast": f"{r['fast']:.3f}",
        "slow": f"{r['slow']:.3f}",
        "change": _pct(r["change_pct"]),
        "95 % interval": f"[{_pct(r['ci_lo_pct'])}, {_pct(r['ci_hi_pct'])}]",
        "one-sided upper limit": _pct(r["one_sided_bound_pct"]),
        "plus or minus one per cent": r["verdict"],
    }


def _split(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def check_doc(doc: str, rows: list[dict], tel: dict | None = None,
              tot: dict | None = None, pro: dict | None = None) -> list[str]:
    """Every cell of every arm row, and every figure the prose quotes.

    The prose checks live here and not in `main`, so the unit suite exercises the
    same function CI does. A check that only one of the two runs is a check with
    half the coverage its name claims.
    """
    text = (ROOT / doc).read_text(encoding="utf-8")
    lines = text.splitlines()
    head = [k for k, ln in enumerate(lines)
            if ln.startswith("| arm |") and "one-sided upper limit" in ln]
    if len(head) != 1:
        return [f"{doc}: {len(head)} result tables found, expected exactly one"]
    cols = _split(lines[head[0]])
    want_all = cells(rows[0])
    unknown = [c for c in cols if c not in want_all]
    if unknown:
        # a column added to the table and not wired here is A51 exactly: it
        # would sit in a published table with nothing deriving it
        return [f"{doc}: the table has column(s) {unknown} that nothing derives"]
    bad = []
    for r in rows:
        want = cells(r)
        got = [ln for ln in lines[head[0]:] if _split(ln)[:1] == [f"`{r['arm']}`"]]
        if len(got) != 1:
            bad.append(f"{doc}: {len(got)} rows for {r['arm']}, expected one")
            continue
        for col, cell in zip(cols, _split(got[0])):
            if cell != want[col]:
                bad.append(f"{doc}: {r['arm']} column {col!r} is {cell!r} "
                           f"and the data gives {want[col]!r}")
    for cond in ("fast", "slow"):
        for name, v in (tel or {}).get(cond, {}).items():
            # one decimal, as both documents spell them
            if f"{v:.1f}" not in text:
                bad.append(f"{doc}: does not carry {name} {v:.1f} for the {cond} "
                           f"condition of {CONTRASTS[0][0]}")
    for what, (v, forb) in (pro or {}).items():
        forbidden = sorted({w for w in forb if w != v})
        if v not in text:
            bad.append(f"{doc}: does not carry {v} as the {what}")
        for w in forbidden:
            if w != v and w in text:
                bad.append(f"{doc}: carries {w} where the {what} is {v}")
    for arm, totals in (tot or {}).items():
        if arm.endswith("-ecore"):
            continue
        for n in totals:
            if n and f"{n:,}".replace(",", " ") not in text:
                bad.append(f"{doc}: does not carry {arm}'s token total "
                           + f"{n:,}".replace(",", " "))
    return bad


def token_totals() -> dict:
    out = {}
    for p in sorted(RUN.glob("*__rep*.json")):
        j = json.loads(p.read_text(encoding="utf-8"))
        g, d, a = out.setdefault(j["arm"], [0, 0, 0])
        out[j["arm"]] = [g + sum(r["predicted_n"] for r in j["rows"]),
                         d + sum(r["draft_n"] for r in j["rows"]),
                         a + sum(r["draft_n_accepted"] for r in j["rows"])]
    return out


def telemetry() -> dict:
    """Per-condition means of the `gpu_after` snapshots, speculative arm only.

    One snapshot per arm-run, taken when it finished, so twelve per condition.
    The field order is the manifest's `gpu_fields` rather than a literal here: a
    hand-written column list is what this repository keeps finding one level up.
    """
    man = json.loads((RUN / "manifest.json").read_text(encoding="utf-8"))
    fields = [f.strip() for f in str(man["gpu_fields"]).split(",")]
    idx = {n: i for i, n in enumerate(fields)}
    for n in ("utilization.gpu", "power.draw", "temperature.gpu",
              "clocks.current.sm"):
        if n not in idx:
            sys.exit(f"the manifest's gpu_fields has no {n!r}, so the snapshot "
                     f"columns cannot be read by name")
    arm = CONTRASTS[0][0]
    out = {}
    for cond, suffix in (("fast", ""), ("slow", "-ecore")):
        acc = {}
        n = 0
        for p in sorted(RUN.glob(f"{arm}{suffix}__rep*.json")):
            j = json.loads(p.read_text(encoding="utf-8"))
            if j["arm"] != arm + suffix:
                continue
            cells = [c.strip() for c in j["gpu_after"].split(",")]
            for name in ("utilization.gpu", "power.draw", "temperature.gpu",
                         "clocks.current.sm"):
                acc.setdefault(name, []).append(
                    float(cells[idx[name]].split()[0]))
            n += 1
        if n != BLOCKS:
            sys.exit(f"{arm}{suffix}: {n} snapshots, not {BLOCKS}")
        out[cond] = {k: st.fmean(v) for k, v in acc.items()}
    return out


# ERRATA A16's step, which this archive's own checker holds. Taken as an input
# here rather than re-derived, and named so that the share below is traceable to
# it: the share was the one figure that moved with the log-ratio defect, because
# it was computed against the wrong bound.
A16_STEP_PCT = 3.93


def derived_prose() -> dict:
    """The remaining quoted figures that the committed tree can answer for.

    Not the row count or digest of the untruncated trace, which is on the bench
    host and not here, and not the dates.
    """
    csv = sorted((RUN.parent).glob("gpu_telemetry_Y_*.csv"))
    if len(csv) != 1:
        sys.exit(f"{len(csv)} telemetry traces beside the run directory, expected one")
    rows = csv[0].read_text(encoding="utf-8").splitlines()
    change = abs(derive()[0]["change_pct"])
    # Not here, and each for a reason. The untruncated trace's row count and
    # digest are of a file on the bench host. The megabytes of server logs are of
    # files this round does not commit, by the same policy as the v4 archive. The
    # arm-run count is spelled as a word, and is already structural: `derive`
    # refuses anything but twelve complete blocks per contrast and `telemetry`
    # anything but twelve snapshots per condition, which is the same forty-eight.
    def grouped(n):
        return f"{n:,}".replace(",", " ")

    first = derive()[0]
    log_change = abs(first["mean_log_ratio"] * 100)
    bound = first["one_sided_bound_pct"]
    # the bound as it was published, on the log scale
    log_bound = abs(first["mean_log_ratio"] * 100) + (bound - abs(
        (math.exp(first["mean_log_ratio"]) - 1) * 100))
    # Each figure comes with the values that must NOT appear. A presence check
    # alone is weak where a figure is quoted twice, which the row count is: one
    # occurrence can be edited and the other still satisfies it. The forbidden
    # neighbours are what makes it bite, and for the share they include the value
    # the log-ratio defect produced.
    return {
        # the header is not a sample, and the document says so
        "trace rows inside the invocation":
            (grouped(len(rows) - 1), [grouped(len(rows)), grouped(len(rows) - 2)]),
        # A16's step as a share of a full displacement, the one figure that moved
        # with the log-ratio defect. Both arms round alike.
        "A16's step as a share of a full displacement":
            (f"{A16_STEP_PCT / change:.2f}",
             # the log-scale change; the one-sided bound, which is not a full
             # displacement and is what the published `0.23` divided by; the same
             # bound on the log scale, which is where that `0.23` actually came
             # from; and the two adjacent roundings
             [f"{A16_STEP_PCT / log_change:.2f}",
              f"{A16_STEP_PCT / bound:.2f}",
              f"{A16_STEP_PCT / log_bound:.2f}",
              f"{A16_STEP_PCT / change - 0.01:.2f}",
              f"{A16_STEP_PCT / change + 0.01:.2f}"]),
    }


# the pre-registration's power table: blocks, hours, the one-sided bound a true
# null would leave, and a simulated two-sided reading
PLAN_POWER_BLOCKS = (6, 12, 18)


def plan_power() -> dict:
    """The bound column of the plan's power table, and the hours it predicted.

    The bound is closed form: the one-sided t point on n-1 degrees of freedom
    times the within-invocation SD over the root of n. The SD is the one
    `analysis/load_run_power.py` measures from run T4's adjacent differences, so
    the two plans rest on the same number rather than on two of them.

    The hours column is a prediction, and this run is the thing that tests it:
    the invocation's own span is compared against the row that was chosen. The
    fourth column, the simulated probability that a two-sided reading declares
    "holds" under a true null, is NOT derived here. It needs the simulator
    `load_run_power.py` carries for the other plan's designs, and that one is
    written around a between-condition contrast rather than this one. Named
    rather than left for a reader to notice.
    """
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    import load_run_power                                        # noqa: E402
    sd = load_run_power.measured()["adjacent_diff_sd"]
    man = json.loads((RUN / "manifest.json").read_text(encoding="utf-8"))
    created = man["created"]
    ends = []
    for q in sorted(RUN.glob("*__rep*.json")):
        j = json.loads(q.read_text(encoding="utf-8"))
        ends += [r["t_end"] for r in j["rows"]]
    starts = []
    for q in sorted(RUN.glob("*__rep*.json")):
        j = json.loads(q.read_text(encoding="utf-8"))
        starts += [r["t_start"] for r in j["rows"]]
    span_h = (max(ends) - min(starts)) / 3600.0
    return {
        "sd_pct": sd,
        "created": created,
        "observed_span_hours": span_h,
        "bounds": {n: t_critical_95_one_sided(n - 1) * sd / math.sqrt(n)
                   for n in PLAN_POWER_BLOCKS},
    }


def check_plan_power(pp: dict) -> list[str]:
    """The plan's own power table, row by row, anchored on the block count."""
    doc = DOCS[1]
    bad = []
    lines = (ROOT / doc).read_text(encoding="utf-8").splitlines()
    seen = set()
    for ln in lines:
        cells_ = _split(ln)
        if len(cells_) != 4:
            continue
        label = cells_[0].replace("*", "").strip()
        if not label.isdigit() or int(label) not in pp["bounds"]:
            continue
        n = int(label)
        seen.add(n)
        want = f"{pp['bounds'][n]:.2f} %"
        got = cells_[2].replace("*", "").strip()
        if got != want:
            bad.append(f"{doc}: the bound for {n} blocks is {got!r} and the "
                       f"measured spread gives {want!r}")
    missing = sorted(set(pp["bounds"]) - seen)
    if missing:
        bad.append(f"{doc}: the power table has no row for {missing} blocks, so "
                   f"those bounds are derived against nothing")
    # the chosen row against what the invocation actually took
    chosen = 12
    predicted = None
    for ln in lines:
        cells_ = _split(ln)
        if len(cells_) == 4 and cells_[0].replace("*", "").strip() == str(chosen):
            predicted = float(cells_[1].replace("*", "").strip())
    if predicted is None:
        bad.append(f"{doc}: no row for the {chosen} blocks that ran")
    elif abs(predicted - pp["observed_span_hours"]) > 0.05:
        bad.append(f"{doc}: it predicted {predicted} hours for {chosen} blocks "
                   f"and the invocation spanned "
                   f"{pp['observed_span_hours']:.2f}")
    return bad


def main() -> None:
    show_only = "--show" in sys.argv[1:]
    for a in sys.argv[1:]:
        if a != "--show":
            sys.exit(f"unknown argument {a!r}; the only one is --show")
    rows = derive()
    print(f"{RUN.name}: {BLOCKS} blocks, {rows[0]['df']} degrees of freedom")
    for r in rows:
        print(f"  mean log ratio {r['mean_log_ratio']:+.5f}  ->  "
              f"change {r['change_pct']:+.2f} %   "
              f"(printing the log ratio as a percentage gives "
              f"{r['mean_log_ratio'] * 100:+.2f} %, which is the defect)")
        c = cells(r)
        print("  | " + " | ".join(c[k] for k in (
            "arm", "fast", "slow", "change", "95 % interval",
            "one-sided upper limit", "plus or minus one per cent")) + " |")
    tot = token_totals()
    for fast, slow in CONTRASTS:
        if tot[fast] != tot[slow]:
            sys.exit(f"{fast}: generated, drafted and accepted totals are "
                     f"{tot[fast]} fast and {tot[slow]} slow. The documents say "
                     f"the output is identical to the token, and it is not.")
        print(f"  {fast}: generated {tot[fast][0]}, drafted {tot[fast][1]}, "
              f"accepted {tot[fast][2]}, identical in both conditions")
    tel, pro, pp = telemetry(), derived_prose(), plan_power()
    print(f"  {CONTRASTS[0][0]} `gpu_after` means, twelve snapshots per condition:")
    for name, unit in (("clocks.current.sm", " MHz"), ("power.draw", " W"),
                       ("utilization.gpu", " %"), ("temperature.gpu", " °C")):
        print(f"    {name:20s} fast {tel['fast'][name]:8.1f}{unit}   "
              f"slow {tel['slow'][name]:8.1f}{unit}")
    for what, (v, forb) in pro.items():
        print(f"    {what}: {v}   "
              f"(and not {', '.join(sorted({w for w in forb if w != v}))})")
    print(f"  the plan's power table, from a within-invocation spread of "
          f"{pp['sd_pct']:.3f} %:")
    for n, b in pp["bounds"].items():
        print(f"    {n:2d} blocks -> one-sided bound {b:.2f} %")
    print(f"    the invocation that ran spanned {pp['observed_span_hours']:.2f} "
          f"hours over twelve blocks")
    if show_only:
        return
    bad = []
    for doc in DOCS:
        # the prose figures are quoted by the round's own README and not by the
        # plan, which published the table before the data existed
        bad += check_doc(doc, rows, tel, tot,
                         pro if doc == DOCS[0] else None)
    bad += check_plan_power(pp)
    if bad:
        print("\nFAILED")
        for b in bad:
            print("  " + b)
        sys.exit(1)
    print(f"\nboth copies of the table match the data, in {len(DOCS)} documents")


if __name__ == "__main__":
    main()
