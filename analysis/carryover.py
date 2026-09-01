#!/usr/bin/env python3
"""Does an arm's decode rate depend on what ran immediately before it?

    python analysis/carryover.py <run-dir> [<run-dir> ...]
    python analysis/carryover.py --json <run-dir> ...
    python analysis/carryover.py --include-row-boundaries <run-dir> ...
    python analysis/carryover.py --vs baseline <run-dir> ...

Why this exists
---------------
Runs V2 and V3 balanced treatment POSITION and left first-order carryover
alone. V3's cyclic rotation put every capped arm immediately after its own
uncapped twin in 9 of 9 within-repeat adjacencies, so "which mode" and "what
ran before" were the same variable. ERRATA A17 said the predecessor contrast
"points the same way each time" and that a fixed rotation cannot separate it;
this is the file that separates it, given a schedule that permits the question.

It **refuses** to report a carryover contrast for a schedule that is not
first-order balanced, rather than printing a number that means something else.
That is the whole point: a design that cannot answer the question should not
produce an answer to it.

Order comes from the data, not the manifest
-------------------------------------------
`t_start` on every request is `time.perf_counter()` inside the one driver
process, so it is monotonic across the entire run and orders the arm-runs
without trusting the schedule that was planned. Run T's manifest says `latin`
and its arms sit at positions [1, 3, 2, 1]; the label is not the run.
"""
from __future__ import annotations

import glob
import json
import os
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from length_mode import contrast, interval  # noqa: E402  A17's own two


def arm_runs(run_dir: str) -> list[dict]:
    """Every arm-run, in the order it actually ran, usable ones marked.

    Ordering happens over ALL arm-run files and exclusion happens afterwards.
    Dropping a crashed arm-run first would hand its successor the wrong
    predecessor: the crashed one still ran on that card, in that slot, and
    whatever it left behind is what the next arm inherited. Filtering before
    ordering silently shifts every attribution after the gap.
    """
    out = []
    for f in glob.glob(os.path.join(run_dir, "*__rep*.json")):
        r = json.loads(Path(f).read_text(encoding="utf-8"))
        rows = r.get("rows") or []
        t0 = None
        if rows:
            try:
                t0 = min(x["t_start"] for x in rows)
            except KeyError:
                sys.exit(f"{os.path.basename(run_dir)}: {os.path.basename(f)} has "
                         f"no t_start, so the order it ran in cannot be recovered "
                         f"from the data. This tool will not fall back to the "
                         f"manifest's planned schedule.")
        ms = sum(x["predicted_ms"] for x in rows) if rows else 0
        n = sum(x["predicted_n"] for x in rows) if rows else 0
        out.append({"arm": r["arm"], "repeat": r["repeat"], "t0": t0,
                    "usable": bool(rows) and not r.get("crashed") and ms > 0,
                    "tok_s": (1000.0 * n / ms) if ms else None})
    # An arm-run with no rows has no clock of its own, so its slot in the
    # sequence cannot be recovered from the data - and guessing it from the
    # manifest is the thing this file refuses to do. One such gap breaks the
    # design's guarantee for every arm-run after it, so the run is refused
    # rather than analysed with a chain that is quietly one short. The driver
    # will not have written RUN_COMPLETE.json for such a run either.
    blind = [d for d in out if d["t0"] is None]
    if blind:
        names = ", ".join(f"{d['arm']}__rep{d['repeat']}" for d in blind[:3])
        sys.exit(f"{os.path.basename(run_dir)}: {len(blind)} arm-run(s) produced "
                 f"no requests ({names}), so the order they ran in is not in the "
                 f"data and every predecessor after the gap would be wrong.")
    out.sort(key=lambda d: d["t0"])
    for i, d in enumerate(out):
        d["predecessor"] = out[i - 1]["arm"] if i else None
        d["same_repeat"] = bool(i) and out[i - 1]["repeat"] == d["repeat"]
    return out


def hardcap_suffix(run_dir: str) -> str:
    """Which suffix marks a capped arm, from the run's own manifest."""
    try:
        m = json.loads((Path(run_dir) / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ""
    return (m.get("hardcap_suffix") or "").strip()


def within_repeat_pairs(runs: list[dict]) -> dict[str, dict[str, int]]:
    """Ordered (predecessor -> arm) counts, within a repeat only."""
    c: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for d in runs:
        # the design's balance is a property of the SCHEDULE, so count every
        # scheduled adjacency; usability is a separate question, asked below
        if d["same_repeat"]:
            c[d["arm"]][d["predecessor"]] += 1
    return {a: dict(v) for a, v in c.items()}


def is_balanced(runs: list[dict]) -> tuple[bool, str]:
    """Every arm preceded by every other exactly once, within repeats."""
    arms = sorted({d["arm"] for d in runs})
    pairs = within_repeat_pairs(runs)
    for a in arms:
        preds = pairs.get(a, {})
        want = set(arms) - {a}
        if set(preds) != want:
            missing = sorted(want - set(preds))
            worst = max(preds.items(), key=lambda kv: kv[1], default=("none", 0))
            return False, (f"{a} is preceded by {len(preds)} of {len(want)} other "
                           f"arms; most often by {worst[0]} ({worst[1]}x)"
                           + (f", never by {missing[:3]}" if missing else ""))
        if set(preds.values()) != {1}:
            return False, (f"{a}'s predecessors are not equally often: "
                           + ", ".join(f"{k} {v}x" for k, v in
                                       sorted(preds.items(), key=lambda kv: -kv[1])[:3]))
    return True, "every arm preceded by every other exactly once"


def capped_contrast(runs: list[dict], suffix: str,
                    boundaries: bool = False) -> dict:
    """Rate after a capped predecessor minus rate after an uncapped one.

    The one hypothesis A17 names and cannot test: that the arm is slower after
    a capped neighbour. One degree of freedom, computed per arm.

    WITHIN-REPEAT adjacencies only. A Williams square balances the pairs inside
    a row; the transitions between rows are the n-1 adjacencies no n x n square
    can balance, and folding them in unbalances nine arms of ten. Measured on
    the schedule this repository actually ran: `spec-dflash-n2` gets 6 capped
    and 4 free predecessors using every adjacency, against the 5 and 4 the
    design promises, and that arm is the one the experiment is about.

    `boundaries=True` folds them in anyway. That is the pre-registered
    sensitivity analysis for run W2 and nothing else: it answers "does the
    answer depend on the boundary rule", and its split is unbalanced BY
    CONSTRUCTION, which `split_is_balanced` still reports per arm.
    """
    by_arm: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"after_cap": [], "after_free": []})
    for d in runs:
        if (not d["usable"] or d["predecessor"] is None
                or not (boundaries or d["same_repeat"])):
            continue
        key = "after_cap" if d["predecessor"].endswith(suffix) else "after_free"
        by_arm[d["arm"]][key].append(d["tok_s"])
    out = {}
    for a, v in by_arm.items():
        if not v["after_cap"] or not v["after_free"]:
            continue
        # the balanced split for a 2k-arm square: an uncapped arm sees k capped
        # and k-1 free predecessors, a capped arm the reverse
        k = len(by_arm) // 2
        want = (k, k - 1) if not a.endswith(suffix) else (k - 1, k)
        got = (len(v["after_cap"]), len(v["after_free"]))
        mc, mf = st.mean(v["after_cap"]), st.mean(v["after_free"])
        out[a] = {"after_cap": mc, "after_free": mf,
                  "delta_pct": 100.0 * (mc / mf - 1.0),
                  "n_cap": got[0], "n_free": got[1],
                  "split_is_balanced": got == want,
                  "split_expected": list(want)}
    return out


def capped_contrast_matched(runs: list[dict], suffix: str,
                            boundaries: bool = False) -> dict:
    """The same question, with predecessor IDENTITY held fixed.

    `capped_contrast` above pools every capped predecessor against every free
    one, and those two groups are not matched. For an uncapped current arm `X`
    the capped group contains `X-cap` -- its own twin -- while the free group
    cannot contain `X`, because nothing precedes itself; for `X-cap` it is the
    other way round. So the grouped contrast carries three things at once: the
    predecessor's cap mode, one unmatched predecessor configuration, and a k
    against k-1 weighting. It is not a predecessor-mode effect on its own.

    This pairs within base configuration. For every OTHER base `Y`, the rate
    after `Y-cap` is compared with the rate after `Y`, and the arm's own twin is
    dropped from both sides. Each pair differs in cap mode and in nothing else,
    so their mean is a mode contrast. Sessions stay the resampling unit, as
    they are for the grouped version.
    """
    def base(a: str) -> str:
        return a[:-len(suffix)] if suffix and a.endswith(suffix) else a

    per: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for d in runs:
        if (d["usable"] and d["predecessor"] is not None
                and (boundaries or d["same_repeat"])):
            per[d["arm"]][d["predecessor"]].append(d["tok_s"])

    out = {}
    for a, preds in per.items():
        own = base(a)
        pairs, dropped = [], []
        for p_arm in sorted({base(x) for x in preds}):
            if p_arm == own:
                dropped.append(p_arm)          # the unmatched twin, excluded
                continue
            cap = preds.get(p_arm + suffix) or []
            free = preds.get(p_arm) or []
            if not cap or not free:
                continue
            mc, mf = st.mean(cap), st.mean(free)
            pairs.append({"predecessor_base": p_arm,
                          "after_cap": mc, "after_free": mf,
                          "delta_pct": 100.0 * (mc / mf - 1.0)})
        if not pairs:
            continue
        deltas = [x["delta_pct"] for x in pairs]
        out[a] = {"delta_pct": st.mean(deltas),
                  "n_pairs": len(pairs),
                  "own_twin_dropped": dropped,
                  "pairs": pairs,
                  # NOT the session SD. These four deltas are one session's
                  # four matched predecessor pairs, so this is spread WITHIN a
                  # session; the published interval is over sessions and is
                  # built by the caller from `delta_pct`. Named for what it is
                  # because a field called `sd_pct` sitting under a docstring
                  # that says "sessions stay the resampling unit" is an
                  # invitation to quote it as the session SD, and nothing reads
                  # it today to notice.
                  "sd_over_pairs_within_session_pct":
                      st.stdev(deltas) if len(deltas) > 1 else 0.0}
    return out


def mode_shift_by_predecessor(runs: list[dict], suffix: str,
                             boundaries: bool = False) -> dict:
    """`shift_pp` computed twice, on each half of the predecessor split.

    The rate contrasts above are one rate each. What A17's four-design table
    publishes is `length_mode.contrast`'s `shift_pp`: an arm's advantage over
    the baseline under the hard cap MINUS its advantage free-running, in
    percentage points. The V2-versus-W disagreement that run W2 exists to
    settle is 2.4 of those points, and a bound on one rate does not bound a
    difference of two ratios of four rates. This computes the published
    quantity separately for the arm-runs that follow a capped neighbour and
    those that follow a free one, so the two are in the same units and no
    conversion stands between them.

    GROUPED, not matched: an uncapped arm's after-cap bucket contains its own
    twin as a predecessor and its after-free bucket cannot contain itself, the
    same asymmetry `capped_contrast_matched` exists to remove. Matching here
    would need `shift_pp` per predecessor identity, which is one observation
    per cell per session. This is reported as the grouped figure it is.
    """
    buckets: dict[str, dict[str, list[float]]] = {"cap": {}, "free": {}}
    for d in runs:
        if (not d["usable"] or d["predecessor"] is None
                or not (boundaries or d["same_repeat"])):
            continue
        k = "cap" if d["predecessor"].endswith(suffix) else "free"
        buckets[k].setdefault(d["arm"], []).append(d["tok_s"])
    out = {}
    for k, by_arm in buckets.items():
        rates = {a: st.mean(v) for a, v in by_arm.items()}
        free = {a: r for a, r in rates.items() if not a.endswith(suffix)}
        cap = {a[:-len(suffix)]: r for a, r in rates.items()
               if a.endswith(suffix)}
        out[k] = contrast(free, cap)
    return {a: {"after_cap_pp": out["cap"][a]["shift_pp"],
                "after_free_pp": out["free"][a]["shift_pp"],
                "delta_pp": out["cap"][a]["shift_pp"] - out["free"][a]["shift_pp"]}
            for a in sorted(out["cap"]) if a in out["free"]}


def spread_by_predecessor(runs: list[dict]) -> dict:
    """How much an arm's rate moves across its nine predecessors."""
    per: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for d in runs:
        if d["same_repeat"] and d["usable"]:
            per[d["arm"]][d["predecessor"]].append(d["tok_s"])
    out = {}
    for a, preds in per.items():
        means = {p: st.mean(v) for p, v in preds.items()}
        if len(means) < 3:
            continue
        vals = list(means.values())
        out[a] = {"mean": st.mean(vals), "sd": st.stdev(vals),
                  "cv_pct": 100.0 * st.stdev(vals) / st.mean(vals),
                  "range_pct": 100.0 * (max(vals) - min(vals)) / st.mean(vals),
                  "slowest_after": min(means, key=means.get),
                  "fastest_after": max(means, key=means.get),
                  "by_predecessor": means}
    return out


def report(dirs: list[str], as_json: bool, boundaries: bool = False,
           reference: str = "") -> dict:
    out: dict = {"runs": [], "refused": [],
                 "row_boundaries_included": boundaries}
    per_arm_delta: dict[str, list[float]] = defaultdict(list)
    per_arm_delta_matched: dict[str, list[float]] = defaultdict(list)
    per_arm_shift: dict[str, list[float]] = defaultdict(list)
    for d in sorted(dirs):
        runs = arm_runs(d)
        name = os.path.basename(d.rstrip("/"))
        if len(runs) < 4:
            out["refused"].append({"run": name, "why": "too few usable arm-runs"})
            continue
        ok, why = is_balanced(runs)
        rec = {"run": name, "arm_runs": len(runs),
               "first_order_carryover_balanced": ok, "balance_note": why}
        if not as_json:
            print(f"\n=== {name} — {len(runs)} arm-runs ===")
            print(f"  carryover balance: {'YES' if ok else 'NO'} — {why}")
            if boundaries:
                # printed per session, not once at the top, because the tables
                # below are what gets copied out of this output
                print("  ROW BOUNDARIES INCLUDED: sensitivity only. The "
                      "predecessor split is unbalanced by construction.")
        if not ok:
            # a number computed here would be the predecessor and the treatment
            # at once, which is the thing this file exists to avoid
            rec["refused"] = ("not first-order balanced; a carryover contrast "
                              "from this schedule would be aliased with treatment")
            out["runs"].append(rec)
            if not as_json:
                print("  refusing to report a carryover contrast for this schedule")
            continue
        suffix = hardcap_suffix(d)
        if not suffix:
            rec["refused"] = ("no hardcap_suffix in the manifest, so which arms "
                              "are capped is not recorded and the contrast "
                              "cannot be defined")
            out["runs"].append(rec)
            if not as_json:
                print("  refusing: the manifest does not name the hard-cap suffix")
            continue
        rec["hardcap_suffix"] = suffix
        rec["capped_predecessor"] = capped_contrast(runs, suffix, boundaries)
        rec["capped_predecessor_matched"] = capped_contrast_matched(
            runs, suffix, boundaries)
        rec["spread"] = spread_by_predecessor(runs)
        rec["mode_shift"] = mode_shift_by_predecessor(runs, suffix, boundaries)
        unbalanced = sorted(a for a, v in rec["capped_predecessor"].items()
                            if not v["split_is_balanced"])
        if unbalanced and not boundaries:
            # the file printed n_cap and n_free and never looked at them, so its
            # own contamination was invisible in its own output
            rec["refused"] = (f"the capped/free predecessor split is not the "
                              f"balanced one for {unbalanced[:3]}")
            out["runs"].append(rec)
            if not as_json:
                print(f"  refusing: unbalanced predecessor split for {unbalanced[:3]}")
            continue
        if unbalanced:
            # Under --include-row-boundaries the split is unbalanced by
            # construction, so refusing would refuse the sensitivity analysis
            # itself. It is reported instead, marked, and every consumer that
            # reads this file has to see the marker before it sees a number.
            rec["split_unbalanced_by_construction"] = unbalanced
        for a, v in rec["capped_predecessor"].items():
            per_arm_delta[a].append(v["delta_pct"])
        for a, v in rec["capped_predecessor_matched"].items():
            per_arm_delta_matched[a].append(v["delta_pct"])
        for a, v in rec["mode_shift"].items():
            per_arm_shift[a].append(v["delta_pp"])
        if not as_json:
            print(f"  {'arm':<24} {'after -cap':>11} {'after free':>11} "
                  f"{'delta':>8}   {'spread over 9 predecessors':>26}")
            for a in sorted(rec["capped_predecessor"],
                            key=lambda k: -abs(rec["capped_predecessor"][k]["delta_pct"])):
                v = rec["capped_predecessor"][a]
                s = rec["spread"].get(a)
                sp = "" if not s else (f"CV {s['cv_pct']:.2f} %, range "
                                       f"{s['range_pct']:.2f} %")
                print(f"  {a:<24} {v['after_cap']:>10.2f} {v['after_free']:>11.2f} "
                      f"{v['delta_pct']:>+7.2f} %   {sp:>26}")
        out["runs"].append(rec)

    if per_arm_delta and max(len(v) for v in per_arm_delta.values()) > 1:
        out["across_sessions"] = {}
        if not as_json:
            print("\n  rate after a capped predecessor vs after a free one, "
                  "over the balanced sessions")
            print(f"  {'arm':<24} {'mean':>9} {'95 % t':>22} {'sessions':>9}")
        for a, vs in sorted(per_arm_delta.items(), key=lambda kv: -abs(st.mean(kv[1]))):
            m, lo, hi, n = interval(vs)
            out["across_sessions"][a] = {"mean_delta_pct": m, "lo": lo, "hi": hi,
                                         "sessions": n, "per_session": vs}
            if not as_json:
                rng = "" if lo is None else f"[{lo:+.2f} %, {hi:+.2f} %]"
                print(f"  {a:<24} {m:>+8.2f} % {rng:>22} {n:>9}")

    if per_arm_delta_matched and max(len(v) for v in per_arm_delta_matched.values()) > 1:
        out["across_sessions_matched"] = {}
        if not as_json:
            print("\n  MATCHED on predecessor configuration: the same contrast with "
                  "the arm's own\n  twin dropped and each capped predecessor paired "
                  "against its free counterpart.\n  This is the one that is a mode "
                  "contrast; the grouped figures above are not.")
            print(f"  {'arm':<24} {'mean':>9} {'95 % t':>22} {'sessions':>9}")
        for a, vs in sorted(per_arm_delta_matched.items(),
                            key=lambda kv: -abs(st.mean(kv[1]))):
            m, lo, hi, n = interval(vs)
            out["across_sessions_matched"][a] = {"mean_delta_pct": m, "lo": lo,
                                                 "hi": hi, "sessions": n,
                                                 "per_session": vs}
            if not as_json:
                rng = "" if lo is None else f"[{lo:+.2f} %, {hi:+.2f} %]"
                print(f"  {a:<24} {m:>+8.2f} % {rng:>22} {n:>9}")

    if per_arm_shift and max(len(v) for v in per_arm_shift.values()) > 1:
        out["across_sessions_mode_shift"] = {}
        if not as_json:
            print("\n  THE PUBLISHED QUANTITY, split by predecessor mode: A17's "
                  "`shift_pp` computed\n  on the arm-runs that follow a capped "
                  "neighbour minus the same on those that\n  follow a free one. "
                  "Percentage points, so a gap quoted in points compares with\n"
                  "  it directly. GROUPED, and post hoc: run W2's plan "
                  "pre-registers the matched\n  rate contrast above, not this.")
            print(f"  {'arm':<24} {'mean':>9} {'95 % t':>22} {'sessions':>9}")
        for a, vs in sorted(per_arm_shift.items(),
                            key=lambda kv: -abs(st.mean(kv[1]))):
            m, lo, hi, n = interval(vs)
            out["across_sessions_mode_shift"][a] = {
                "mean_delta_pp": m, "lo": lo, "hi": hi, "sessions": n,
                "per_session": vs}
            if not as_json:
                rng = "" if lo is None else f"[{lo:+.2f}, {hi:+.2f}]"
                print(f"  {a:<24} {m:>+7.2f} pp {rng:>22} {n:>9}")

    if reference and reference in per_arm_delta_matched:
        out["across_sessions_matched_vs"] = {"reference": reference, "arms": {}}
        ref = per_arm_delta_matched[reference]
        if not as_json:
            print(f"\n  AGAINST `{reference}`, per session: this arm's matched "
                  f"contrast minus\n  `{reference}`'s. A speedup is a ratio of "
                  f"two rates, so only the part of a\n  predecessor effect that "
                  f"does NOT also move `{reference}` can move a speedup.\n"
                  f"  This is the quantity a gap quoted in percentage POINTS has "
                  f"to be compared\n  with; the tables above are one rate each "
                  f"and do not bound it.")
            print(f"  {'arm':<24} {'mean':>9} {'95 % t':>22} {'sessions':>9}")
        for a, vs in sorted(per_arm_delta_matched.items(),
                            key=lambda kv: -abs(st.mean(
                                [x - y for x, y in zip(kv[1], ref)]))):
            if len(vs) != len(ref) or a == reference:
                # the reference against itself is identically zero, and a
                # zero-width 95 % interval in a table of intervals is a number
                # someone will quote
                continue
            diffs = [x - y for x, y in zip(vs, ref)]
            m, lo, hi, n = interval(diffs)
            out["across_sessions_matched_vs"]["arms"][a] = {
                "mean_delta_pct": m, "lo": lo, "hi": hi, "sessions": n,
                "per_session": diffs}
            if not as_json:
                rng = "" if lo is None else f"[{lo:+.2f} %, {hi:+.2f} %]"
                print(f"  {a:<24} {m:>+8.2f} % {rng:>22} {n:>9}")
    return out


def main() -> None:
    as_json = "--json" in sys.argv
    boundaries = "--include-row-boundaries" in sys.argv
    argv = sys.argv[1:]
    reference = ""
    if "--vs" in argv:
        i = argv.index("--vs")
        if i + 1 >= len(argv) or argv[i + 1].startswith("-"):
            sys.exit("--vs needs an arm name, for example `--vs baseline`")
        reference = argv[i + 1]
        argv = argv[:i] + argv[i + 2:]
    args = [a for a in argv
            if a not in ("--json", "--include-row-boundaries")]
    unknown = [a for a in args if a.startswith("-")]
    if unknown:
        sys.exit(f"unrecognised option(s): {' '.join(unknown)}. A mistyped flag "
                 f"used to be treated as a path pattern and reported as 'no "
                 f"directories', which hides the typo.")
    dirs = [d for a in args for d in glob.glob(a) if os.path.isdir(d)]
    if not dirs:
        sys.exit("usage: python analysis/carryover.py [--json] "
                 "[--include-row-boundaries] [--vs <arm>] <run-dir> [...]")
    out = report(dirs, as_json, boundaries, reference)
    if as_json:
        print(json.dumps(out, indent=2, sort_keys=True))
    elif not any(r.get("first_order_carryover_balanced") for r in out["runs"]):
        print("\n  no schedule here permits a carryover contrast.")


if __name__ == "__main__":
    main()
