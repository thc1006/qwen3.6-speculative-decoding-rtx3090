#!/usr/bin/env python3
"""What `ignore_eos` does to a speculative arm, with the invocation controlled.

Run V measured the hard cap by running one whole matrix free-running and
another whole matrix capped, sixteen minutes later. ERRATA A16 finds an
unexplained DFlash-specific invocation effect of the same size as the shift run
V reported - runs U3 and U5 are six minutes apart and differ by 8.30 pp on that
drafter with nothing changed - so run V measures a difference it cannot
attribute.

Two designs answer that, and this reads both.

**crossover** (`bench/run_v2_crossover.sh`): eight sessions of two halves each,
in AB BA BA AB BA AB AB BA order, so each mode runs first four times and second
four times with the two orders balanced in mean time position. The session is
the resampling unit. Within a session,

    shift_pp(a) = 100*(r_a,cap/r_base,cap - 1) - 100*(r_a,free/r_base,free - 1)

an **absolute change in percentage points** of the arm-versus-baseline figure,
which is the quantity A17's tables publish and the quantity
`analysis/length_matching.py` reports for the same arms by a different method.
Because the order is balanced, an order effect cannot masquerade as a mode
effect in that mean - and splitting the sessions by which mode ran first says
whether one is present.

This file used to define the session effect as a difference of log ratios,

    d(a) = log( r_a,cap / r_base,cap ) - log( r_a,free / r_base,free )

say the mode effect was the mean of d, and then average `shift_pp` instead.
Those are different estimands and they diverge exactly where the baseline ratio
is far from 1: `spec-draft-n8` moves from about -76.8 % to -70.5 %, which is
+6.31 pp absolutely and about +27 % multiplicatively. The pp figure is the one
published, and the log contrast is computed and reported beside it as a
sensitivity so a reader can see when the choice matters.

**within** (`bench/run_v3_within.sh`): one balanced square containing both
modes, `<arm>` and `<arm>-cap` side by side. The same d(a), computed inside a
single invocation, so the drafter is in whatever state it is in for both halves
of the contrast. This is **within-invocation replication with a fixed
predecessor**, not a design that identifies the effect: the square is a cyclic
rotation, so every capped arm follows its own uncapped twin and mode is aliased
with first-order carryover. The AB/BA crossover is the primary evidence; this
says whether the same contrast survives inside one invocation.

    python analysis/length_mode.py <run-dir> [<run-dir> ...]
    python analysis/length_mode.py --json <run-dir> ...

Directories are classified by what they contain, not by their names.
"""

import argparse
import glob
import json
import math
import os
import statistics as st
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paired_blocks import t_critical_975  # noqa: E402  - same directory

CAP = "-cap"


def row_first(run_dir):
    """{(arm, repeat)} for the arm-run that OPENED each row of the square.

    A Williams square balances first-order carryover among the adjacencies
    INSIDE a row. Running its ten rows back to back adds nine more transitions,
    from the last treatment of one row to the first of the next, and no n x n
    square balances those. `carryover.py` already excludes them from the
    predecessor contrast; this makes the same set available to the mode
    estimator, which pooled every arm-run including these.

    Derived from the data's own timestamps, not from the schedule: the first
    arm-run of each repeat is the one whose predecessor sits in another repeat.
    """
    import carryover as _co
    try:
        runs = _co.arm_runs(run_dir)
    except SystemExit:
        # `arm_runs` refuses a directory whose arm-runs carry no `t_start`,
        # because the order they ran in is then not in the data and it will not
        # fall back to the manifest's PLANNED schedule. That refusal is right,
        # and it must not take the whole analysis with it: the boundary
        # exclusion is a sensitivity pass, so a run that cannot support it is
        # reported as such and the all-data estimate still stands.
        return None
    return {(d["arm"], d["repeat"]) for d in runs if not d["same_repeat"]}


def pooled(run_dir, arm, drop_row_first=False):
    """Pooled decode rate over every repeat: 1000 * sum(n) / sum(ms).

    `drop_row_first` removes the arm-runs that opened a row, whose predecessor
    the design does not balance. Position balance puts every treatment in the
    first slot exactly once per session, so this drops one arm-run per arm per
    session -- symmetric across arms, and the observations it drops are the ones
    the carryover guarantee never covered.
    """
    ms = n = 0
    gen = []
    if drop_row_first:
        skip = row_first(run_dir)
        if skip is None:
            return None
    else:
        skip = set()
    for f in sorted(glob.glob(os.path.join(run_dir, f"{arm}__rep*.json"))):
        if skip:
            _rep = int(os.path.basename(f).split("__rep")[1].split(".")[0])
            if (arm, _rep) in skip:
                continue
        with open(f, encoding="utf-8") as fh:
            body = json.load(fh)
        # a crashed arm-run can still carry the requests it got through before
        # it died; averaging those into a rate is averaging a partial run with
        # whole ones, which is not what "pooled over every repeat" means
        if body.get("crashed"):
            continue
        rows = body["rows"]
        ms += sum(r["timings"]["predicted_ms"] for r in rows)
        n += sum(r["timings"]["predicted_n"] for r in rows)
        gen += [r["timings"]["predicted_n"] for r in rows]
    if not n:
        return None
    return {"tok_s": 1000.0 * n / ms, "generated": n,
            "lengths": sorted(set(gen)), "requests": len(gen)}


def arms_of(run_dir):
    return sorted({os.path.basename(f).split("__rep")[0]
                   for f in glob.glob(os.path.join(run_dir, "*__rep*.json"))})


def manifest(run_dir):
    p = os.path.join(run_dir, "manifest.json")
    if not os.path.exists(p):
        return {}
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def complete(run_dir):
    return os.path.exists(os.path.join(run_dir, "RUN_COMPLETE.json"))


def contrast(free_rates, cap_rates, base="baseline"):
    """Everything the two reports say about one configuration, computed once.

    The first version of this had `delta()` returning a log ratio and each
    report recomputing the percentage-point shift from the rates directly - the
    same quantity by two formulas, in two places, and the log value was never
    actually used. That is the defect ERRATA B8 is about, written into the
    analysis of the run that was meant to settle A17. One definition now, and
    both reports read it.

      change_pct   how much faster than the baseline MEASURED IN THE SAME MODE
      shift_pp     the hard-cap change minus the freerun change, which is what
                   A17's table publishes
      log_delta    the same contrast as a log ratio. Reported as a
                   sensitivity, not as the published figure: it answers a
                   different question, and for an arm at -76 % of baseline the
                   two disagree by a factor of four
    """
    if base not in free_rates or base not in cap_rates:
        return {}
    out = {}
    for a in sorted(free_rates):
        if a == base or a not in cap_rates:
            continue
        free_pct = 100.0 * (free_rates[a] / free_rates[base] - 1.0)
        cap_pct = 100.0 * (cap_rates[a] / cap_rates[base] - 1.0)
        out[a] = {"free_pct": free_pct, "cap_pct": cap_pct,
                  "shift_pp": cap_pct - free_pct,
                  "log_delta": (math.log(cap_rates[a] / cap_rates[base])
                                - math.log(free_rates[a] / free_rates[base]))}
    return out


def interval(values):
    """Mean and a two-sided 95 % Student-t interval over the given values."""
    n = len(values)
    m = st.mean(values)
    if n < 2:
        return m, None, None, n
    half = t_critical_975(n - 1) * st.stdev(values) / math.sqrt(n)
    return m, m - half, m + half, n


def classify(dirs):
    """Split the directories into within-invocation runs and crossover halves."""
    within, halves = [], []
    for d in dirs:
        arms = arms_of(d)
        if not arms:
            continue
        if any(a.endswith(CAP) for a in arms):
            within.append(d)
        else:
            man = manifest(d)
            mode = ("hardcap" if man.get("ignore_eos") else "freerun")
            base = os.path.basename(d.rstrip("/"))
            halves.append((base, mode, d))
    return within, halves


def session_of(name):
    """Which session a half belongs to.

    `matrix_V2_s3_hardcap_20260827_041902` -> `s3`. Run V's own halves carry no
    session marker - `matrix_V_freerun_20260826_210956` and
    `matrix_V_hardcap_20260826_210956` - so they fall back to the shared
    trailing timestamp and come out as ONE session, which is what they are, and
    one session buys no interval.
    """
    parts = name.split("_")
    for part in parts:
        if len(part) > 1 and part[0] == "s" and part[1:].isdigit():
            return part
    stamp = [p for p in parts if p.isdigit() and len(p) in (6, 8)]
    return "_".join(stamp[-2:]) if len(stamp) >= 2 else name


ALLOW_INCOMPLETE = "--allow-incomplete" in sys.argv


def report_within(dirs, out):
    print("=== both modes inside one invocation ===")
    per_arm = defaultdict(list)
    per_arm_log = defaultdict(list)
    per_arm_nb = defaultdict(list)      # the same contrast, row openers dropped
    for d in sorted(dirs):
        arms = arms_of(d)
        free = {a: pooled(d, a)["tok_s"] for a in arms
                if not a.endswith(CAP) and pooled(d, a)}
        cap = {a[:-len(CAP)]: pooled(d, a)["tok_s"] for a in arms
               if a.endswith(CAP) and pooled(d, a)}
        # The same two dictionaries with the row-opening arm-runs removed. Those
        # observations have a predecessor from the previous row, which no n x n
        # Williams square balances, and pooling them made the mode estimator
        # rest on adjacencies the design does not cover. Position balance means
        # exactly one per arm per session is dropped, so the exclusion is
        # symmetric.
        _orderable = row_first(d) is not None
        if _orderable:
            free_nb = {a: pooled(d, a, True)["tok_s"] for a in arms
                       if not a.endswith(CAP) and pooled(d, a, True)}
            cap_nb = {a[:-len(CAP)]: pooled(d, a, True)["tok_s"] for a in arms
                      if a.endswith(CAP) and pooled(d, a, True)}
        else:
            free_nb = cap_nb = {}
            if manifest(d).get("order_mode") == "williams":
                # A Williams run is the one design whose primary estimate is
                # supposed to exclude the row openers. If its order cannot be
                # recovered, that estimate does not exist and the run must not
                # be reported as though it did.
                sys.exit(f"{os.path.basename(d.rstrip('/'))} is a williams run "
                         f"whose arm-run order cannot be recovered from the "
                         f"data, so the row-boundary exclusion its design needs "
                         f"cannot be computed.")
            print("    (row-boundary exclusion unavailable: no recoverable "
                  "arm-run order in this directory)")
        man = manifest(d)
        print(f"\n  {os.path.basename(d.rstrip('/'))}  "
              f"complete={complete(d)}  order={man.get('order_mode')}  "
              f"balanced={man.get('schedule_is_position_balanced')}  "
              f"carryover={man.get('schedule_first_order_carryover_balanced')}  "
              f"repeats={man.get('repeats')}")
        # `complete` was printed and never acted on. A run the driver refused
        # to attest is a run whose cells are not all there, and averaging it
        # with runs that are is how an incomplete matrix reaches a table.
        if not complete(d) and not ALLOW_INCOMPLETE:
            sys.exit(f"{os.path.basename(d.rstrip('/'))} has no RUN_COMPLETE.json, "
                     f"so the driver did not attest it. Pass --allow-incomplete "
                     f"to analyse it anyway, and do not publish the result.")
        # every free arm must have a capped twin and the reverse, or the
        # contrast is over whichever arms happened to appear in both
        if set(free) != set(cap):
            only_f, only_c = sorted(set(free) - set(cap)), sorted(set(cap) - set(free))
            sys.exit(f"{os.path.basename(d.rstrip('/'))}: arms without a twin "
                     f"(free only: {only_f}, capped only: {only_c})")
        c = contrast(free, cap)
        if not c:
            print("    no usable baseline pair in this run")
            continue
        print(f"    {'configuration':<22} {'freerun':>9} {'hard cap':>10} "
              f"{'shift':>9}")
        for a, v in sorted(c.items(), key=lambda kv: -kv[1]["shift_pp"]):
            print(f"    {a:<22} {v['free_pct']:+8.2f}% {v['cap_pct']:+9.2f}% "
                  f"{v['shift_pp']:+8.2f} pp")
            per_arm[a].append(v["shift_pp"])
            per_arm_log[a].append(v["log_delta"])
        c_nb = contrast(free_nb, cap_nb) if free_nb and cap_nb else None
        for a, v in (c_nb or {}).items():
            per_arm_nb[a].append(v["shift_pp"])
        out.setdefault("within", []).append(
            {"dir": os.path.basename(d.rstrip("/")), "complete": complete(d),
             "freerun_tok_s": free, "hardcap_tok_s": cap,
             "shift_pp": {a: v["shift_pp"] for a, v in c.items()},
             "log_delta": {a: v["log_delta"] for a, v in c.items()}})
    if per_arm:
        print(f"\n  across {len(dirs)} invocation(s):")
        out["within_summary"] = {}
        for a, vs in sorted(per_arm.items(), key=lambda kv: -st.mean(kv[1])):
            m, lo, hi, n = interval(vs)
            rng = "" if lo is None else f"  [{lo:+.2f}, {hi:+.2f}]"
            lm = st.mean(per_arm_log[a])
            pct = 100.0 * math.expm1(lm)
            print(f"    {a:<22} {m:+8.2f} pp{rng}   n={n}"
                  f"   (log contrast {pct:+.2f} %)")
            out["within_summary"][a] = {
                "mean_pp": m, "lo": lo, "hi": hi, "n": n,
                "estimand": "absolute change in percentage points of the "
                            "arm-vs-baseline figure",
                "log_contrast_pct": pct}
    if per_arm_nb:
        print("\n  row-boundary EXCLUDED -- the arm-run that opened each row is "
              "dropped,\n  because its predecessor is in the previous row and the "
              "square does not\n  balance that adjacency. One per arm per session.")
        out["within_summary_no_boundary"] = {}
        for a, vs in sorted(per_arm_nb.items(), key=lambda kv: -st.mean(kv[1])):
            m, lo, hi, n = interval(vs)
            rng = "" if lo is None else f"  [{lo:+.2f}, {hi:+.2f}]"
            print(f"    {a:<22} {m:+8.2f} pp{rng}   n={n}")
            out["within_summary_no_boundary"][a] = {
                "mean_pp": m, "lo": lo, "hi": hi, "n": n,
                "estimand": "absolute change in percentage points of the "
                            "arm-vs-baseline figure, row openers excluded"}


def report_crossover(halves, out):
    print("\n=== the crossover: the session is the resampling unit ===")
    by_session = defaultdict(dict)
    order = {}
    for name, mode, d in halves:
        s = session_of(name)
        if mode in by_session[s]:
            sys.exit(f"two directories claim to be session {s}'s {mode} half: "
                     f"{os.path.basename(by_session[s][mode].rstrip('/'))} and "
                     f"{name}. Silently keeping the last would publish an "
                     f"interval over a set nobody chose.")
        by_session[s][mode] = d
        order.setdefault(s, []).append((name, mode))
    for s in order:
        order[s] = [m for _, m in sorted(order[s])]
    usable, dropped = [], []
    for s, halves_ in sorted(by_session.items()):
        if set(halves_) == {"freerun", "hardcap"} and all(
                complete(v) for v in halves_.values()):
            usable.append(s)
        else:
            dropped.append((s, sorted(halves_), [complete(v) for v in halves_.values()]))
    print(f"  sessions with both halves complete: {len(usable)} of {len(by_session)}")
    for s, present, ok in dropped:
        print(f"    dropped {s}: halves {present}, complete {ok}")
    # FAIL CLOSED, as the within-invocation path already does. This printed the
    # dropped sessions and carried on, so an eight-session crossover could
    # become a seven- or six-session one and still be published -- silently
    # turning a design into a subset of itself. The existing data is complete,
    # so nothing here changes; what changes is what happens the next time it is
    # not.
    if dropped and not ALLOW_INCOMPLETE:
        sys.exit(f"{len(dropped)} of {len(by_session)} crossover session(s) do "
                 f"not have both halves complete: "
                 f"{[s for s, _, _ in dropped]}. Pass --allow-incomplete to "
                 f"analyse the rest anyway, and do not publish the result.")
    if not usable:
        return

    if dropped:
        out["crossover_publishable"] = False
        out["crossover_dropped_sessions"] = [s for s, _, _ in dropped]
    deltas = defaultdict(dict)
    first_mode = {}
    for s in usable:
        f, c = by_session[s]["freerun"], by_session[s]["hardcap"]
        # which ran first, from the manifests' own timestamps
        tf, tc = manifest(f).get("created", ""), manifest(c).get("created", "")
        # `tf < tc else hardcap` called a session hard-cap-first whenever a
        # stamp was missing or the two were equal, which is a guess wearing the
        # clothes of a measurement. An unknown order is recorded as unknown and
        # left out of the order split, not assigned to one side.
        if not tf or not tc or tf == tc:
            first_mode[s] = None
        else:
            first_mode[s] = "freerun" if tf < tc else "hardcap"
        # The two halves must be the same experiment run twice. A mismatched
        # model, binary or prompt set makes the difference between them
        # something other than the mode, and `contrast()` would have quietly
        # reported it as the mode anyway.
        mf, mc = manifest(f), manifest(c)
        for field in ("target_sha256", "draft_sha256", "prompt_set", "n_prompts",
                      "max_tokens", "think", "ctx", "fit_target", "concurrency",
                      "server_loaded_commit"):
            if mf.get(field) != mc.get(field):
                sys.exit(f"session {s}: the two halves differ in {field} "
                         f"({mf.get(field)!r} vs {mc.get(field)!r}), so their "
                         f"difference is not the mode alone")
        fr = {a: pooled(f, a)["tok_s"] for a in arms_of(f) if pooled(f, a)}
        cp = {a: pooled(c, a)["tok_s"] for a in arms_of(c) if pooled(c, a)}
        # `contrast()` iterates the free arms and skips any without a capped
        # twin, so an arm missing from one half used to shrink the comparison
        # without saying so
        if set(fr) != set(cp):
            only_f, only_c = sorted(set(fr) - set(cp)), sorted(set(cp) - set(fr))
            sys.exit(f"session {s}: the halves do not carry the same arms "
                     f"(only free: {only_f}, only capped: {only_c})")
        for a, v in contrast(fr, cp).items():
            deltas[a][s] = v

    print(f"\n  {'configuration':<22} {'mean shift':>11} {'95 % t over sessions':>24}"
          f" {'sessions':>9}")
    out["crossover"] = {}
    for a in sorted(deltas, key=lambda k: -st.mean(
            [v["shift_pp"] for v in deltas[k].values()])):
        shifts = [v["shift_pp"] for v in deltas[a].values()]
        m, lo, hi, n = interval(shifts)
        rng = "" if lo is None else f"[{lo:+.2f} pp, {hi:+.2f} pp]"
        # the same sessions under the other estimand, so the reader can see
        # where the choice of estimand changes the answer and where it does not
        lg = [v["log_delta"] for v in deltas[a].values()]
        lm, llo, lhi, _ = interval(lg)
        pct = 100.0 * math.expm1(lm)
        lrng = "" if llo is None else (f"[{100 * math.expm1(llo):+.2f} %, "
                                       f"{100 * math.expm1(lhi):+.2f} %]")
        print(f"  {a:<22} {m:+10.2f} pp {rng:>24} {n:>9}"
              f"   (log contrast {pct:+.2f} % {lrng})")
        out["crossover"][a] = {"mean_shift_pp": m, "lo": lo, "hi": hi,
                               "sessions": n,
                               "estimand": "absolute change in percentage "
                                           "points of the arm-vs-baseline figure",
                               "log_contrast_pct": pct,
                               "log_contrast_lo_pct":
                                   None if llo is None else 100 * math.expm1(llo),
                               "log_contrast_hi_pct":
                                   None if lhi is None else 100 * math.expm1(lhi),
                               "per_session": {s: v["shift_pp"]
                                               for s, v in deltas[a].items()},
                               "log_delta": {s: v["log_delta"]
                                             for s, v in deltas[a].items()}}

    print("\n  does the order matter? the same shift, split by which mode ran first")
    print(f"  {'configuration':<22} {'freerun first':>14} {'hard cap first':>15}"
          f" {'difference':>12}")
    out["order_contrast"] = {}
    for a in sorted(deltas):
        ff = [v["shift_pp"] for s, v in deltas[a].items() if first_mode[s] == "freerun"]
        hf = [v["shift_pp"] for s, v in deltas[a].items() if first_mode[s] == "hardcap"]
        if not ff or not hf:
            continue
        print(f"  {a:<22} {st.mean(ff):+13.2f} {st.mean(hf):+14.2f} "
              f"{st.mean(ff) - st.mean(hf):+11.2f} pp")
        out["order_contrast"][a] = {"freerun_first_pp": st.mean(ff),
                                    "hardcap_first_pp": st.mean(hf),
                                    "difference_pp": st.mean(ff) - st.mean(hf),
                                    "n_freerun_first": len(ff),
                                    "n_hardcap_first": len(hf)}
    print("\n  A large difference in that last column is an order or state effect,"
          "\n  not a mode effect. The balanced design is what makes it visible.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--json", help="write the numbers here")
    ap.add_argument("--allow-incomplete", action="store_true",
                    help="analyse runs the driver did not attest; the result "
                         "is exploratory and must not be published")
    args = ap.parse_args()
    dirs = [d for d in args.dirs if os.path.isdir(d)]
    missing = [d for d in args.dirs if not os.path.isdir(d)]
    if missing:
        # a mistyped path used to vanish into the filter and the run just
        # analysed fewer directories than the caller asked for
        sys.exit(f"not a directory: {' '.join(missing[:3])}")
    if not dirs:
        sys.exit("no run directories")
    within, halves = classify(dirs)
    out: dict = {}
    if within:
        report_within(within, out)
    if halves:
        report_crossover(halves, out)
    if not within and not halves:
        sys.exit("nothing recognisable in those directories")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=1, sort_keys=True)
        print(f"\n  wrote {args.json}")


if __name__ == "__main__":
    main()
