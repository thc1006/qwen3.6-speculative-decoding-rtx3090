#!/usr/bin/env python3
"""Plan Z layer A: bandwidth against thermal state, from a pilot directory.

    python analysis/layer_a_slope.py DIR
    python analysis/layer_a_slope.py --check DIR   # and exit 1 if a premise failed

Reads what `bench/run_z_layer_a.sh` wrote and answers, in order:

  1. WAS IT A MEASUREMENT OF MEMORY. Bandwidth against the locked core clock, from
     the calibration. Flat means the copy is memory-bound and the core clock buys
     no throughput; a rise means the measurement is partly of the SMs and the
     slope below is not what it says.
  2. DID THE CLOCKS HOLD. Both, for every sample, with the spread. `-lgc` is a
     request and the power cap outranks it.
  3. THE SLOPE, per phase. Ordinary least squares of bandwidth on temperature with
     a Student t interval, and the lag-one autocorrelation of the residuals beside
     it, because the ascending phase's windows are consecutive and an OLS interval
     understates uncertainty when they are.
  4. TEMPERATURE OR TIME. The two phases cross the same temperatures in opposite
     directions at different elapsed times. Compared in one-degree bins where both
     have samples: if bandwidth is a function of thermal state they agree, and if it
     is a function of time since the load began they do not. That is layer A's
     control and the reason the descending phase exists.
  5. THE NOISE FLOOR, from the windows at the thermal plateau, because a slope
     smaller than the scatter it is fitted through is not a slope.
  6. THE PRE-REGISTERED READING. R is the bandwidth change across the full thermal
     range traversed, as a fraction of the cold value. Plan Z: if R is below the
     3.93 % step ERRATA A16 is about, memory thermal state cannot produce that step,
     because reaching it would need an elasticity above one -- the fully
     memory-bound ceiling -- AND a divergence equal to the whole range, while the
     divergence actually observed between A16's two levels is under two degrees and
     has the wrong sign.

The join is on wall clock, through the one parser that reads every spelling the
traces here carry: `bench/check_telemetry_cover.py`. A bandwidth window is matched
to the telemetry samples inside it.
"""
from __future__ import annotations

import csv
import datetime as dt
import importlib.util
import math
import pathlib
import statistics as st
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
from paired_blocks import t_critical_975                          # noqa: E402

A16_STEP_PCT = 3.93
# the core-temperature divergence between A16's two levels, from
# `analysis/plan_z_power.py`, which derives it from run T4's own snapshots. The
# faster level is the HOTTER one, so this is used as a magnitude and its sign is
# against the hypothesis rather than for it.
A16_DIVERGENCE_C = 1.67
# the control needs temperatures both phases visited; fewer than this and it did
# not run, whatever else the run produced
MIN_SHARED_BINS = 3
# the elasticity ceiling: a workload cannot lose more than one per cent of its rate
# per one per cent of bandwidth unless it is more than fully memory-bound
E_CEILING = 1.0


def _load(rel: str, name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _tz_of(trace: pathlib.Path) -> dt.timezone:
    """The zone the trace was written in, from its own wall column.

    Not the local zone: `bench/check_telemetry_cover.py` records that a verifier
    whose verdict depends on `TZ` is not a verifier, and this reads both sides.
    """
    with trace.open(encoding="utf-8") as fh:
        rd = csv.DictReader(fh)
        for row in rd:
            v = (row.get("wall_iso") or "").strip()
            if v:
                try:
                    off = dt.datetime.fromisoformat(v).utcoffset()
                    if off is not None:
                        return dt.timezone(off)
                except ValueError:
                    pass
            break
    return dt.timezone(dt.timedelta(hours=8))


def telemetry(d: pathlib.Path):
    traces = sorted(d.glob("gpu_telemetry_*.csv"))
    if len(traces) != 1:
        sys.exit(f"{len(traces)} telemetry traces in {d}, expected one")
    ctc = _load("bench/check_telemetry_cover.py", "ctc_a")
    tz = _tz_of(traces[0])
    out = []
    with traces[0].open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            row = {k.strip(): (v or "").strip() for k, v in row.items() if k}
            t = ctc._stamp(row, tz)
            if t is None:
                continue
            def num(key, *alts):
                for k in (key,) + alts:
                    if k in row and row[k]:
                        try:
                            return float(row[k].split()[0])
                        except ValueError:
                            pass
                return None
            out.append({"t": t, "temp": num("temp_c", "temp", "temperature.gpu"),
                        "sm": num("sm_mhz", "clk_sm"),
                        "mem": num("mem_mhz", "clk_mem"),
                        "power": num("power_w", "pwr"),
                        "throttle": row.get("throttle_active", "")})
    if not out:
        sys.exit(f"no telemetry row in {traces[0].name} parsed")
    return traces[0].name, tz, out


def windows(d: pathlib.Path, phase: str, tz, tel):
    p = d / f"{phase}.csv"
    if not p.is_file():
        sys.exit(f"no {phase}.csv in {d}")
    ctc = _load("bench/check_telemetry_cover.py", "ctc_b")
    rows = []
    with p.open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            t0 = ctc._stamp({"wall_iso": r["wall_iso"]}, tz)
            if t0 is None:
                sys.exit(f"{phase}: cannot read the stamp {r['wall_iso']!r}")
            dur = float(r["window_s"])
            inside = [s for s in tel if t0 <= s["t"] <= t0 + dur]
            rows.append({
                "t0": t0, "elapsed": float(r["elapsed_s"]), "dur": dur,
                "bw": float(r["gbytes_per_s"]), "uuid": r["gpu_uuid"],
                "n_samples": len(inside),
                "temp": st.fmean([s["temp"] for s in inside if s["temp"] is not None])
                        if any(s["temp"] is not None for s in inside) else None,
                "power": st.fmean([s["power"] for s in inside if s["power"] is not None])
                         if any(s["power"] is not None for s in inside) else None,
            })
    return rows


def ols(xs, ys):
    """Slope, intercept, a t interval on the slope, and the residuals' lag-one
    autocorrelation. No scipy: this has to run in the claims job."""
    n = len(xs)
    if n < 3 or len(set(xs)) < 2:
        return None
    mx, my = st.fmean(xs), st.fmean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    a = my - b * mx
    res = [y - (a + b * x) for x, y in zip(xs, ys)]
    s2 = sum(r * r for r in res) / (n - 2)
    se = math.sqrt(s2 / sxx)
    half = t_critical_975(n - 2) * se
    num = sum(res[i] * res[i + 1] for i in range(n - 1))
    den = sum(r * r for r in res)
    return {"slope": b, "intercept": a, "half": half, "n": n,
            "resid_sd": math.sqrt(s2), "lag1": num / den if den else float("nan")}


def main() -> int:
    argv = sys.argv[1:]
    check_only = "--check" in argv
    argv = [a for a in argv if a != "--check"]
    if len(argv) != 1:
        print(__doc__, file=sys.stderr)
        return 2
    d = pathlib.Path(argv[0])
    if not d.is_dir():
        print(f"no such directory: {d}", file=sys.stderr)
        return 2
    print((d / "manifest.txt").read_text(encoding="utf-8").rstrip()
          if (d / "manifest.txt").is_file() else "(no manifest)")
    tname, tz, tel = telemetry(d)

    print("\n1. was it a measurement of memory")
    cal = [l.split(",") for l in (d / "calibrate.csv").read_text().split() if l]
    vals = {int(a): float(b) for a, b in cal}
    for c in sorted(vals):
        print(f"   core {c:5d} MHz -> {vals[c]:7.1f} GB/s")
    spread = 100.0 * (max(vals.values()) - min(vals.values())) / max(vals.values())
    print(f"   spread across the calibrated clocks: {spread:.3f} % "
          f"-> {'memory bound' if spread < 1.0 else 'NOT clearly memory bound'}")

    print("\n2. did the clocks hold")
    for k in ("sm", "mem"):
        v = [s[k] for s in tel if s[k] is not None]
        print(f"   {k:4s} min {min(v):.0f} max {max(v):.0f} "
              f"sd {st.stdev(v) if len(v) > 1 else 0.0:.3f}  n={len(v)}")
    thr = {}
    for s in tel:
        t = s["throttle"]
        if t and t not in ("0x0000000000000000",):
            thr[t] = thr.get(t, 0) + 1
    for t, n in sorted(thr.items(), key=lambda x: -x[1])[:4]:
        print(f"   throttle {t} in {n} of {len(tel)} samples")

    phases = {}
    for phase in ("ascending", "descending"):
        w = [r for r in windows(d, phase, tz, tel) if r["temp"] is not None]
        if not w:
            sys.exit(f"{phase}: no window joined a telemetry sample. The join is on "
                     f"wall clock and one side did not line up.")
        phases[phase] = w

    print("\n3. the slope, per phase")
    for phase, w in phases.items():
        temps = [r["temp"] for r in w]
        bws = [r["bw"] for r in w]
        fit = ols(temps, bws)
        print(f"   {phase}: {len(w)} windows, "
              f"{min(r['n_samples'] for r in w)}-{max(r['n_samples'] for r in w)} "
              f"telemetry samples each")
        print(f"     temperature {min(temps):.1f} to {max(temps):.1f} C "
              f"(range {max(temps) - min(temps):.1f})")
        print(f"     bandwidth   {min(bws):.1f} to {max(bws):.1f} GB/s "
              f"(range {100 * (max(bws) - min(bws)) / max(bws):.3f} %)")
        if fit:
            per_c = 100.0 * fit["slope"] / st.fmean(bws)
            half_c = 100.0 * fit["half"] / st.fmean(bws)
            print(f"     slope {fit['slope']:+.3f} GB/s per C "
                  f"= {per_c:+.4f} % per C, 95 % +-{half_c:.4f}")
            print(f"     residual sd {fit['resid_sd']:.3f} GB/s, "
                  f"lag-1 autocorrelation {fit['lag1']:+.2f}")
            phases[phase] = (w, fit, per_c, half_c)
        else:
            phases[phase] = (w, None, None, None)

    print("\n4. temperature or time: the two phases in one-degree bins")
    bins = {}
    for phase, (w, *_rest) in phases.items():
        for r in w:
            bins.setdefault(round(r["temp"]), {}).setdefault(phase, []).append(r["bw"])
    both = sorted(b for b, v in bins.items() if len(v) == 2)
    mean_off = None
    if not both:
        print("   no temperature is visited by both phases, so this cannot separate"
              " temperature from time. The descending phase did not reach the"
              " ascending one's range.")
    else:
        worst = 0.0
        for b in both:
            a = st.fmean(bins[b]["ascending"])
            c = st.fmean(bins[b]["descending"])
            diff = 100.0 * (c - a) / a
            worst = max(worst, abs(diff))
            print(f"   {b:3d} C  ascending {a:7.1f}  descending {c:7.1f}  "
                  f"{diff:+.3f} %")
        mean_off = st.fmean(
            [100.0 * (st.fmean(bins[b]["descending"]) - st.fmean(bins[b]["ascending"]))
             / st.fmean(bins[b]["ascending"]) for b in both])
        print(f"   largest disagreement {worst:.3f} %, mean offset "
              f"{mean_off:+.3f} % over {len(both)} shared bins")
        # What this offset can and cannot be. The two phases are two separate
        # INVOCATIONS of the load, so a constant offset between them is exactly what
        # run-to-run variation looks like -- and a constant offset is what appears
        # across every bin, so the bins are not independent evidence. Measured on
        # 2026-10-02: seven invocations at a flat plateau, same clocks, same
        # temperature, gave a run-to-run spread of 0.013 %, against a
        # window-to-window spread within one invocation of 0.012 %. An offset of
        # this size is therefore not distinguishable from two runs differing.
        #
        # A balanced test the same day also refuted the obvious instrument
        # explanation: five-second and ten-second windows differ by 0.006 % at a
        # flat plateau, inside the scatter, so the two phases' different window
        # lengths do not account for it either.
        print("   NOT attributable from one pair: the phases are two invocations, a")
        print("   constant offset is what run-to-run variation looks like, and seven")
        print("   invocations at a flat plateau spread 0.013 % -- about this size.")
        print("   Attribution needs replicated pairs, or both phases in one process.")

    print("\n5. the noise floor, at the thermal plateau")
    asc = phases["ascending"][0]
    hot = max(r["temp"] for r in asc)
    plateau = [r["bw"] for r in asc if r["temp"] >= hot - 1.0]
    if len(plateau) > 2:
        print(f"   {len(plateau)} windows within one degree of {hot:.1f} C: "
              f"mean {st.fmean(plateau):.1f}, sd {st.stdev(plateau):.3f} GB/s "
              f"= {100 * st.stdev(plateau) / st.fmean(plateau):.4f} %")
    else:
        print(f"   only {len(plateau)} windows at the plateau; not a floor")

    print("\n6. the pre-registered reading")
    allw = [r for w in (phases[p][0] for p in phases) for r in w]
    temps = [r["temp"] for r in allw]
    bws = [r["bw"] for r in allw]
    lo, hi = min(temps), max(temps)
    cold = st.fmean([r["bw"] for r in allw if r["temp"] <= lo + 1.0])
    hot_bw = st.fmean([r["bw"] for r in allw if r["temp"] >= hi - 1.0])
    R = 100.0 * (cold - hot_bw) / cold
    pooled = ols(temps, bws)
    mean_bw = st.fmean(bws)
    B = 100.0 * pooled["slope"] / mean_bw if pooled else 0.0
    B_hi = abs(B) + (100.0 * pooled["half"] / mean_bw if pooled else 0.0)
    print(f"   thermal range traversed: {lo:.1f} to {hi:.1f} C = {hi - lo:.1f}")
    print(f"   bandwidth cold {cold:.1f} -> hot {hot_bw:.1f} GB/s, R = {R:+.3f} %")
    print(f"   pooled slope B = {B:+.5f} % per C, 95 % upper bound on |B| "
          f"= {B_hi:.5f}")
    print(f"   plan Z excludes the hypothesis for A16's step if R is below "
          f"{A16_STEP_PCT} %: {'EXCLUDED' if abs(R) < A16_STEP_PCT else 'NOT excluded'}")
    # Against A16's OWN divergence rather than an invented one. Dividing the step by
    # a near-zero B gives a number like 70,000 C, which is arithmetic rather than an
    # argument; the question a reader has is what sensitivity the step would need.
    need_B = A16_STEP_PCT / A16_DIVERGENCE_C
    print(f"   to make {A16_STEP_PCT} % from A16's own {A16_DIVERGENCE_C} C "
          f"divergence, at the memory-bound elasticity ceiling of {E_CEILING:.0f},")
    print(f"   the sensitivity would have to be {need_B:.2f} % per C, which is "
          f"{need_B / B_hi:,.0f}x the upper bound here")
    print(f"   or, at that upper bound, the memory would have to diverge "
          f"{A16_STEP_PCT / B_hi:,.0f} C between the two levels")
    floor = (100.0 * st.stdev(plateau) / st.fmean(plateau)
             if len(plateau) > 2 else float("nan"))
    if floor == floor:
        print(f"   and {A16_STEP_PCT} % is {A16_STEP_PCT / floor:,.0f} noise floors "
              f"away from zero")

    # resolved, so a relative DIR on the command line does not break the
    # relative_to below -- the one in the message, not the lookup
    readme = d.resolve().parent.parent / "README.md"
    checks_readme = None
    if readme.is_file():
        print(f"\n6b. {readme.name} beside the data, row by row")
        # By ROW LABEL, not by presence anywhere in the file. Presence is weak where
        # a figure is quoted twice -- R is, once in the table and once in the prose --
        # and forbidding the values one step away produced a FALSE one here, because
        # the run-to-run spread named in the limits is 0.013 % and R is 0.014 %. Two
        # unrelated quantities can be numerically adjacent; a row cannot be mistaken
        # for another row. `analysis/rederive_run_y.py` compares by header for the
        # same reason.
        text = (readme.read_text(encoding="utf-8")
                .replace("**", "").replace("\u00b0C", "C"))
        # No row label may contain a pipe, escaped or not: a `\|` in a cell splits
        # the row into three on any naive split, and the two-cell filter then drops
        # it silently. The label says "its magnitude" rather than "|B|".
        rows = {}
        for line in text.splitlines():
            if not line.strip().startswith("|"):
                continue
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if len(cells) == 2 and not cells[0].startswith("---"):
                rows[cells[0]] = cells[1]
        want_rows = {
            "the thermal range": f"{lo:.1f} to {hi:.1f} C",
            "R, the bandwidth change across it": f"{abs(R):.3f} %",
            "the slope B": f"{B:+.5f} % per C".replace("+", ""),
            "the 95 % upper bound on its magnitude": f"{B_hi:.5f} % per C",
            f"the noise floor, {len(plateau)} windows at the plateau":
                f"{floor:.4f} %",
            "the shared bins the control used": f"{len(both)}",
            "the mean phase offset in them":
                (f"{mean_off:+.3f} %".replace("+", "") if mean_off is not None
                 else None),
        }
        bad = []
        for k, v in want_rows.items():
            if v is None:
                continue
            got = rows.get(k)
            ok = got == v
            print(f"    {'ok     ' if ok else 'WRONG  '} {k}: {v}"
                  + ("" if ok else f"  (the row says {got!r})"))
            if not ok:
                bad.append(k)
        # and the two the prose carries, which are distinctive enough to find
        for what, v in (("the required sensitivity", f"{need_B:.2f} % per C"),
                        ("the ratio to the bound", f"{need_B / B_hi:,.0f}")):
            ok = v in text
            print(f"    {'ok     ' if ok else 'MISSING'} {what}: {v}")
            if not ok:
                bad.append(what)
        checks_readme = not bad

    print("\n7. the premises, as conditions")
    checks = []
    for k, want in (("sm", None), ("mem", None)):
        v = [x[k] for x in tel if x[k] is not None]
        sd = st.stdev(v) if len(v) > 1 else 0.0
        checks.append((f"the {k} clock held for all {len(v)} samples", sd == 0.0,
                       f"sd {sd:.3f}"))
    checks.append(("the measurement was memory bound", spread < 1.0,
                   f"{spread:.3f} % across the calibrated clocks"))
    if checks_readme is not None:
        checks.append(("the round's README quotes the figures this derives",
                       checks_readme, "every cell" if checks_readme
                       else "see 6b above"))
    checks.append((f"the control ran: at least {MIN_SHARED_BINS} temperatures "
                   f"visited by both phases", len(both) >= MIN_SHARED_BINS,
                   f"{len(both)} shared bins. Two phases that share no temperature "
                   f"cannot separate hotter from running longer, which is the only "
                   f"reason the second one exists"))
    failed = 0
    for what, ok, detail in checks:
        print(f"   {'ok     ' if ok else 'FAILED '} {what} ({detail})")
        failed += not ok
    if check_only:
        if failed:
            print(f"\n{failed} premise(s) failed: this is not a layer A run",
                  file=sys.stderr)
            return 1
        print("\nevery premise holds")
    return 0


if __name__ == "__main__":
    sys.exit(main())
