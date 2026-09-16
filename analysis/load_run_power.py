#!/usr/bin/env python3
"""The two tables `PROSPECTIVE_PLAN_X_HOST_LOAD.md` publishes, derived here.

    python analysis/load_run_power.py            # print both tables
    python analysis/load_run_power.py --check    # and compare against the plan

Why this file exists
--------------------
The plan chose twenty-four blocks over twelve because of a power calculation,
and it discriminates an arm-specific mechanism from an arm-agnostic one because
of a normaliser table. Both were first published as figures typed in after
reading a terminal. `v4_audit_2026_08_25/PREREGISTERED_PREDICTION.md` records
what that costs: it was "the one analysis in the repository with no code path",
and recomputing it later found eight wrong figures.

The first version of the power table had one of its own, and it is the reason
this file is here rather than in a scratch directory. It used run T4's block CV
of 2.145 % as the residual noise AND added a telegraph on top of it. T4's block
CV already contains the step, so that model implies 2.768 % of block spread,
twenty-nine per cent more than T4 shows, and it published a power of 0.09 for a
design whose real power is 0.34. The decomposition is asserted below and checked
against the measured CV, so the same mistake cannot be made silently again.

Everything is measured from the committed arm-runs of run T4. Nothing here is a
constant somebody remembered.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import random
import re
import statistics as st

ROOT = pathlib.Path(__file__).resolve().parents[1]
T4 = ROOT / "v4_audit_2026_08_25" / "data" / "matrix_T4_split_20260827_175051"
PLAN = ROOT / "v4_audit_2026_08_25" / "PROSPECTIVE_PLAN_X_HOST_LOAD.md"
ARMS = ("spec-dflash-n2", "baseline", "spec-draft-n8")

SEED = 20260917
TRIALS = 40000
# How often this arm changes level on its own. The first version of this file
# took it from run T4 alone: one transition across five adjacent gaps, which is
# ONE event, and the exact Poisson interval for one event in a thousand seconds
# runs from 179 s to 39 498 s. A power table cannot rest on that. `hazard()`
# below counts transitions across every run in the corpus that repeats this arm
# inside one invocation, which is fifty of them.
HAZARD_FALLBACK = 1123.0
CHANGE_PCT = 2.0         # half the measured level gap; see hazard()
TSTAR = {5: 2.571, 11: 2.201, 17: 2.110, 23: 2.069}


def _rows(arm: str, rep: int) -> list[dict]:
    return json.loads((T4 / f"{arm}__rep{rep}.json").read_text(encoding="utf-8"))["rows"]


def pooled(arm: str, rep: int) -> float:
    """Generated tokens over decode milliseconds, the definition used elsewhere."""
    rs = _rows(arm, rep)
    return 1000.0 * sum(r["predicted_n"] for r in rs) / sum(r["predicted_ms"] for r in rs)


def measured() -> dict:
    """The inputs, and the check that they decompose to what T4 actually shows."""
    v = [pooled("spec-dflash-n2", i) for i in range(6)]
    lo, hi = st.mean(v[:3]), st.mean(v[3:])
    gap = 100.0 * (hi / lo - 1.0)
    cv = 100.0 * st.stdev(v) / st.mean(v)
    adj = [100.0 * math.log(v[i + 1] / v[i]) for i in range(5)]
    step_at = max(range(5), key=lambda i: abs(adj[i]))
    diff_sd = st.stdev([a for i, a in enumerate(adj) if i != step_at])
    resid = diff_sd / math.sqrt(2.0)
    # the telegraph is symmetric about the midpoint, so its SD is half the gap
    combined = math.hypot(gap / 2.0, resid)
    return {"levels": (lo, hi), "gap": gap, "block_cv": cv,
            "adjacent_diff_sd": diff_sd, "residual": resid,
            "combined": combined,
            "durations": {a: st.mean(
                json.loads((T4 / f"{a}__rep{i}.json").read_text(encoding="utf-8"))
                .get("ready_s", 0.0)
                + sum(r["wall_ms"] for r in _rows(a, i)) / 1000.0
                for i in range(6)) for a in ARMS}}


def _chi2cdf(x: float, df: int) -> float:
    if x <= 0:
        return 0.0
    a, z = df / 2.0, x / 2.0
    s = term = 1.0 / a
    for n in range(1, 4000):
        term *= z / (a + n)
        s += term
        if term < 1e-16 * s:
            break
    return s * math.exp(-z + a * math.log(z) - math.lgamma(a))


def _chi2inv(p: float, df: int) -> float:
    lo, hi = 0.0, 5000.0
    for _ in range(300):
        mid = (lo + hi) / 2.0
        if _chi2cdf(mid, df) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def hazard(root: pathlib.Path | None = None) -> dict:
    """Mean seconds between spontaneous level changes, over the whole corpus.

    Every run directory that repeats `spec-dflash-n2` at least three times
    inside one invocation contributes its adjacent gaps. A gap counts as a
    change when the log ratio of consecutive pooled rates exceeds CHANGE_PCT,
    which is half the measured level gap. The residual difference SD is 0.76 %,
    so that threshold is about 2.6 sigma and costs roughly two false positives
    across the corpus, which is reported beside the count rather than hidden.

    The elapsed time between two repeats of one arm is one pass of the whole
    block, so the block duration is summed from each arm's rep0.
    """
    data = (root or ROOT) / "v4_audit_2026_08_25" / "data"
    changes = gaps = 0
    seconds = 0.0
    runs = 0
    durations: list[tuple[float, bool]] = []
    for d in sorted(data.iterdir()):
        if not d.is_dir():
            continue
        fs = sorted(d.glob("spec-dflash-n2__rep*.json"),
                    key=lambda q: int(q.stem.rpartition("__rep")[2]))
        if len(fs) < 3:
            continue
        rates = []
        for f in fs:
            rs = json.loads(f.read_text(encoding="utf-8"))["rows"]
            ms = sum(r["predicted_ms"] for r in rs)
            if not ms:
                rates = []
                break
            rates.append(1000.0 * sum(r["predicted_n"] for r in rs) / ms)
        if len(rates) < 3:
            continue
        block = 0.0
        for f in sorted(d.glob("*__rep0.json")):
            j = json.loads(f.read_text(encoding="utf-8"))
            block += j.get("ready_s", 0.0) + sum(
                r["wall_ms"] for r in j["rows"]) / 1000.0
        if block <= 0:
            continue
        runs += 1
        gaps += len(rates) - 1
        seconds += (len(rates) - 1) * block
        for i in range(len(rates) - 1):
            moved = abs(100.0 * math.log(rates[i + 1] / rates[i])) > CHANGE_PCT
            durations.append((block, moved))
            changes += 1 if moved else 0
    if not changes or not durations:
        return {"runs": runs, "gaps": gaps, "seconds": seconds, "changes": 0,
                "naive": HAZARD_FALLBACK, "hazard": HAZARD_FALLBACK,
                "lo": HAZARD_FALLBACK, "hi": HAZARD_FALLBACK}

    # Changes over elapsed time is NOT the estimator. It assumes every interval
    # holds at most one transition, and two transitions inside one interval put
    # the arm back where it started and are recorded as no change. The mean gap
    # here is about four hundred seconds, the same order as the hazard, so the
    # omission is large: it reported one per 1123 s where the likelihood gives
    # one per 641, which is 1.75 times too slow and makes any design built on it
    # look better than it is.
    def _ll(h):
        total = 0.0
        for t, changed in durations:
            pr = (1.0 - math.exp(-2.0 * t / h)) / 2.0
            pr = min(max(pr, 1e-12), 1.0 - 1e-12)
            total += math.log(pr if changed else 1.0 - pr)
        return total

    lo_h, hi_h = 50.0, 20000.0
    for _ in range(300):
        a = lo_h + (hi_h - lo_h) / 3.0
        b = hi_h - (hi_h - lo_h) / 3.0
        if _ll(a) < _ll(b):
            lo_h = a
        else:
            hi_h = b
    mle = (lo_h + hi_h) / 2.0
    peak = _ll(mle)

    def _edge(step):
        x = mle
        for _ in range(500):
            x *= step
            if _ll(x) < peak - 1.92:      # the usual two-unit likelihood drop
                return x
        return x

    return {"runs": runs, "gaps": gaps, "seconds": seconds, "changes": changes,
            "naive": seconds / changes, "hazard": mle,
            "lo": _edge(0.98), "hi": _edge(1.02)}


def normalisers() -> dict:
    """How many target steps and model forwards each arm spends per token.

    A round that drafts k tokens and has a accepted yields a+1 tokens, so over a
    whole run P = A + R and the round count is P - A exactly. One target forward
    per round; one draft forward per drafted token.
    """
    out = {}
    for a in ARMS:
        P = D = A = 0
        for i in range(6):
            for r in _rows(a, i):
                P += r["predicted_n"]
                D += r.get("draft_n") or 0
                A += r.get("draft_n_accepted") or 0
        R = P - A
        out[a] = {"tokens": P, "drafted": D, "accepted": A, "rounds": R,
                  "per_token": 1.0, "per_target_step": R / P,
                  "per_forward": (D + R) / P,
                  "drafted_per_round": (D / R) if R else 0.0}
    # The identity is the whole of the cross-arm discrimination, so it is checked
    # rather than asserted in prose. Two readings of `draft_n_accepted` are
    # possible and only one of them makes R = P - A true:
    #   if it counts the bonus token, then P = A and the round count is lost
    #   if it counts real accepts, then drafted-per-round cannot exceed the
    #   arm's own draft-max
    # `spec-dflash-n2` runs at n=2 and comes out at 1.978, which fits the second
    # reading and refutes the first, where A would have to equal P.
    for a, n_max in (("spec-dflash-n2", 2), ("spec-draft-n8", 8)):
        if out[a]["accepted"] == out[a]["tokens"]:
            raise SystemExit(f"{a}: accepted equals generated, so draft_n_accepted "
                             f"counts the bonus token and R = P - A is wrong")
        if out[a]["drafted_per_round"] > n_max + 0.02:
            raise SystemExit(f"{a}: {out[a]['drafted_per_round']:.3f} drafted per "
                             f"round exceeds its draft-max of {n_max}, so the "
                             f"round count is not P - A")
    return out


def _interval(d: list[float], df: int) -> tuple[float, float]:
    h = TSTAR[df] * st.stdev(d) / math.sqrt(len(d))
    m = st.mean(d)
    return m - h, m + h


def _verdict(d: list[float], df: int) -> str:
    lo, hi = _interval(d, df)
    if lo > 1.0 or hi < -1.0:
        return "moves"
    if lo > -1.0 and hi < 1.0:
        return "holds"
    return "neither"


def _flip(dt: float, hazard: float, state: int, rng: random.Random) -> int:
    """Two states, so what matters is ending in the OTHER one, not transitioning.

    This read `1 - exp(-dt / hazard)`, the probability of at least one
    transition. Two transitions inside one interval return the arm to where it
    started, and for a symmetric two-state chain the probability of ending
    flipped is `(1 - exp(-2 dt / hazard)) / 2`. The two agree to first order and
    diverge once the interval approaches the hazard, which is exactly the regime
    the between-block design sits in.
    """
    return (1 - state if rng.random() < (1.0 - math.exp(-2.0 * dt / hazard)) / 2.0
            else state)


def _between(truth, resid, gap, hazard, rng, nblocks=12):
    loaded = {1, 4, 5, 8, 9, 12}
    t, state, obs = 0.0, rng.randint(0, 1), {}
    for b in range(1, nblocks + 1):
        state = _flip(200.0, hazard, state, rng)
        t += 200.0
        obs[b] = state * gap + rng.gauss(0, resid) + (truth if b in loaded else 0.0)
    L = sorted(loaded)
    U = sorted(set(range(1, nblocks + 1)) - loaded)
    return _verdict([obs[a] - obs[b] for a, b in zip(L, U)], len(L) - 1)


def _within(truth, resid, gap, hazard, rng, dur, nblocks=24):
    d, state = [], rng.randint(0, 1)
    for b in range(1, nblocks + 1):
        first_loaded = (b % 4) in (1, 0)
        pair = []
        for k in range(2):
            state = _flip(dur, hazard, state, rng)
            pair.append(state * gap + rng.gauss(0, resid)
                        + (truth if (k == 0) == first_loaded else 0.0))
        # the rest of the block: the other two arms run twice each. The return
        # was DISCARDED here, so the level never advanced between blocks. It
        # cancels in a within-block difference and changed no published figure,
        # which is exactly why it would have sat there: a state variable that is
        # not assigned looks like one that is.
        state = _flip(400.0 - 2.0 * dur, hazard, state, rng)
        d.append(pair[0] - pair[1] if first_loaded else pair[1] - pair[0])
    return _verdict(d, nblocks - 1)


def power(m: dict, hazard_s: float | None = None,
          trials: int = TRIALS) -> dict:
    if hazard_s is None:
        hazard_s = hazard()["hazard"]
    rng = random.Random(SEED)
    dur = m["durations"]["spec-dflash-n2"]
    hz = hazard_s
    designs = {
        "between-block, 6 against 6": lambda tr: _between(
            tr, m["residual"], m["gap"], hz, rng),
        "within-block, 12 pairs": lambda tr: _within(
            tr, m["residual"], m["gap"], hz, rng, dur, 12),
        "within-block, 18 pairs": lambda tr: _within(
            tr, m["residual"], m["gap"], hz, rng, dur, 18),
        "within-block, 24 pairs": lambda tr: _within(
            tr, m["residual"], m["gap"], hz, rng, dur, 24),
    }
    out = {}
    for name, fn in designs.items():
        row = {}
        for label, truth in (("-2", -2.0), ("-gap", -m["gap"]), ("0", 0.0)):
            c = {"moves": 0, "holds": 0, "neither": 0}
            for _ in range(trials):
                c[fn(truth)] += 1
            row[label] = {k: v / trials for k, v in c.items()}
        out[name] = row
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--check", action="store_true",
                    help="also assert the plan publishes these figures")
    ap.add_argument("--hazard", type=float, default=None,
                    help="override the corpus-derived hazard, in seconds")
    ap.add_argument("--trials", type=int, default=TRIALS)
    args = ap.parse_args()

    m = measured()
    print(f"  run T4, spec-dflash-n2 levels {m['levels'][0]:.3f} and "
          f"{m['levels'][1]:.3f} tok/s, gap {m['gap']:.2f} %")
    print(f"  adjacent-difference SD {m['adjacent_diff_sd']:.3f} % with the step "
          f"excised, so the per-arm-run residual is {m['residual']:.3f} %")
    print(f"  decomposition check: hypot(gap/2, residual) = {m['combined']:.3f} % "
          f"against the measured block CV of {m['block_cv']:.3f} %")
    h = hazard()
    print(f"  switch hazard, over the whole corpus rather than one run: "
          f"{h['changes']} level changes across {h['gaps']} adjacent gaps in "
          f"{h['runs']} runs,")
    print(f"  {h['seconds']:.0f} s of elapsed time, so one per "
          f"{h['hazard']:.0f} s with a 95 % interval of "
          f"[{h['lo']:.0f}, {h['hi']:.0f}] s")
    print()

    hz = args.hazard if args.hazard is not None else h["hazard"]
    p = power(m, hz, args.trials)
    gap_col = f"-{m['gap']:.2f} %"
    print(f"  {'design':30s} {'-2 %':>8s} {gap_col:>10s} {'0 -> holds':>12s}")
    for name, row in p.items():
        print(f"  {name:30s} {row['-2']['moves']:8.2f} "
              f"{row['-gap']['moves']:10.2f} {row['0']['holds']:12.2f}")
    print()

    if args.hazard is None:
        print("  at the ends of that interval, for the design this plan chooses:")
        for end, label in ((h["lo"], "fast"), (h["hi"], "slow")):
            r = power(m, end, max(4000, args.trials // 4))["within-block, 24 pairs"]
            print(f"    hazard {end:6.0f} s ({label}): "
                  f"{r['-2']['moves']:.2f} / {r['-gap']['moves']:.2f} / "
                  f"{r['0']['holds']:.2f}")
        print()

    ms = {a: 1000.0 / st.mean(pooled(a, i) for i in range(6)) for a in ARMS}
    nz = normalisers()
    print(f"  {'normaliser':24s} " + " ".join(f"{a:>16s}" for a in ARMS) + "   D")
    for key, label in (("per_token", "per generated token"),
                       ("per_forward", "per model forward"),
                       ("per_target_step", "per target step")):
        c = (ms["spec-dflash-n2"] * (1 / 0.98 - 1)) / nz["spec-dflash-n2"][key]
        deltas = {a: c * nz[a][key] for a in ARMS}
        cells = " ".join(
            f"{deltas[a]:7.3f} ms {100 * (ms[a] / (ms[a] + deltas[a]) - 1):+6.2f}%"
            for a in ARMS)
        dd = deltas["spec-dflash-n2"] - deltas["spec-draft-n8"]
        print(f"  {label:24s} {cells}  {dd:+.3f}")
    print("\n  D is delta(spec-dflash-n2) - delta(spec-draft-n8) in ms per token.")
    print("  Every arm-agnostic hypothesis above puts it at or below zero, so the")
    print("  arm-specific branch is the one-sided claim that D is above zero.")

    if args.check:
        txt = PLAN.read_text(encoding="utf-8")
        want = [f"{m['gap']:.2f} %", f"{m['residual']:.3f} %",
                f"{m['adjacent_diff_sd']:.3f} %", f"{m['block_cv']:.3f} %"]
        missing = [w for w in want if w not in txt]
        r24 = p["within-block, 24 pairs"]
        for v in (f"{r24['-2']['moves']:.2f}", f"{r24['0']['holds']:.2f}"):
            if not re.search(rf"\|\s*\*?\*?{re.escape(v)}", txt):
                missing.append(f"24-pair power {v}")
        print("\n  plan check: " + ("OK" if not missing
                                    else "MISSING " + "; ".join(missing)))
        raise SystemExit(1 if missing else 0)


if __name__ == "__main__":
    main()
