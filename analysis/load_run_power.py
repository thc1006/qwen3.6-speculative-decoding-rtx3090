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
CV already contains the step, so that model implies 2.909 % of block spread,
thirty-six per cent more than T4 shows, and it published a power of 0.09 for a
design whose real power is 0.28. An earlier version of this paragraph said
2.768 % and twenty-nine, which is the same arithmetic done with A16's rounded
3.5 % gap rather than the measured 3.93 % this file uses everywhere else. The decomposition is asserted below and checked
against the measured CV, so the same mistake cannot be made silently again.

Everything is measured from the committed arm-runs of run T4. Nothing here is a
constant somebody remembered.
"""
from __future__ import annotations

import argparse
import datetime
import json
import math
import pathlib
import random
import statistics as st

ROOT = pathlib.Path(__file__).resolve().parents[1]
T4 = ROOT / "v4_audit_2026_08_25" / "data" / "matrix_T4_split_20260827_175051"
PLAN = ROOT / "v4_audit_2026_08_25" / "PROSPECTIVE_PLAN_X_HOST_LOAD.md"
ARMS = ("spec-dflash-n2", "baseline", "spec-draft-n8")
# each speculative arm's draft maximum, from its own server argv
NMAX = {"spec-dflash-n2": 2, "spec-draft-n8": 8}

SEED = 20260917
TRIALS = 40000
# How often this arm changes level on its own. The first version of this file
# took it from run T4 alone: one transition across five adjacent gaps, which is
# ONE event, and the exact Poisson interval for one event in a thousand seconds
# runs from 179 s to 39 498 s. A power table cannot rest on that. `hazard()`
# below counts transitions across every run in the corpus that repeats this arm
# inside one invocation, which is fifty of them.
HAZARD_FALLBACK = 1123.0
# The washout the plan mandates between the two members of a pair. It was
# not in the model at all, and it is not free: it lengthens the window in
# which the arm can change level inside a pair.
WASHOUT = 30.0
CHANGE_PCT = 2.0         # half the measured level gap; see hazard()
# df 17 to 23 is the range the plan permits once pairs may be dropped;
# the first version held 17 and 23 only and raised KeyError between them
TSTAR = {5: 2.571, 11: 2.201, 17: 2.110, 18: 2.101, 19: 2.093,
         20: 2.086, 21: 2.080, 22: 2.074, 23: 2.069}


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
    rest = sorted((abs(x) for i, x in enumerate(adj) if i != step_at), reverse=True)
    # Excising the largest of five UNCONDITIONALLY biases the residual low by
    # about a quarter when there is no step to excise. Here the excised value is
    # more than three times the next largest, so there is one; the guard makes
    # that a condition rather than a coincidence.
    if abs(adj[step_at]) < 3.0 * rest[0]:
        raise SystemExit(
            f"no step to excise: the largest adjacent change is "
            f"{abs(adj[step_at]):.3f} % against {rest[0]:.3f} % for the next, "
            f"so removing it would bias the residual low rather than isolate a "
            f"level change")
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

    The round count is `drafted / draft-max`, NOT generated minus accepted.

    An earlier version of this file used the second, on the reasoning that a
    round which drafts k tokens and has a accepted yields a+1 tokens. That is
    true of the mechanism and false of the counter. ERRATA A1 quotes the server
    source: on partial acceptance the checkpoint-and-restore branch returns
    BEFORE `slot.n_draft_accepted` is incremented, so on any arm that takes that
    branch the server's accepted counter under-counts. A13 measures it: the two
    arms here read 72.8 % against 73.0 % with zero checkpoints for
    `spec-dflash-n2`, and 29.7 % against 41.3 % with 772 checkpoints for
    `spec-draft-n8`.

    `drafted / draft-max` is exact for both, because the drafter always proposes
    its maximum: 14646 / 2 and 33408 / 8 are both whole numbers, and the
    acceptance they imply is 72.90 % and 41.38 % against A13's drafter-side
    73.0 % and 41.3 %. The earlier reading implied 41.38 -> 29.54 % for
    `spec-draft-n8`, which is A13's under-counted figure, and it moved that
    arm's target-step weight from 0.232 to 0.452. That is not a rounding
    difference: it is what decided whether a purely arm-agnostic per-target-step
    cost produces a positive cross-arm contrast, and it does.

    The check that caught nothing was "drafted per round does not exceed the
    arm's draft maximum". The wrong round count satisfies it too, at 4.109
    against 8. Integrality is the test that separates them.
    """
    out = {}
    for a in ARMS:
        P = D = A = 0
        for i in range(6):
            for r in _rows(a, i):
                P += r["predicted_n"]
                D += r.get("draft_n") or 0
                A += r.get("draft_n_accepted") or 0
        if a in NMAX:
            rounds = D / NMAX[a]
            if abs(rounds - round(rounds)) > 1e-9:
                raise SystemExit(f"{a}: {D} drafted is not a whole number of "
                                 f"rounds at draft-max {NMAX[a]}, so the drafter "
                                 f"does not always propose its maximum and this "
                                 f"round count is not derivable this way")
        else:
            rounds = float(P)
        out[a] = {"tokens": P, "drafted": D, "accepted_server": A,
                  "rounds": rounds,
                  "acceptance_drafter": (P - rounds) / D if D else 0.0,
                  "acceptance_server": A / D if D else 0.0,
                  "per_token": 1.0,
                  "per_target_step": rounds / P,
                  "per_forward": (D + rounds) / P}
    return out


def fit_agnostic(deltas: dict, sigmas: dict, weights: dict) -> dict:
    """Least-squares fit of one arm-agnostic cost, and its goodness of fit.

    Each hypothesis is one free scale over fixed per-arm weights, so with three
    arms it leaves two degrees of freedom and is falsifiable on its own. The
    previous pre-registration used a single one-sided contrast between two arms
    instead, and that only worked while every hypothesis put the contrast on the
    same side of zero. With the round counts corrected they do not.
    """
    num = sum(weights[a] * deltas[a] / sigmas[a] ** 2 for a in deltas)
    den = sum(weights[a] ** 2 / sigmas[a] ** 2 for a in deltas)
    c = num / den if den else 0.0
    chi2 = sum(((deltas[a] - c * weights[a]) / sigmas[a]) ** 2 for a in deltas)
    return {"c": c, "chi2": chi2, "df": max(0, len(deltas) - 1),
            "resid": {a: deltas[a] - c * weights[a] for a in deltas}}


def t4_step() -> tuple[dict, dict]:
    """T4's change in ms per generated token across its own step, with a real SE.

    The standard error is the paired-block one the three low and three high
    blocks actually give, not a fraction of the effect. The first version of
    this function invented it as a quarter of each arm's own change, which makes
    the chi-square a function of the number that was chosen rather than of the
    data, and it published three figures computed that way.
    """
    lo, hi, dl, sg = {}, {}, {}, {}
    for a in ARMS:
        low = [1000.0 / pooled(a, i) for i in range(3)]
        high = [1000.0 / pooled(a, i) for i in range(3, 6)]
        lo[a], hi[a] = st.mean(low), st.mean(high)
        dl[a] = hi[a] - lo[a]
        sg[a] = math.sqrt(st.stdev(low) ** 2 / 3 + st.stdev(high) ** 2 / 3)
    return dl, sg


def corpus_correlation(root: pathlib.Path | None = None) -> dict:
    """Do the three arms move together, block by block, across the whole corpus?

    A host cost of any denominator is shared, so it moves the arms together and
    would show as a correlation near one. The plan publishes these three numbers
    and nothing derived them until this function existed, which is the same
    defect it corrects elsewhere.

    Residuals are standardised within (directory, arm) so that a directory's own
    level and scale cannot induce a correlation, and arms are aligned by repeat
    index, which is the block.
    """
    data = (root or ROOT) / "v4_audit_2026_08_25" / "data"
    pairs: dict[tuple[str, str], list[tuple[float, float]]] = {}
    ndirs = 0
    for d in sorted(data.iterdir()):
        if not d.is_dir():
            continue
        per = {}
        for a in ARMS:
            got = {}
            for f in d.glob(f"{a}__rep*.json"):
                j = json.loads(f.read_text(encoding="utf-8"))
                rs = j["rows"]
                if not rs or j.get("crashed"):
                    continue
                n = sum(r["predicted_n"] for r in rs)
                msec = sum(r["predicted_ms"] for r in rs)
                if n and msec:
                    got[int(f.stem.rpartition("__rep")[2])] = msec / n
            per[a] = got
        if any(not per[a] for a in ARMS):
            continue
        reps = sorted(set.intersection(*[set(per[a]) for a in ARMS]))
        if len(reps) < 4:
            continue
        ndirs += 1
        z = {}
        for a in ARMS:
            v = [per[a][r] for r in reps]
            mu, sd = st.mean(v), st.stdev(v)
            z[a] = [(x - mu) / sd if sd else 0.0 for x in v]
        for i in range(len(ARMS)):
            for j2 in range(i + 1, len(ARMS)):
                k = tuple(sorted((ARMS[i], ARMS[j2])))
                pairs.setdefault(k, []).extend(zip(z[ARMS[i]], z[ARMS[j2]]))
    out = {"directories": ndirs, "r": {}}
    for k, v in pairs.items():
        xs = [a for a, _ in v]
        ys = [b for _, b in v]
        mx, my = st.mean(xs), st.mean(ys)
        num = sum((a - mx) * (b - my) for a, b in v)
        den = math.sqrt(sum((a - mx) ** 2 for a in xs)
                        * sum((b - my) ** 2 for b in ys))
        out["r"][k] = (num / den if den else float("nan"), len(v))
    return out


def wall_clock(nblocks: int = 24, washout: float = WASHOUT) -> dict:
    """What run X costs, derived rather than typed.

    Run T4 prices the overhead the arm-run records do not contain: its manifest
    to its completion marker is the invocation's wall clock, and the difference
    against the summed `ready_s` plus request time is warm-up, teardown settle,
    two `nvidia-smi` calls and the JSON writes, per arm-run.

    The washout is per PAIR, and there is one pair per arm per block. A version
    of this plan costed it at 20 s while specifying 30, and published the total
    that gave: the figure is derived here so that changing one changes the other.
    """
    d = json.loads((T4 / "manifest.json").read_text(encoding="utf-8"))
    c = json.loads((T4 / "RUN_COMPLETE.json").read_text(encoding="utf-8"))
    k0 = next(k for k in ("created", "started", "start") if k in d)
    k1 = next(k for k in ("completed_at", "completed", "finished") if k in c)
    t0 = datetime.datetime.fromisoformat(d[k0])
    t1 = datetime.datetime.fromisoformat(c[k1])
    runs = 0
    total = 0.0
    for a in ARMS:
        for i in range(6):
            j = json.loads((T4 / f"{a}__rep{i}.json").read_text(encoding="utf-8"))
            total += j.get("ready_s", 0.0) + sum(
                r["wall_ms"] for r in j["rows"]) / 1000.0
            runs += 1
    overhead = ((t1 - t0).total_seconds() - total) / runs
    per_pass = total / 6.0                       # six blocks of three arms in T4
    arm_runs = nblocks * len(ARMS) * 2           # loaded and unloaded
    measured = per_pass * nblocks * 2
    pairs = nblocks * len(ARMS)
    return {"overhead_per_arm_run": overhead, "arm_runs": arm_runs,
            "pairs": pairs, "washout": washout,
            "measured_h": measured / 3600.0,
            "with_overhead_h": (measured + overhead * arm_runs) / 3600.0,
            "total_h": (measured + overhead * arm_runs + pairs * washout) / 3600.0}


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


def _within(truth, resid, gap, hazard, rng, dur, nblocks=24, washout=0.0):
    d, state = [], rng.randint(0, 1)
    for b in range(1, nblocks + 1):
        first_loaded = (b % 4) in (1, 0)
        pair = []
        for k in range(2):
            # the second member is separated from the first by one arm-run AND
            # by the washout the plan mandates between them; a longer washout is
            # a longer window for the level to change inside the pair
            state = _flip(dur + (washout if k else 0.0), hazard, state, rng)
            pair.append(state * gap + rng.gauss(0, resid)
                        + (truth if (k == 0) == first_loaded else 0.0))
        # the rest of the block: the other two arms run twice each. The return
        # was DISCARDED here, so the level never advanced between blocks. It
        # cancels in a within-block difference and changed no published figure,
        # which is exactly why it would have sat there: a state variable that is
        # not assigned looks like one that is.
        state = _flip(max(0.0, 400.0 - 2.0 * dur - washout), hazard, state, rng)
        d.append(pair[0] - pair[1] if first_loaded else pair[1] - pair[0])
    return _verdict(d, nblocks - 1)


def power(m: dict, hazard_s: float | None = None,
          trials: int = TRIALS, washout: float = WASHOUT) -> dict:
    if hazard_s is None:
        hazard_s = hazard()["hazard"]
    rng = random.Random(SEED)
    dur = m["durations"]["spec-dflash-n2"]
    hz = hazard_s
    designs = {
        "between-block, 6 against 6": lambda tr: _between(
            tr, m["residual"], m["gap"], hz, rng),
        "within-block, 12 pairs": lambda tr: _within(
            tr, m["residual"], m["gap"], hz, rng, dur, 12, washout),
        "within-block, 18 pairs": lambda tr: _within(
            tr, m["residual"], m["gap"], hz, rng, dur, 18, washout),
        "within-block, 24 pairs": lambda tr: _within(
            tr, m["residual"], m["gap"], hz, rng, dur, 24, washout),
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


def check_plan(txt, m, p_table, ms, nz, keys, dl, sg, cc, wc, ends, washouts):
    """Every published figure this file derives, compared against the document.

    Returned as a list rather than printed, so that a regression test can
    corrupt the document and assert WHICH comparison fails. The count of
    corruptions this catches was a sentence in the changelog with nothing
    deriving it, which is the failing this file exists to remove.
    """
    missing = []
    for want in (f"{m['gap']:.2f} %", f"{m['residual']:.3f} %",
                 f"{m['adjacent_diff_sd']:.3f} %", f"{m['block_cv']:.3f} %"):
        if want not in txt:
            missing.append(want)
    # ROW-ANCHORED, not a bare substring: the first version looked for the
    # value anywhere after a pipe, so swapping two rows of the table, or
    # flipping every sign in the normaliser table, passed unnoticed.
    rows = {"between-block, 6 against 6": "between-block, 6 against 6",
            "within-block, 12 pairs": "within-block, 12 pairs",
            "within-block, 18 pairs": "within-block, 18 pairs",
            "within-block, 24 pairs": "within-block, 24 pairs"}
    for name, label in rows.items():
        r = p_table[name]
        want = [f"{r['-2']['moves']:.2f}", f"{r['-gap']['moves']:.2f}",
                f"{r['0']['holds']:.2f}"]
        line = next((l for l in txt.splitlines()
                     if l.startswith("|") and label in l), None)
        if line is None:
            missing.append(f"row {label!r} absent")
            continue
        cells = [c.strip().strip("*") for c in line.strip("|").split("|")]
        if cells[1:4] != want:
            missing.append(f"row {label!r} prints {cells[1:4]} not {want}")
    # the chi-squares the plan quotes, to the precision it quotes them
    chis = []
    for key, _label in keys:
        f = fit_agnostic(dl, sg, {a: nz[a][key] for a in ARMS})
        chis.append(f"{f['chi2']:.0f}")
    flat = " ".join(txt.split())
    if f"chi-square {chis[0]}, {chis[1]} and {chis[2]}" not in flat:
        missing.append(f"chi-squares {chis}")
    # the two tables the first version of this check never looked at
    levels = {a: (st.mean(pooled(a, i) for i in range(3)),
                  st.mean(pooled(a, i) for i in range(3, 6))) for a in ARMS}
    for a in ARMS:
        lo_, hi_ = levels[a]
        want = [f"{lo_:.3f}", f"{hi_:.3f}",
                f"{100 * (hi_ / lo_ - 1):+.2f} %".replace("+", "+")]
        line = next((l for l in txt.splitlines()
                     if l.startswith("|") and f"`{a}`" in l), None)
        if line is None:
            missing.append(f"arm-levels row {a!r} absent"); continue
        got = [c.strip().strip("*").replace("\u2212", "-")
               for c in line.strip("|").split("|")][1:4]
        if [g.replace(" ", "") for g in got] != [w.replace(" ", "") for w in want]:
            missing.append(f"arm-levels row {a!r} prints {got} not {want}")
    for w, r in washouts.items():
        label = "none" if w == 0 else f"{w:.0f} s"
        line = next((l for l in txt.splitlines()
                     if l.startswith("|") and label in l
                     and "washout" not in l.lower()), None)
        if line is None:
            missing.append(f"washout row {label!r} absent"); continue
        cells = [c.strip().strip("*") for c in line.strip("|").split("|")]
        wantw = [f"{r['-2']['moves']:.2f}", f"{r['0']['holds']:.2f}"]
        if cells[1:3] != wantw:
            missing.append(f"washout row {label!r} prints {cells[1:3]} not {wantw}")
    for label, r in ends.items():
        for v in (f"{r['-2']['moves']:.2f}", f"{r['0']['holds']:.2f}"):
            if v not in txt:
                missing.append(f"hazard {label} end {v}")
    for want in (f"{wc['with_overhead_h']:.2f} hours",
                 f"{wc['total_h']:.2f}"):
        if want not in txt:
            missing.append(f"wall clock {want!r}")
    for r, _n in cc["r"].values():
        if f"{r:+.3f}".replace("+", "+") not in txt.replace("−", "-"):
            missing.append(f"corpus correlation {r:+.3f}")
    for key, label in keys:
        c = (ms["spec-dflash-n2"] * (1 / 0.98 - 1)) / nz["spec-dflash-n2"][key]
        want = [f"{100 * (ms[a] / (ms[a] + c * nz[a][key]) - 1):+.2f} %".replace("+", "+")
                for a in ARMS]
        line = next((l for l in txt.splitlines()
                     if l.startswith("|") and label in l), None)
        if line is None:
            missing.append(f"normaliser row {label!r} absent")
            continue
        got = [x.strip().strip("*").replace("−", "-") for x in line.strip("|").split("|")][1:4]
        if [g.replace(" ", "") for g in got] != [w.replace(" ", "").replace("+-", "-")
                                                 for w in want]:
            missing.append(f"normaliser row {label!r} prints {got} not {want}")


    return missing


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
    ends: dict = {}
    washouts: dict = {}
    p_table = power(m, hz, args.trials)
    p = p_table
    gap_col = f"-{m['gap']:.2f} %"
    print(f"  {'design':30s} {'-2 %':>8s} {gap_col:>10s} {'0 -> holds':>12s}")
    for name, row in p.items():
        print(f"  {name:30s} {row['-2']['moves']:8.2f} "
              f"{row['-gap']['moves']:10.2f} {row['0']['holds']:12.2f}")
    print()

    if args.hazard is None:
        print("  at the ends of that interval, for the design this plan chooses:")
        for end, label in ((h["lo"], "fast"), (h["hi"], "slow")):
            r = power(m, end, args.trials)["within-block, 24 pairs"]
            ends[label] = r
            print(f"    hazard {end:6.0f} s ({label}): "
                  f"{r['-2']['moves']:.2f} / {r['-gap']['moves']:.2f} / "
                  f"{r['0']['holds']:.2f}")
        print()

    if args.hazard is None:
        print("  what the mandated washout costs, at the corpus hazard:")
        for w in (0.0, 15.0, 30.0, 60.0):
            r = (p_table if w == WASHOUT
                 else power(m, None, args.trials, w))["within-block, 24 pairs"]
            washouts[w] = r
            print(f"    washout {w:5.0f} s: {r['-2']['moves']:.2f} / "
                  f"{r['-gap']['moves']:.2f} / {r['0']['holds']:.2f}")
        print()

    ms = {a: 1000.0 / st.mean(pooled(a, i) for i in range(6)) for a in ARMS}
    nz = normalisers()
    # the labels are the plan's own row labels, because --check anchors on them
    keys = (("per_token", "per generated token"),
            ("per_forward", "per model forward pass"),
            ("per_target_step", "per target-model step"))
    print(f"  {'arm-agnostic cost':24s} " + " ".join(f"{a:>16s}" for a in ARMS))
    for key, label in keys:
        c = (ms["spec-dflash-n2"] * (1 / 0.98 - 1)) / nz["spec-dflash-n2"][key]
        cells = " ".join(
            f"{c * nz[a][key]:7.3f} ms {100 * (ms[a] / (ms[a] + c * nz[a][key]) - 1):+6.2f}%"
            for a in ARMS)
        print(f"  {label:24s} {cells}")
    print("\n  Each row is ONE free scale over fixed per-arm weights, so with three")
    print("  arms it leaves two degrees of freedom and can be rejected on its own.")
    print("  The previous pre-registration used a one-sided contrast between two")
    print("  arms, which only worked while every row put that contrast on the same")
    print("  side of zero. With the round counts taken from the drafter rather than")
    print("  from the server's under-counted accepted field, they do not:")
    for key, label in keys:
        c = (ms["spec-dflash-n2"] * (1 / 0.98 - 1)) / nz["spec-dflash-n2"][key]
        dd = c * nz["spec-dflash-n2"][key] - c * nz["spec-draft-n8"][key]
        print(f"    {label:24s} D = {dd:+.3f} ms per token"
              + ("   <- above zero, which the old rule read as arm-specific"
                 if dd > 1e-9 else ""))
    print("\n  So the test is a fit, not a contrast. T4's own step already fails all")
    print("  three, because the arms moved in opposite directions across it:")
    dl, sg = t4_step()
    for key, label in keys:
        w = {a: nz[a][key] for a in ARMS}
        f = fit_agnostic(dl, sg, w)
        print(f"    {label:24s} chi2 = {f['chi2']:8.1f} on {f['df']} df")

    wc = wall_clock()
    print("\n  wall clock, derived from run T4's own invocation:")
    print(f"    overhead not in the arm-run records: "
          f"{wc['overhead_per_arm_run']:.2f} s per arm-run")
    print(f"    {wc['arm_runs']} arm-runs: {wc['measured_h']:.2f} h measured, "
          f"{wc['with_overhead_h']:.2f} h with overhead")
    print(f"    plus {wc['pairs']} pairs x {wc['washout']:.0f} s of washout: "
          f"{wc['total_h']:.2f} h")

    cc = corpus_correlation()
    print(f"\n  block-aligned correlation of ms/token residuals, "
          f"{cc['directories']} directories:")
    for k, (r, n) in sorted(cc["r"].items(), key=lambda kv: -kv[1][0]):
        print(f"    {k[0]:16s} vs {k[1]:16s} r = {r:+.3f}  n = {n}")
    print("  A shared host cost of any denominator would put these near one.")

    if args.check:
        missing = check_plan(PLAN.read_text(encoding="utf-8"), m, p_table,
                             ms, nz, keys, dl, sg, cc, wc, ends, washouts)
        print("\n  plan check: " + ("OK" if not missing
                                    else "MISMATCH\n    "
                                    + "\n    ".join(missing)))
        raise SystemExit(1 if missing else 0)


if __name__ == "__main__":
    main()
