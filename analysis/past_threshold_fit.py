#!/usr/bin/env python3
"""Recompute every number in v4_audit_2026_08_25/PREREGISTERED_PREDICTION.md.

The pre-registration was written before its data existed and its prediction
section reproduces exactly. Its *outcome* section did not: it was computed by
ad-hoc commands that were never committed, and eight of its figures were wrong
— two baselines taken from a single repeat, two derived from the checkpoint
size A12 withdrew, a decode rate labelled end to end, a checkpoint count
attributed to one request instead of ten, an error column rounded twice, and a
scatter range nothing produces. Two more that looked wrong were not: the
residual step and the amortisation deviation both reproduce, once you know the
conventions the document never stated, which are now stated where they are
used. This script exists so none of it can happen again: every figure the
document publishes is produced here, from the committed data, and asserted in
analysis/verify_claims.py.

    python3 analysis/past_threshold_fit.py            # print the report
    python3 analysis/past_threshold_fit.py --json     # machine-readable
"""

import argparse
import glob
import json
import math
import os
import statistics as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUDIT = os.path.join(ROOT, "v4_audit_2026_08_25")
C_DIR = os.path.join(AUDIT, "data", "C_master_matrix_think_on")
E_DIR = os.path.join(AUDIT, "data", "E_past_threshold")
COUNTERS = os.path.join(AUDIT, "data", "acceptance_counter_comparison.json")

# the runs the two directories came from, as recorded in the counter dump
C_RUN = "matrix_C_20260825_204529"
E_RUN = "matrix_E_threshold_20260825_224802"

# server-reported common_prompt_checkpoint::size(), ERRATA A12. NOT 101.345:
# that is this number plus the draft component it already contains.
CHECKPOINT_MIB = 82.079

# 256 routed experts, top-8 routing. The fraction of routed experts a draft of
# n tokens is expected to touch, which is what MoESD's coverage argument is
# about. It is a property of the routing, not of any measurement here.
N_EXPERTS, TOP_K = 256, 8
coverage = lambda n: 1.0 - (1.0 - TOP_K / N_EXPERTS) ** n

# the first n_max at or past the 95 % coverage point the argument names
THRESHOLD_NMAX = 96

SUB = (1, 2, 4, 8, 16, 32)      # fitted, from run C
SUPRA = (64, 96, 128)           # predicted in advance, then measured in run E


def _dir(n_max):
    return C_DIR if n_max in SUB else E_DIR


def arm(directory, name):
    """Pool every request of every repeat of one arm."""
    files = sorted(glob.glob(os.path.join(directory, f"{name}__rep*.json")))
    if not files:
        raise SystemExit(f"no data for {name} in {directory}")
    ms = n = drafted = accepted = 0
    wall = 0.0
    per_repeat = []
    for f in files:
        rows = json.load(open(f, encoding="utf-8"))["rows"]
        r_ms = sum(r["timings"]["predicted_ms"] for r in rows)
        r_n = sum(r["timings"]["predicted_n"] for r in rows)
        per_repeat.append(1000.0 * r_n / r_ms)
        ms += r_ms
        n += r_n
        wall += sum(r["wall_ms"] for r in rows)
        drafted += sum(r.get("draft_n", 0) or 0 for r in rows)
        accepted += sum(r.get("draft_n_accepted", 0) or 0 for r in rows)
    return {
        "arm": name, "repeats": len(files), "requests": sum(
            len(json.load(open(f, encoding="utf-8"))["rows"]) for f in files),
        "generated": n, "predicted_ms": ms,
        "ms_per_token": ms / n,
        "draft_per_gen": drafted / n,
        "acceptance": (accepted / drafted) if drafted else 0.0,
        "decode_tok_s": 1000.0 * n / ms,
        "wall_tok_s": 1000.0 * n / wall,
        "per_repeat_decode_tok_s": per_repeat,
    }


def ols(X, y):
    """Least squares by the normal equations. X rows carry their own 1.0."""
    k = len(X[0])
    A = [[sum(X[i][p] * X[i][q] for i in range(len(X))) for q in range(k)]
         for p in range(k)]
    b = [sum(X[i][p] * y[i] for i in range(len(X))) for p in range(k)]
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for c in range(k):
        piv = max(range(c, k), key=lambda r: abs(M[r][c]))
        M[c], M[piv] = M[piv], M[c]
        for r in range(k):
            if r != c:
                f = M[r][c] / M[c][c]
                for q in range(c, k + 1):
                    M[r][q] -= f * M[c][q]
    return [M[i][k] / M[i][i] for i in range(k)]


def r_squared(y, yhat):
    m = sum(y) / len(y)
    return 1.0 - (sum((a - b) ** 2 for a, b in zip(y, yhat))
                  / sum((a - m) ** 2 for a in y))


def power_law(xs, ys):
    """y = a * x**b, fitted in logs. Returns (a, b)."""
    c = ols([[math.log(x), 1.0] for x in xs], [math.log(v) for v in ys])
    return math.exp(c[1]), c[0]


def pearson(xs, ys):
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    return (sum((a - mx) * (b - my) for a, b in zip(xs, ys))
            / math.sqrt(sum((a - mx) ** 2 for a in xs)
                        * sum((b - my) ** 2 for b in ys)))


def checkpoints_per_arm_run():
    """checkpoints_created per arm-run, from the committed counter dump."""
    out = {}
    for r in json.load(open(COUNTERS, encoding="utf-8")):
        if r["run"] in (C_RUN, E_RUN) and r["arm"].startswith("spec-draft-n"):
            n_max = int(r["arm"].rsplit("n", 1)[1])
            # run E repeated n_max 32 as an overlap check; run C is the fitted
            # source for it, so prefer C where both exist.
            if n_max in SUB and r["run"] != C_RUN:
                continue
            out.setdefault(n_max, []).append(r["checkpoints_created"])
    return out


def build():
    arms = {n: arm(_dir(n), f"spec-draft-n{n}") for n in SUB + SUPRA}
    base_c, base_e = arm(C_DIR, "baseline"), arm(E_DIR, "baseline")
    # the no-speculation step. Pooled over all five repeats of run C, which is
    # the run the model is fitted on. Repeat 0 alone reads 7.873 and was
    # published as "7.87" until 2026-08-27.
    base_ms = base_c["ms_per_token"]

    rounds = lambda a, n: 1.0 / (a["acceptance"] * n + 1)

    # ---- the three-parameter model, fitted on n_max 1..32 -------------------
    Xf = [[rounds(arms[n], n), arms[n]["draft_per_gen"], 1.0] for n in SUB]
    yf = [arms[n]["ms_per_token"] for n in SUB]
    co = ols(Xf, yf)
    fit3 = {
        "per_round_ms": co[0], "per_draft_token_ms": co[1], "intercept_ms": co[2],
        "r2": r_squared(yf, [sum(c * x for c, x in zip(co, r)) for r in Xf]),
        "span_ms_per_token": [min(yf), max(yf)],
        "intercept_over_baseline_pct": (co[2] / base_ms - 1.0) * 100.0,
    }
    Xv = [[arms[n]["draft_per_gen"], 1.0] for n in SUB]
    cv = ols(Xv, yf)
    r2v = r_squared(yf, [cv[0] * x[0] + cv[1] for x in Xv])
    fit3["draft_volume_only_r2"] = r2v
    fit3["error_removed_by_rounds_pct"] = (fit3["r2"] - r2v) / (1 - r2v) * 100.0
    loo = []
    for i in range(len(SUB)):
        ci = ols([Xf[j] for j in range(len(SUB)) if j != i],
                 [yf[j] for j in range(len(SUB)) if j != i])
        loo.append((SUB[i], sum(c * x for c, x in zip(ci, Xf[i])) - yf[i]))
    fit3["loo_mae_ms"] = sum(abs(e) for _, e in loo) / len(loo)
    fit3["loo_worst"] = max(loo, key=lambda t: abs(t[1]))
    r = pearson([x[0] for x in Xf], [x[1] for x in Xf])
    fit3["regressor_r"] = r
    fit3["regressor_vif"] = 1.0 / (1.0 - r * r)

    # ---- extrapolation, and the numbers registered in advance ---------------
    ka, ba = power_law(list(SUB), [arms[n]["acceptance"] for n in SUB])
    kd, bd = power_law(list(SUB), [arms[n]["draft_per_gen"] for n in SUB])
    predicted = {}
    for n in SUPRA:
        pa, pd = ka * n ** ba, kd * n ** bd
        pm = co[0] / (pa * n + 1) + co[1] * pd + co[2]
        predicted[n] = {"acceptance": pa, "draft_per_gen": pd,
                        "ms_per_token": pm, "tok_s": 1000.0 / pm}

    # The registered numbers are what the committed table published, to one
    # decimal. Scoring the measurement against anything else is scoring it
    # against a prediction nobody made.
    registered = {64: 13.4, 96: 10.6, 128: 8.9}
    outcome = {}
    for n in SUPRA:
        m = arms[n]
        fed = (co[0] * rounds(m, n) + co[1] * m["draft_per_gen"] + co[2])
        outcome[n] = {
            "registered_tok_s": registered[n],
            "measured_tok_s": m["decode_tok_s"],
            "error_pct": (m["decode_tok_s"] / registered[n] - 1.0) * 100.0,
            "measured_input_ms_per_token": fed,
            "measured_input_over_prediction_pct":
                (fed / m["ms_per_token"] - 1.0) * 100.0,
        }

    # ---- the one-regressor law over the whole sweep -------------------------
    order = list(SUB + SUPRA)
    Xa = [[arms[n]["draft_per_gen"], 1.0] for n in order]
    ya = [arms[n]["ms_per_token"] for n in order]
    ca = ols(Xa, ya)
    yha = [ca[0] * x[0] + ca[1] for x in Xa]
    # Residual, in the convention ERRATA A10 already publishes a column of:
    # (measured - predicted) / predicted, so a negative residual is the law
    # over-predicting cost. The pre-registration used this and did not say so,
    # which is why an audit of it in the opposite convention read -0.13.
    resid = {n: (v - h) / h * 100.0 for n, h, v in zip(order, yha, ya)}
    below = [resid[n] for n in order if n < THRESHOLD_NMAX]
    above = [resid[n] for n in order if n >= THRESHOLD_NMAX]
    law = {
        "slope_ms_per_draft_token": ca[0], "intercept_ms": ca[1],
        "r2": r_squared(ya, yha),
        "residual_pct": resid,
        "residual_arc_pct": [min(resid.values()), max(resid.values())],
        "mean_residual_below_pct": sum(below) / len(below),
        "mean_residual_at_or_above_pct": sum(above) / len(above),
        "step_pp": sum(above) / len(above) - sum(below) / len(below),
        "slope_over_baseline": ca[0] / base_ms,
        "intercept_over_baseline": ca[1] / base_ms,
    }

    # ---- hypothesis 1: marginal cost per drafted token amortises ------------
    marginal = {n: (arms[n]["ms_per_token"] - base_ms) / arms[n]["draft_per_gen"]
                for n in order}
    # "the sub-threshold points" are the six the model was fitted on, n_max
    # 1..32; the three that follow were all predicted before they were measured.
    sub_pts = [(n, marginal[n]) for n in SUB]
    cs = ols([[1.0 / n, 1.0] for n, _ in sub_pts], [v for _, v in sub_pts])
    amort = {
        "marginal_ms_per_draft_token": marginal,
        "sub_threshold_curve": {"asymptote_ms": cs[1], "per_n_ms": cs[0]},
        "supra_deviation_pct": {
            n: (marginal[n] / (cs[0] / n + cs[1]) - 1.0) * 100.0 for n in SUPRA},
    }
    ca9 = ols([[1.0 / n, 1.0] for n in order], [marginal[n] for n in order])
    amort["all_nine_curve"] = {"asymptote_ms": ca9[1], "per_n_ms": ca9[0]}
    amort["all_nine_deviation_pct"] = {
        n: (marginal[n] / (ca9[0] / n + ca9[1]) - 1.0) * 100.0 for n in SUPRA}

    # ---- hypothesis 2: checkpoint traffic dominates -------------------------
    ck = checkpoints_per_arm_run()
    traffic = {}
    for n in order:
        counts = ck.get(n)
        if not counts:
            continue
        gen_per_run = arms[n]["generated"] / arms[n]["repeats"]
        traffic[n] = st.mean(counts) * CHECKPOINT_MIB / gen_per_run
    ckpt = {
        "checkpoint_mib": CHECKPOINT_MIB,
        "checkpoints_per_arm_run": {n: st.mean(v) for n, v in ck.items()},
        "requests_per_arm_run": arms[1]["requests"] // arms[1]["repeats"],
        "generated_per_arm_run": arms[1]["generated"] // arms[1]["repeats"],
        "checkpoints_per_request": (st.mean(ck[1])
                                    / (arms[1]["requests"] // arms[1]["repeats"])),
        "mib_per_generated_token": traffic,
        "correlation_with_cost": pearson(
            [traffic[n] for n in sorted(traffic)],
            [arms[n]["ms_per_token"] for n in sorted(traffic)]),
    }

    return {
        "arms": arms,
        "baseline": {"C_pooled_ms_per_token": base_c["ms_per_token"],
                     "C_repeat0_ms_per_token":
                         1000.0 / base_c["per_repeat_decode_tok_s"][0],
                     "C_decode_tok_s": base_c["decode_tok_s"],
                     "C_wall_tok_s": base_c["wall_tok_s"],
                     "E_pooled_ms_per_token": base_e["ms_per_token"],
                     "E_decode_tok_s": base_e["decode_tok_s"]},
        "coverage_pct": {n: coverage(n) * 100.0 for n in order},
        "threshold_nmax": THRESHOLD_NMAX,
        "fit3": fit3, "predicted": predicted, "outcome": outcome, "law": law,
        "amortisation": amort, "checkpoints": ckpt,
        "repeat_sd_tok_s": {n: st.stdev(arms[n]["per_repeat_decode_tok_s"])
                            for n in SUPRA},
    }


def report(d):
    p = print
    p("=" * 72)
    p("  the past-threshold sweep, recomputed from the committed data")
    p("=" * 72)
    p("\n  n_max    ms/tok   draft/gen   accept   decode tok/s   wall tok/s   coverage")
    for n in sorted(d["arms"], key=int):
        a = d["arms"][n]
        p(f"  {n:>5} {a['ms_per_token']:9.3f} {a['draft_per_gen']:11.3f} "
          f"{100*a['acceptance']:7.2f}% {a['decode_tok_s']:14.2f} "
          f"{a['wall_tok_s']:12.2f} {d['coverage_pct'][n]:10.1f}%")
    b = d["baseline"]
    p(f"\n  baseline (run C, five repeats pooled): {b['C_pooled_ms_per_token']:.3f} ms/tok, "
      f"{b['C_decode_tok_s']:.1f} tok/s decode, {b['C_wall_tok_s']:.1f} tok/s wall-clock")
    p(f"  the same baseline from repeat 0 alone:  {b['C_repeat0_ms_per_token']:.3f} ms/tok "
      f"— published as 7.87 until 2026-08-27")

    f = d["fit3"]
    p("\n  three-parameter model, fitted on n_max 1..32:")
    p(f"    ms/tok = {f['per_round_ms']:.2f} x rounds/tok + {f['per_draft_token_ms']:.2f} "
      f"x draft/gen + {f['intercept_ms']:.2f}")
    p(f"    R2 = {f['r2']:.4f}; draft volume alone {f['draft_volume_only_r2']:.4f}, "
      f"the per-round term removes {f['error_removed_by_rounds_pct']:.0f} % of what is left")
    p(f"    leave-one-out MAE {f['loo_mae_ms']:.2f} ms over {f['span_ms_per_token'][0]:.1f}"
      f"-{f['span_ms_per_token'][1]:.1f}; worst n_max {f['loo_worst'][0]} {f['loo_worst'][1]:+.2f}")
    p(f"    intercept is {f['intercept_over_baseline_pct']:+.0f} % of the measured "
      f"no-speculation cost; regressors r = {f['regressor_r']:.3f}, VIF {f['regressor_vif']:.2f}")

    p("\n  registered in advance, then measured:")
    p("    n_max   registered   measured   error     model fed measured inputs")
    for n in sorted(d["outcome"], key=int):
        o = d["outcome"][n]
        p(f"    {n:>5} {o['registered_tok_s']:11.1f} {o['measured_tok_s']:10.2f} "
          f"{o['error_pct']:+7.1f} % {o['measured_input_over_prediction_pct']:+22.1f} %")

    l = d["law"]
    p("\n  one regressor over all nine points:")
    p(f"    ms/tok = {l['intercept_ms']:.2f} + {l['slope_ms_per_draft_token']:.3f} x draft/gen"
      f"   R2 = {l['r2']:.5f}")
    p(f"    slope is {l['slope_over_baseline']:.3f} of a target step; intercept "
      f"{l['intercept_over_baseline']:.1f}x it")
    p(f"    residuals {l['residual_arc_pct'][0]:+.1f} % .. {l['residual_arc_pct'][1]:+.1f} %; "
      f"mean {l['mean_residual_below_pct']:+.2f} % below n_max {d['threshold_nmax']}, "
      f"{l['mean_residual_at_or_above_pct']:+.2f} % at or above")
    p(f"    step at the coverage point: {l['step_pp']:+.2f} percentage points")

    a = d["amortisation"]
    p("\n  marginal cost per drafted token (over the no-speculation step):")
    p("    " + "  ".join(f"n{n}={a['marginal_ms_per_draft_token'][n]:.1f}"
                         for n in sorted(a["marginal_ms_per_draft_token"], key=int)))
    p("    fitted a+b/n below the threshold, the supra points sit "
      + ", ".join(f"{v:+.1f} %" for v in a["supra_deviation_pct"].values())
      + " off it; fitted on all nine, "
      + ", ".join(f"{v:+.1f} %" for v in a["all_nine_deviation_pct"].values()))

    c = d["checkpoints"]
    p(f"\n  checkpoint traffic at {c['checkpoint_mib']} MiB per checkpoint:")
    p(f"    {c['checkpoints_per_arm_run'][1]:.0f} checkpoints per arm-run of "
      f"{c['requests_per_arm_run']} requests and {c['generated_per_arm_run']} generated "
      f"tokens = {c['checkpoints_per_request']:.1f} per request")
    p("    " + "  ".join(f"n{n}={c['mib_per_generated_token'][n]:.1f}"
                         for n in sorted(c["mib_per_generated_token"], key=int))
      + " MiB per generated token")
    p(f"    correlation with cost = {c['correlation_with_cost']:.2f}")
    p("\n  run-to-run SD over the three repeats: "
      + ", ".join(f"n{n}={v:.2f}" for n, v in sorted(d['repeat_sd_tok_s'].items())) + " tok/s")
    p("")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true", help="emit the values as JSON")
    ap.add_argument("--out", help="write the JSON here as well")
    args = ap.parse_args()
    d = build()
    payload = json.dumps(d, indent=1, sort_keys=True, default=str)
    if args.json:
        print(payload)
    else:
        report(d)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(payload + "\n")


if __name__ == "__main__":
    main()
