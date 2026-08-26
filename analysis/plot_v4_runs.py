"""Charts for the 2026-08-26 runs: batching (I) and the DFlash sweep (J, K).

Two figures, and the choice of quantity in each is the point.

  plot_batching.png      aggregate throughput - generated tokens over the
                         wall-clock of the whole prompt set - because at
                         concurrency > 1 a per-request rate is not system
                         throughput. The number of concurrent CLIENT requests is read out of the
                         request timestamps and printed on the chart, so a run
                         that asked for eight and got one cannot be mistaken for
                         a batching measurement.

  plot_dflash_sweep.png  percent change against each run's OWN no-speculation
                         baseline. Runs J and K differ in context and in the
                         memory fitter's margin, so their absolute rates are not
                         comparable; their deltas are, and plotting the delta is
                         what lets the two appear on one axis honestly.

Run: python analysis/plot_v4_runs.py
"""
from __future__ import annotations

import glob
import json
import sys
import statistics as st
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "v4_audit_2026_08_25" / "data"
OUT = ROOT / "analysis"

C_REF = "#3f6d9e"       # no speculation
C_ACTIVE = "#c0504d"    # matched-vocabulary drafter
C_DFLASH = "#4a7c59"    # DFlash
C_DFLASH2 = "#8fae7f"   # DFlash, second configuration
C_INACTIVE = "#8d9aa8"  # drafter-free n-gram methods

FOOTER = ("2026-08-26, llama.cpp 3737e4137, one RTX 3090, Qwen3.6-35B-A3B-UD-Q4_K_XL, "
          "greedy, thinking on, ten prompts, three repeats per arm. "
          "Controls and caveats: ERRATA.md, v4_audit_2026_08_25/README.md.")


SERIES: dict = {}
CHECK = "--check" in sys.argv


def record(chart: str, **series) -> None:
    """Register what a chart actually plots.

    A PNG cannot be diffed usefully and matplotlib output is not byte-stable
    across versions, so CI cannot tell a chart that is current from one left
    behind by a data change. Every chart registers its plotted series here;
    `--check` recomputes them and compares against the committed
    `analysis/plot_data.json`, which is the part a reader would be misled by.
    """
    def _clean(v):
        if isinstance(v, float):
            return None if v != v else round(v, 4)
        if isinstance(v, (list, tuple)):
            return [_clean(x) for x in v]
        if isinstance(v, dict):
            return {k: _clean(x) for k, x in v.items()}
        return v
    SERIES[chart] = _clean(series)


def _footer(fig, extra="", base=None):
    fig.text(0.5, 0.004, (FOOTER if base is None else base) + extra, ha="center",
             va="bottom", fontsize=7.2, color="#5a5a5a", style="italic", wrap=True)


def load(pattern: str) -> dict[str, list[dict]]:
    arms = defaultdict(list)
    for f in sorted(glob.glob(str(DATA / pattern))):
        r = json.load(open(f, encoding="utf-8"))
        if r.get("rows"):
            arms[r["arm"]].append(r)
    return arms


def agg(runs: list[dict]) -> tuple[float, float]:
    v = [r["aggregate_tok_s"] for r in runs if r.get("wall_s")]
    return (st.mean(v), st.stdev(v) if len(v) > 1 else 0.0) if v else (float("nan"), 0.0)


def pooled(runs: list[dict]) -> float:
    n = sum(x["predicted_n"] for r in runs for x in r["rows"])
    ms = sum(x["predicted_ms"] for r in runs for x in r["rows"])
    return 1000 * n / ms if ms else float("nan")


# --------------------------------------------------------------- run I ----
def plot_batching() -> None:
    levels, base_y, base_e, spec_y, spec_e, widths = [], [], [], [], [], []
    for c in (1, 4, 8):
        arms = load(f"matrix_I2_conc{c}_*/*__rep*.json")
        if "baseline" not in arms or "spec-draft-n8" not in arms:
            print(f"  concurrency {c} missing - skipping batching chart")
            return
        levels.append(c)
        m, s = agg(arms["baseline"]);      base_y.append(m); base_e.append(s)
        m, s = agg(arms["spec-draft-n8"]); spec_y.append(m); spec_e.append(s)
        seen = {r.get("max_client_requests_in_flight", r.get("max_in_flight"))
                for r in arms["baseline"] + arms["spec-draft-n8"]}
        widths.append(sorted(x for x in seen if x is not None))

    record("batching", levels=levels, base_y=base_y, base_e=base_e,
           spec_y=spec_y, spec_e=spec_e, widths=widths)
    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    x = range(len(levels))
    ax.errorbar(x, base_y, yerr=base_e, color=C_REF, marker="o", lw=2.0,
                capsize=4, label="no speculation")
    ax.errorbar(x, spec_y, yerr=spec_e, color=C_ACTIVE, marker="s", lw=2.0,
                capsize=4, label="spec-draft-n8 (matched vocabulary)")

    # White boxes because the c=8 error bar is large enough to run through a
    # bare label, and the ratio callout sits between the two series.
    box = dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85)
    for i, c in enumerate(levels):
        ax.annotate(f"{base_y[i]:.0f}", (i, base_y[i] + base_e[i]),
                    textcoords="offset points", xytext=(0, 9), ha="center",
                    fontsize=9, color=C_REF, bbox=box)
        ax.annotate(f"{spec_y[i]:.0f}", (i, spec_y[i] - spec_e[i]),
                    textcoords="offset points", xytext=(0, -15), ha="center",
                    fontsize=9, color=C_ACTIVE, bbox=box)
        ax.annotate(f"{spec_y[i] / base_y[i]:.2f}x", (i, (base_y[i] + spec_y[i]) / 2),
                    ha="center", fontsize=9.6, color="#3f3f46",
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#d0d0d0", lw=0.7))

    ax.set_xlim(-0.35, len(levels) - 0.65)
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"{c}\nclient requests in flight: "
                        f"{', '.join(str(w) for w in widths[i])}"
                        for i, c in enumerate(levels)], fontsize=9)
    ax.set_xlabel("requests in flight")
    ax.set_ylabel("aggregate throughput (generated tokens / wall-clock second)")
    ax.set_ylim(0, max(base_y) * 1.18)
    ax.set_title("Aggregate throughput against concurrent client requests\n"
                 "the no-speculation arm gains, the external-drafter arm does not: "
                 "0.28x at one, 0.16x at eight",
                 fontsize=11.5)
    # Below the axes: the only large empty region is where the c=1 labels sit.
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2,
              fontsize=9, frameon=False)
    ax.grid(axis="y", color="#dcdcdc", lw=0.6)
    ax.set_axisbelow(True)
    plt.tight_layout(rect=[0, 0.075, 1, 1])
    _footer(fig, " Error bars are the run-to-run SD of three repeats. The x axis is "
                 "CONCURRENT CLIENT REQUESTS, verified from request timestamps; the "
                 "server's decode batch width was not instrumented.")
    if not CHECK:  # --check verifies the numbers, it must not dirty the tree
        plt.savefig(OUT / "plot_batching.png", dpi=150, bbox_inches="tight")
    plt.close()
    if not CHECK:
        print(f"  wrote {(OUT / 'plot_batching.png').relative_to(ROOT)}")


# ------------------------------------------------------------ runs J, K ----
def plot_dflash_sweep() -> None:
    series = []
    for label, pattern, colour in (
            ("run J:  -c 16384, fitter default margin", "matrix_J2_*/*__rep*.json", C_DFLASH),
            ("run K:  -c 8192, --fit-target 2048", "matrix_K1_sweep_*/*__rep*.json", C_DFLASH2)):
        arms = load(pattern)
        if "baseline" not in arms:
            continue
        b = agg(arms["baseline"])[0]
        pts = sorted((int(a.rsplit("n", 1)[1]), 100 * (agg(arms[a])[0] / b - 1),
                      100 * sum(x["draft_n_accepted"] for r in arms[a] for x in r["rows"])
                      / max(1, sum(x["draft_n"] for r in arms[a] for x in r["rows"])))
                     for a in arms if a.startswith("spec-dflash-n"))
        if pts:
            series.append((label, pts, colour, b))
    record("dflash_sweep",
           series=[{"label": lb, "points": pt, "baseline": bs}
                   for lb, pt, _c, bs in series])
    if not series:
        print("  no DFlash data - skipping sweep chart")
        return

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(9.2, 7.4), sharex=True,
                                  gridspec_kw={"height_ratios": [2.1, 1]})
    # The two runs land almost on top of each other at n_max 4 and 8 - which is
    # the replication, and also a label collision. Push each series' labels to a
    # different distance rather than letting them overprint.
    for k, (label, pts, colour, b) in enumerate(series):
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", lw=2.0,
                color=colour, label=f"{label}   (its baseline: {b:.1f} tok/s)")
        ax2.plot([p[0] for p in pts], [p[2] for p in pts], marker="o", lw=1.8,
                 color=colour)
        up, down = (24, -31) if k == 0 else (10, -17)
        for n, d, _ in pts:
            ax.annotate(f"{d:+.1f}%", (n, d), textcoords="offset points",
                        xytext=(0, up if d > 0 else down), ha="center",
                        fontsize=8.6, color=colour,
                        bbox=dict(boxstyle="round,pad=0.15", fc="white",
                                  ec="none", alpha=0.85))

    ax.axhline(0, color=C_REF, ls="--", lw=1.2)
    # On the zero line itself, not in a corner: the corner it used to sit in is
    # exactly where the worst-performing point lands.
    ax.text(0.012, 0.0, " no speculation", transform=ax.get_yaxis_transform(),
            ha="left", va="bottom", fontsize=8.6, color=C_REF,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.9))
    ys = [d for _, pp, _, _ in series for _, d, _ in pp] + [0.0]
    pad = max(6.0, 0.12 * (max(ys) - min(ys)))
    ax.set_ylim(min(ys) - 1.9 * pad, max(ys) + 1.6 * pad)
    ax.set_ylabel("change in aggregate throughput\nagainst the same run's baseline (%)")
    ax.set_title("DFlash self-speculation: a plateau at draft length 2-4, then a cliff\n"
                 "one binary, one placement policy, three repeats per point; "
                 "two runs, independently",
                 fontsize=11.5)
    ax.legend(loc="lower left", fontsize=8.8, frameon=False)
    ax.grid(color="#dcdcdc", lw=0.6)
    ax.set_axisbelow(True)

    ax2.set_ylabel("draft tokens\naccepted (%)")
    ax2.set_xlabel("--spec-draft-n-max  (maximum tokens the drafter may propose)")
    ax2.set_xscale("log", base=2)
    ax2.set_xticks([p[0] for _, pts, _, _ in series for p in pts])
    ax2.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax2.grid(color="#dcdcdc", lw=0.6)
    ax2.set_axisbelow(True)

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    _footer(fig, " Runs J and K differ in context and fitter margin, so only their "
                 "deltas against their own baselines are comparable.")
    if not CHECK:  # --check verifies the numbers, it must not dirty the tree
        plt.savefig(OUT / "plot_dflash_sweep.png", dpi=150, bbox_inches="tight")
    plt.close()
    if not CHECK:
        print(f"  wrote {(OUT / 'plot_dflash_sweep.png').relative_to(ROOT)}")


# ----------------------------------------------------------------- run L ----
def plot_acceptance_threshold() -> None:
    """Acceptance against speed-up, with the out-of-sample points drawn too.

    The line is fitted on run L only. Runs J and K are plotted as open markers
    because they never informed it - ERRATA A10 is a single-regressor fit that
    looked excellent in sample and was falsified out of it, so the chart has to
    show the test, not just the fit.
    """
    def series(pattern, only=None, length_matched=False):
        """Per-prompt (acceptance, delta-vs-baseline) points.

        The baseline arm has to be loaded even when it is not plotted, because
        it is the denominator; filtering the glob instead of the arms is what
        broke the first version of this function.
        """
        per, acc = defaultdict(lambda: defaultdict(list)), defaultdict(lambda: defaultdict(lambda: [0, 0]))
        lens: dict = defaultdict(lambda: defaultdict(set))
        for f in glob.glob(str(DATA / pattern)):
            r = json.load(open(f, encoding="utf-8"))
            for x in r["rows"]:
                per[x["tag"]][r["arm"]].append(x["predicted_per_second"])
                a = acc[x["tag"]][r["arm"]]; a[0] += x["draft_n_accepted"]; a[1] += x["draft_n"]
                lens[x["tag"]][r["arm"]].add(x["predicted_n"])
        out = []
        for t in per:
            if "baseline" not in per[t]:
                continue
            # ERRATA A17: with thinking off the arms stop in different places,
            # so a point built from them compares different amounts of work
            if length_matched and len({n for a in lens[t] for n in lens[t][a]}) > 1:
                continue
            for arm in per[t]:
                if arm == "baseline" or not acc[t][arm][1]:
                    continue
                if only and only not in arm:
                    continue
                out.append((100 * acc[t][arm][0] / acc[t][arm][1],
                            100 * (st.mean(per[t][arm]) / st.mean(per[t]["baseline"]) - 1)))
        return out

    fit = [("thinking on", "matrix_L_thinkon_*/*__rep*.json", "#2f5d8a"),
           ("thinking off", "matrix_L_thinkoff_*/*__rep*.json", "#c0504d")]
    xs, ys = [], []
    fig, ax = plt.subplots(figsize=(9.4, 6.0))
    for label, pat, colour in fit:
        # Only the DFlash arms inform the line. spec-draft-n8 is a separate
        # draft model paying a full forward pass per token, so its cost per unit
        # of acceptance is a different quantity; it appears in the out-of-sample
        # markers instead, where it is the worst miss.
        d = series(pat, only="dflash")
        xs += [p[0] for p in d]; ys += [p[1] for p in d]
        ax.scatter([p[0] for p in d], [p[1] for p in d], s=34, color=colour,
                   alpha=0.8, edgecolor="white", linewidth=0.5,
                   label=f"run L, {label}  ({len(d)} points, fitted)")
    mx, my = st.mean(xs), st.mean(ys)
    slope = sum((a - mx) * (b - my) for a, b in zip(xs, ys)) / sum((a - mx) ** 2 for a in xs)
    inter = my - slope * mx
    brk = -inter / slope
    num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    den = (sum((a - mx) ** 2 for a in xs) * sum((b - my) ** 2 for b in ys)) ** 0.5

    oos = []
    for pat, bpat in (("matrix_K1_sweep_*/spec-dflash-*__rep*.json", "matrix_K1_sweep_*/baseline__rep*.json"),
                      ("matrix_J2_*/spec-*__rep*.json", "matrix_J2_*/baseline__rep*.json")):
        arms = load(pat); base = load(bpat)["baseline"]
        bp = pooled(base)
        for a, runs in arms.items():
            dn = sum(x["draft_n"] for r in runs for x in r["rows"])
            da = sum(x["draft_n_accepted"] for r in runs for x in r["rows"])
            if dn:
                oos.append((100 * da / dn, 100 * (pooled(runs) / bp - 1)))
    ax.scatter([p[0] for p in oos], [p[1] for p in oos], s=78, facecolor="none",
               edgecolor="#3f3f46", linewidth=1.4, marker="D",
               label=f"runs J and K, whole arms ({len(oos)} points, NOT fitted)")

    # the same fit restricted to prompts where every arm ran to the same length
    xs_lm, ys_lm = [], []
    for label, pat, colour in fit:
        for a, b in series(pat, only="dflash", length_matched=True):
            xs_lm.append(a)
            ys_lm.append(b)
    mxl, myl = st.mean(xs_lm), st.mean(ys_lm)
    slope_lm = (sum((a - mxl) * (b - myl) for a, b in zip(xs_lm, ys_lm))
                / sum((a - mxl) ** 2 for a in xs_lm))
    brk_lm = -(myl - slope_lm * mxl) / slope_lm

    record("acceptance_threshold", fitted=sorted(zip(xs, ys)),
           out_of_sample=sorted(oos), slope=slope, intercept=inter,
           break_even=brk, r=num / den,
           break_even_length_matched=brk_lm, slope_length_matched=slope_lm,
           n_fitted=len(xs), n_fitted_length_matched=len(xs_lm))

    lo, hi = 15, 95
    ax.plot([lo, hi], [slope * lo + inter, slope * hi + inter], color="#7a7a82",
            lw=1.3, ls="-", label=f"least squares on run L  (r = {num / den:+.3f})")
    ax.axhline(0, color="#3f6d9e", ls="--", lw=1.1)
    ax.axvline(brk, color="#4a7c59", ls=":", lw=1.6)
    ax.annotate(f"break-even, {brk:.1f} % accepted", (brk, ax.get_ylim()[1]),
                textcoords="offset points", xytext=(6, -14), fontsize=9, color="#4a7c59")
    ax.set_xlim(lo, hi)
    ax.set_xlabel("draft tokens accepted (%)")
    ax.set_ylabel("change in decode rate against the matched baseline (%)")
    ax.set_title("Acceptance decides the sign, and only the sign\n"
                 "line fitted on run L alone; runs J and K are the out-of-sample test",
                 fontsize=11.5)
    ax.legend(loc="upper left", fontsize=8.6, frameon=False)
    ax.grid(color="#dcdcdc", lw=0.6)
    ax.set_axisbelow(True)
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    # This figure spans both workloads and five repeats, so the shared footer
    # ("thinking on, three repeats") would be wrong on it.
    _footer(fig, " Acceptance is llama.cpp's server-side counter; it under-reports on "
                 "arms that take speculative checkpoints (ERRATA A13), which is every "
                 "external-drafter point here and none of the fitted ones. "
                 "Half the fitted points are run L's thinking-off half, where the arms "
                 "generated DIFFERENT numbers of tokens (ERRATA A17): refitting without "
                 "that confound moves the crossing to 46.5 %, and the thinking-on half "
                 "alone gives 45.4 %. Read the threshold as 45-48 %.",
            base="2026-08-26, llama.cpp 3737e4137, one RTX 3090, "
                 "Qwen3.6-35B-A3B-UD-Q4_K_XL, greedy, ten prompts. Fitted points: run L, "
                 "both workloads, five repeats per arm, per prompt. Out-of-sample points: "
                 "runs J and K, whole arms, three repeats. "
                 "Controls and caveats: ERRATA.md, v4_audit_2026_08_25/README.md.")
    if not CHECK:  # --check verifies the numbers, it must not dirty the tree
        plt.savefig(OUT / "plot_acceptance_threshold.png", dpi=150, bbox_inches="tight")
    plt.close()
    if not CHECK:
        print(f"  wrote {(OUT / 'plot_acceptance_threshold.png').relative_to(ROOT)}")


# ----------------------------------------------------------------- run O ----
def plot_head_to_head() -> None:
    """Every method against one baseline, in one matrix, under one policy.

    Bars are coloured by what the drafter IS, not by how it scored, because
    that is the finding: the winners and losers separate by drafter
    architecture and not by acceptance, draft length, or n-gram versus model.
    """
    # Prefer run O2, the balanced Latin square, and draw its block-level
    # intervals. Run O is the same arms at three repeats with the list merely
    # reversed on odd repeats, which leaves arm position confounded with time.
    src = sorted(glob.glob(str(DATA / "matrix_O2_latin_*")))
    tag = "O2"
    if not src:
        src = sorted(glob.glob(str(DATA / "matrix_O_headtohead_*")))
        tag = "O"
    if not src:
        print("  no head-to-head data - skipping chart")
        return
    run = Path(src[-1])
    arms = load(f"{run.name}/*__rep*.json")
    if "baseline" not in arms:
        print("  no baseline in the head-to-head run - skipping chart")
        return
    ci = {}
    pbj = run / "paired_blocks.json"
    if pbj.exists():
        d = json.loads(pbj.read_text(encoding="utf-8"))
        ci = {a["arm"]: a for a in d["arms"]}
        n_blocks = d["blocks"]
    else:
        n_blocks = None
    base = pooled(arms["baseline"])
    # Grouped by what the draft path IS, not by a causal claim: all three load a
    # separate drafter GGUF through -md.
    FAMILY = {
        "spec-dflash": ("DFlash draft head (purpose-built for this target)", C_DFLASH),
        "spec-mtp":    ("MTP draft head (exported from the target)", "#2f7d5a"),
        "spec-draft":  ("general-purpose 0.8 B draft model", C_ACTIVE),
        "ngram":       ("drafter-free n-gram", C_INACTIVE),
        "baseline":    ("no speculation", C_REF),
    }
    def fam(a):
        for k in FAMILY:
            if a.startswith(k):
                return k
        return "ngram"
    rows = []
    for a, runs in arms.items():
        dn = sum(x["draft_n"] for r in runs for x in r["rows"])
        da = sum(x["draft_n_accepted"] for r in runs for x in r["rows"])
        rows.append((a, pooled(runs), (100*da/dn) if dn else None, fam(a)))
    record("head_to_head", rows=[{"arm": a, "pooled": pl,
                                 "acceptance_pct": ac, "family": fm}
                                for a, pl, ac, fm in rows])
    rows.sort(key=lambda r: r[1])

    fig, ax = plt.subplots(figsize=(10.4, 6.2))
    y = range(len(rows))
    for i, (a, p_, acc, f) in enumerate(rows):
        ax.barh(i, p_, height=0.66, color=FAMILY[f][1], alpha=0.92,
                edgecolor="white", linewidth=0.6)
        d = 100*(p_/base - 1)
        c = ci.get(a)
        if c:
            lo, hi = c["ci95_t_pct"]
            # the interval, drawn on the bar, in the same pooled units
            x_lo, x_hi = base * (1 + lo/100), base * (1 + hi/100)
            ax.plot([x_lo, x_hi], [i, i], color="#2f2f2f", lw=1.1, alpha=0.85,
                    solid_capstyle="butt", zorder=3)
            for x in (x_lo, x_hi):
                ax.plot([x, x], [i - 0.16, i + 0.16], color="#2f2f2f", lw=1.1,
                        alpha=0.85, zorder=3)
            dtxt = f"{c['point_pct']:+.1f} % [{lo:+.1f}, {hi:+.1f}]"
        else:
            dtxt = f"{d:+.1f} %"
        lbl = f"{p_:.1f}   {dtxt}" + (f"   acc {acc:.1f} %" if acc is not None else "")
        ax.text(max(p_, x_hi if c else p_) + 2.0, i, lbl, va="center", ha="left",
                fontsize=8.4, color="#1f1f24")
    ax.set_yticks(list(y))
    ax.set_yticklabels([r[0] for r in rows], fontsize=9)
    ax.axvline(base, color=C_REF, ls="--", lw=1.2)
    ax.text(base, len(rows)-0.35, f" no speculation, {base:.1f}", color=C_REF,
            fontsize=8.8, va="top", ha="left")
    ax.set_xlim(0, max(r[1] for r in rows) * 1.46)
    ax.set_xlabel("pooled decode throughput (tokens / second)  -  higher is faster")
    ax.set_title(f"Eight speculative configurations and one baseline, "
                 f"balanced Latin square, {n_blocks or '?'} blocks\n"
                 f"an observational ranking: these arms differ in several ways at once, "
                 f"so no single cause is isolated",
                 fontsize=11.5)
    handles = [plt.Rectangle((0, 0), 1, 1, color=v[1], label=v[0])
               for v in FAMILY.values()]
    ax.legend(handles=handles, loc="lower right", fontsize=8.4, frameon=False)
    ax.grid(axis="x", color="#dcdcdc", lw=0.6)
    ax.set_axisbelow(True)
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    # This figure is nine Latin-square blocks, not the three repeats the shared
    # footer describes, and spec-draft-n1's figure moved when it was re-measured.
    _footer(fig,
            base=f"2026-08-26, llama.cpp 3737e4137, one RTX 3090, "
                 f"Qwen3.6-35B-A3B-UD-Q4_K_XL, greedy, thinking on, ten prompts. "
                 f"Run O2: {n_blocks} balanced Latin-square blocks, every arm at every "
                 f"position exactly once. Controls and caveats: ERRATA.md, "
                 f"v4_audit_2026_08_25/README.md.",
            extra=" Bars are pooled decode rate; the interval is a 95 % paired block "
                  "interval against the baseline measured in the same block. Acceptance "
                  "is the server-side counter, which under-reports on the checkpointing "
                  "rows (ERRATA A13): spec-draft-n1 reads 69.7 % here and 100.0 % from "
                  "the drafter, and is 74.8 % slower either way.")
    if not CHECK:  # --check verifies the numbers, it must not dirty the tree
        plt.savefig(OUT / "plot_head_to_head.png", dpi=150, bbox_inches="tight")
    plt.close()
    if not CHECK:
        print(f"  wrote {(OUT / 'plot_head_to_head.png').relative_to(ROOT)}")


def main() -> None:
    check = "--check" in sys.argv
    if check:
        # draw to a file-free backend: --check verifies the numbers, and
        # rewriting the PNGs would make a read-only check dirty the tree
        import matplotlib
        matplotlib.use("Agg")
    plot_batching()
    plot_dflash_sweep()
    plot_acceptance_threshold()
    plot_head_to_head()
    ref = ROOT / "analysis" / "plot_data.json"
    if check:
        if not ref.exists():
            sys.exit(f"{ref} is missing; run this script without --check first")
        want = json.loads(ref.read_text(encoding="utf-8"))
        if want != SERIES:
            bad = sorted({k for k in set(want) | set(SERIES)
                          if want.get(k) != SERIES.get(k)})
            sys.exit(f"the committed charts are stale: {bad} no longer match the "
                     f"data. Re-run `python analysis/plot_v4_runs.py`.")
        print(f"  charts match the data ({len(SERIES)} series)")
        return
    ref.write_text(json.dumps(SERIES, indent=2, sort_keys=True) + "\n",
                   encoding="utf-8")
    print(f"  wrote {ref.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
