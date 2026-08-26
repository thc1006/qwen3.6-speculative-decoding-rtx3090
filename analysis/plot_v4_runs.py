"""Charts for the 2026-08-26 runs: batching (I) and the DFlash sweep (J, K).

Two figures, and the choice of quantity in each is the point.

  plot_batching.png      aggregate throughput - generated tokens over the
                         wall-clock of the whole prompt set - because at
                         concurrency > 1 a per-request rate is not system
                         throughput. The achieved batch width is read out of the
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
        seen = {r.get("max_in_flight") for r in arms["baseline"] + arms["spec-draft-n8"]}
        widths.append(sorted(x for x in seen if x is not None))

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
    ax.set_xticklabels([f"{c}\nbatch width observed: "
                        f"{', '.join(str(w) for w in widths[i])}"
                        for i, c in enumerate(levels)], fontsize=9)
    ax.set_xlabel("requests in flight")
    ax.set_ylabel("aggregate throughput (generated tokens / wall-clock second)")
    ax.set_ylim(0, max(base_y) * 1.18)
    ax.set_title("Batching helps the target and does nothing for the drafter\n"
                 "so the gap widens: 0.28x at one request in flight, 0.16x at eight",
                 fontsize=11.5)
    # Below the axes: the only large empty region is where the c=1 labels sit.
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2,
              fontsize=9, frameon=False)
    ax.grid(axis="y", color="#dcdcdc", lw=0.6)
    ax.set_axisbelow(True)
    plt.tight_layout(rect=[0, 0.075, 1, 1])
    _footer(fig, " Error bars are the run-to-run SD of three repeats.")
    plt.savefig(OUT / "plot_batching.png", dpi=150, bbox_inches="tight")
    plt.close()
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
    plt.savefig(OUT / "plot_dflash_sweep.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  wrote {(OUT / 'plot_dflash_sweep.png').relative_to(ROOT)}")


# ----------------------------------------------------------------- run L ----
def plot_acceptance_threshold() -> None:
    """Acceptance against speed-up, with the out-of-sample points drawn too.

    The line is fitted on run L only. Runs J and K are plotted as open markers
    because they never informed it - ERRATA A10 is a single-regressor fit that
    looked excellent in sample and was falsified out of it, so the chart has to
    show the test, not just the fit.
    """
    def series(pattern, only=None):
        """Per-prompt (acceptance, delta-vs-baseline) points.

        The baseline arm has to be loaded even when it is not plotted, because
        it is the denominator; filtering the glob instead of the arms is what
        broke the first version of this function.
        """
        per, acc = defaultdict(lambda: defaultdict(list)), defaultdict(lambda: defaultdict(lambda: [0, 0]))
        for f in glob.glob(str(DATA / pattern)):
            r = json.load(open(f, encoding="utf-8"))
            for x in r["rows"]:
                per[x["tag"]][r["arm"]].append(x["predicted_per_second"])
                a = acc[x["tag"]][r["arm"]]; a[0] += x["draft_n_accepted"]; a[1] += x["draft_n"]
        out = []
        for t in per:
            if "baseline" not in per[t]:
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
                 "external-drafter point here and none of the fitted ones.",
            base="2026-08-26, llama.cpp 3737e4137, one RTX 3090, "
                 "Qwen3.6-35B-A3B-UD-Q4_K_XL, greedy, ten prompts. Fitted points: run L, "
                 "both workloads, five repeats per arm, per prompt. Out-of-sample points: "
                 "runs J and K, whole arms, three repeats. "
                 "Controls and caveats: ERRATA.md, v4_audit_2026_08_25/README.md.")
    plt.savefig(OUT / "plot_acceptance_threshold.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  wrote {(OUT / 'plot_acceptance_threshold.png').relative_to(ROOT)}")


# ----------------------------------------------------------------- run O ----
def plot_head_to_head() -> None:
    """Every method against one baseline, in one matrix, under one policy.

    Bars are coloured by what the drafter IS, not by how it scored, because
    that is the finding: the winners and losers separate by drafter
    architecture and not by acceptance, draft length, or n-gram versus model.
    """
    arms = load("matrix_O_headtohead_*/*__rep*.json")
    if "baseline" not in arms:
        print("  no run O data - skipping head-to-head chart")
        return
    base = pooled(arms["baseline"])
    FAMILY = {
        "spec-dflash": ("self-speculative (target's own layers)", C_DFLASH),
        "spec-mtp":    ("self-speculative (target's own MTP head)", "#2f7d5a"),
        "spec-draft":  ("external draft model", C_ACTIVE),
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
    rows.sort(key=lambda r: r[1])

    fig, ax = plt.subplots(figsize=(10.4, 6.2))
    y = range(len(rows))
    for i, (a, p_, acc, f) in enumerate(rows):
        ax.barh(i, p_, height=0.66, color=FAMILY[f][1], alpha=0.92,
                edgecolor="white", linewidth=0.6)
        d = 100*(p_/base - 1)
        lbl = f"{p_:.1f}   {d:+.1f} %" + (f"   acc {acc:.1f} %" if acc is not None else "")
        ax.text(p_ + 2.0, i, lbl, va="center", ha="left", fontsize=8.8, color="#1f1f24")
    ax.set_yticks(list(y))
    ax.set_yticklabels([r[0] for r in rows], fontsize=9)
    ax.axvline(base, color=C_REF, ls="--", lw=1.2)
    ax.text(base, len(rows)-0.35, f" no speculation, {base:.1f}", color=C_REF,
            fontsize=8.8, va="top", ha="left")
    ax.set_xlim(0, max(r[1] for r in rows) * 1.34)
    ax.set_xlabel("pooled decode throughput (tokens / second)  -  higher is faster")
    ax.set_title("Nine methods, one baseline, one matrix, one memory policy\n"
                 "the divide is whether the drafter is a second model - not acceptance, "
                 "not draft length",
                 fontsize=11.5)
    handles = [plt.Rectangle((0, 0), 1, 1, color=v[1], label=v[0])
               for v in FAMILY.values()]
    ax.legend(handles=handles, loc="lower right", fontsize=8.4, frameon=False)
    ax.grid(axis="x", color="#dcdcdc", lw=0.6)
    ax.set_axisbelow(True)
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    _footer(fig, " spec-draft-n1 accepts 69.7 % of its drafts and is 73 % slower: "
                 "acceptance does not decide this.")
    plt.savefig(OUT / "plot_head_to_head.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  wrote {(OUT / 'plot_head_to_head.png').relative_to(ROOT)}")


if __name__ == "__main__":
    plot_batching()
    plot_dflash_sweep()
    plot_acceptance_threshold()
    plot_head_to_head()
