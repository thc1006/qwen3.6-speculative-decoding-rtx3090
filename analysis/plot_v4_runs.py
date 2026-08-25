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

FOOTER = ("2026-08-26, llama.cpp 3737e4137, one RTX 3090, Qwen3.6-35B-A3B-UD-Q4_K_XL, "
          "greedy, thinking on, ten prompts, three repeats per arm. "
          "Controls and caveats: ERRATA.md, v4_audit_2026_08_25/README.md.")


def _footer(fig, extra=""):
    fig.text(0.5, 0.004, FOOTER + extra, ha="center", va="bottom",
             fontsize=7.2, color="#5a5a5a", style="italic", wrap=True)


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


if __name__ == "__main__":
    plot_batching()
    plot_dflash_sweep()
