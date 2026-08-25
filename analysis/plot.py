"""Aggregate and plot the v1 speculative-decoding matrix.

Reads every *.json under results/ and results/verify/ produced by bench_runner.py
and emits:

    analysis/summary.csv                    - one row per measured request
    analysis/summary_by_config.csv          - per-config aggregate, all four summaries
    analysis/plot_mean_by_config.png        - request-mean vs pooled decode throughput
    analysis/plot_per_prompt.png            - per-prompt heatmap, % of matched baseline
    analysis/plot_acceptance_accounting.png - what the "100 % acceptance" number means

Metric definitions (see README "Metric definitions"):

  request-mean   arithmetic mean of the per-request `predicted_per_second`;
                 every prompt gets equal weight.
  pooled         1000 * sum(predicted_n) / sum(predicted_ms); every generated
                 token gets equal weight. Equals the harmonic mean of the
                 per-request rates when all outputs have the same length.
  spread         min-max across the ten prompts. One measurement per
                 prompt/config, so this is workload heterogeneity - NOT
                 repeated-run uncertainty, standard error, or a CI.

Counter semantics (2026-08-25 audit): `draft_n` / `draft_n_accepted` as
reported by llama-server at commit 97895129e count only draft tokens from
verification rounds that were accepted in full. On this hybrid Gated-DeltaNet
target the context reports COMMON_CONTEXT_SEQ_RM_TYPE_FULL, so partially
accepted rounds take an early `continue` and never reach either counter. The
ratio is therefore 1.0 by construction and `draft_n = 0` means "no fully
accepted round was recorded", not "speculation did not run". See
analysis/verbose_accounting.py and ERRATA.md item A1.

Run: python analysis/plot.py  (from repo root)
"""
from __future__ import annotations

import csv
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
RESULT_DIRS = [ROOT / "results", ROOT / "results/verify"]
OUT_DIR = ROOT / "analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PROMPT_ORDER = ["short_greet", "short_q", "medium_chat", "medium_rec",
                "reasoning", "long_explain", "code_small",
                "multi_turn_1", "multi_turn_2", "zh_cn"]

# `zh_cn` is the tag emitted by bench_runner.py; the prompt itself is
# Traditional Chinese (ERRATA C2). The data export keeps the historical tag so
# summary.csv still joins against the raw JSON; only the charts relabel it.
TAG_RENAME = {"zh_cn": "zh_hant"}


def display_tag(tag: str) -> str:
    return TAG_RENAME.get(tag, tag)

FOOTER = (
    "v1 matrix, 2026-04-21, llama.cpp 97895129e, one RTX 3090, Qwen3.6-35B-A3B-UD-Q4_K_XL, "
    "greedy, one measured request per prompt/config. Scope and known confounds: ERRATA.md."
)
SPREAD_NOTE = (
    " Spread bars are min-max across the ten prompts (workload heterogeneity), "
    "not repeated-run uncertainty."
)

C_REF = "#3f6d9e"      # no-speculation reference
C_INACTIVE = "#8d9aa8"  # no draft round recorded
C_ACTIVE = "#c0504d"    # draft rounds recorded


def _footer(fig, extra: str = ""):
    fig.text(0.5, 0.004, FOOTER + extra, ha="center", va="bottom",
             fontsize=7.2, color="#5a5a5a", style="italic", wrap=True)


# ---------------------------------------------------------------- load ----

def load_all() -> list[dict]:
    rows: list[dict] = []
    for d in RESULT_DIRS:
        if not d.exists():
            continue
        for f in sorted(d.glob("*.json")):
            try:
                obj = json.loads(f.read_text(encoding="utf-8"))
            except Exception as e:  # noqa: BLE001 - keep going over a bad file
                print(f"  skip {f}: {e}", file=sys.stderr)
                continue
            meta = obj.get("meta", {}) or {}
            for r in obj.get("rows", []):
                tag = r.get("tag")
                rows.append({
                    "config":       obj.get("config", f.stem),
                    "source_file":  f.name,
                    "prompt":       tag,
                    "tok_s":        float(r.get("predicted_per_second", 0) or 0),
                    "wall_ms":      float(r.get("wall_ms", 0) or 0),
                    "predicted_ms": float(r.get("predicted_ms", 0) or 0),
                    "predicted_n":  int(r.get("predicted_n", 0) or 0),
                    "prompt_n":     int(r.get("prompt_tokens", 0) or 0),
                    "draft_n":      int(r.get("draft_n", 0) or 0),
                    "draft_acc":    int(r.get("draft_n_accepted", 0) or 0),
                    "max_tokens":   int(meta.get("max_tokens", 300)),
                    "fa":           meta.get("fa", True),
                    "kv_q8":        meta.get("kv_q8", True),
                    "commit":       obj.get("llama_cpp_commit", "?"),
                })
    return rows


def by_config(rows: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        out[r["config"]].append(r)
    return out


def aggregate(rows: list[dict]) -> dict[str, dict]:
    agg: dict[str, dict] = {}
    for cfg, v in by_config(rows).items():
        rates = [r["tok_s"] for r in v]
        tot_n = sum(r["predicted_n"] for r in v)
        tot_ms = sum(r["predicted_ms"] for r in v)
        agg[cfg] = {
            "config": cfg,
            "max_tokens": v[0]["max_tokens"],
            "requests": len(v),
            "request_mean_tok_s": st.mean(rates),
            "pooled_tok_s": 1000 * tot_n / tot_ms if tot_ms else float("nan"),
            "median_tok_s": st.median(rates),
            "min_tok_s": min(rates),
            "max_tok_s": max(rates),
            "across_prompt_sd": st.stdev(rates) if len(rates) > 1 else 0.0,
            "requests_with_draft_rounds": sum(1 for r in v if r["draft_n"] > 0),
            "counted_draft_tokens": sum(r["draft_n"] for r in v),
            "counted_draft_accepted": sum(r["draft_acc"] for r in v),
            "generated_tokens": tot_n,
            "decode_ms": tot_ms,
        }
    return agg


def reference_for(cfg: str, agg: dict[str, dict]) -> str | None:
    """Long-output runs must be compared against the long-output baseline.

    Returns None when that reference is not in the data, so a partial results
    directory produces blank deltas rather than a KeyError.
    """
    ref = "baseline-1000tok" if agg[cfg]["max_tokens"] == 1000 else "baseline"
    return ref if ref in agg else None


# ------------------------------------------------------------ csv out ----

def write_csvs(rows: list[dict], agg: dict[str, dict]) -> None:
    fields = ["config", "source_file", "prompt", "tok_s", "wall_ms", "predicted_ms",
              "predicted_n", "prompt_n", "draft_n", "draft_acc", "max_tokens",
              "fa", "kv_q8", "commit"]
    with (OUT_DIR / "summary.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {(OUT_DIR / 'summary.csv').relative_to(ROOT)}")

    cols = ["config", "max_tokens", "requests", "reference",
            "request_mean_tok_s", "request_mean_delta_pct",
            "pooled_tok_s", "pooled_delta_pct",
            "median_tok_s", "min_tok_s", "max_tok_s", "across_prompt_sd",
            "requests_with_draft_rounds", "counted_draft_tokens",
            "counted_draft_accepted", "generated_tokens", "decode_ms"]
    order = sorted(agg, key=lambda c: -agg[c]["request_mean_tok_s"])
    with (OUT_DIR / "summary_by_config.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for cfg in order:
            a = dict(agg[cfg])
            ref = reference_for(cfg, agg)
            a["reference"] = ref or ""
            if ref:
                a["request_mean_delta_pct"] = round(
                    100 * (a["request_mean_tok_s"] / agg[ref]["request_mean_tok_s"] - 1), 2)
                a["pooled_delta_pct"] = round(
                    100 * (a["pooled_tok_s"] / agg[ref]["pooled_tok_s"] - 1), 2)
            else:
                a["request_mean_delta_pct"] = ""
                a["pooled_delta_pct"] = ""
            for k in ("request_mean_tok_s", "pooled_tok_s", "median_tok_s",
                      "min_tok_s", "max_tok_s", "across_prompt_sd", "decode_ms"):
                a[k] = round(a[k], 3)
            w.writerow({k: a[k] for k in cols})
    print(f"  wrote {(OUT_DIR / 'summary_by_config.csv').relative_to(ROOT)}")


# --------------------------------------------------------------- plots ----

def plot_mean_by_config(agg: dict[str, dict]) -> None:
    cfgs = [c for c in agg if agg[c]["max_tokens"] == 300]
    if not cfgs or "baseline" not in agg:
        print("  no 300-token group with a `baseline` reference - skipping bar chart")
        return
    cfgs.sort(key=lambda c: agg[c]["pooled_tok_s"])
    base = agg["baseline"]

    y = np.arange(len(cfgs))
    h = 0.38
    fig, ax = plt.subplots(figsize=(12.4, max(5.0, 0.62 * len(cfgs))))

    for i, c in enumerate(cfgs):
        a = agg[c]
        color = (C_REF if c == "baseline"
                 else C_ACTIVE if a["requests_with_draft_rounds"] else C_INACTIVE)
        ax.barh(y[i] + h / 2, a["request_mean_tok_s"], height=h,
                color=color, alpha=0.95, edgecolor="white", linewidth=0.6)
        ax.barh(y[i] - h / 2, a["pooled_tok_s"], height=h,
                color=color, alpha=0.55, edgecolor="white", linewidth=0.6,
                hatch="///")
        ax.plot([a["min_tok_s"], a["max_tok_s"]], [y[i] + h / 2] * 2,
                color="#2f2f2f", lw=0.9, alpha=0.8, solid_capstyle="butt")
        for x in (a["min_tok_s"], a["max_tok_s"]):
            ax.plot([x, x], [y[i] + h / 2 - 0.09, y[i] + h / 2 + 0.09],
                    color="#2f2f2f", lw=0.9, alpha=0.8)

        dm = 100 * (a["request_mean_tok_s"] / base["request_mean_tok_s"] - 1)
        dp = 100 * (a["pooled_tok_s"] / base["pooled_tok_s"] - 1)
        ax.text(a["max_tok_s"] + 2.0, y[i] + h / 2,
                f"mean {a['request_mean_tok_s']:.1f} ({dm:+.1f} %)",
                va="center", ha="left", fontsize=8.4, color="#1f1f24")
        ax.text(a["max_tok_s"] + 2.0, y[i] - h / 2,
                f"pooled {a['pooled_tok_s']:.1f} ({dp:+.1f} %)",
                va="center", ha="left", fontsize=8.4, color="#4a4a52")

    labels = []
    for c in cfgs:
        a = agg[c]
        n = a["requests_with_draft_rounds"]
        labels.append(f"{c}\n{n}/{a['requests']} req. with a counted draft round"
                      if n else f"{c}\nno counted draft round")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.6)
    ax.set_ylim(-0.8, len(cfgs) - 0.2)

    ax.axvline(base["request_mean_tok_s"], color="#3f6d9e", ls="--", lw=1.1,
               label=f"no-speculation baseline, {base['request_mean_tok_s']:.1f} tok/s")
    ax.set_xlim(0, max(a["max_tok_s"] for a in agg.values()) + 44)
    ax.set_xlabel("decode rate (tokens / second)  -  higher is faster")
    ax.set_title("v1 300-token matrix: request-mean (solid) vs pooled throughput (hatched)\n"
                 "Qwen3.6-35B-A3B UD-Q4_K_XL, one RTX 3090, single request, greedy",
                 fontsize=11.5)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=C_REF, label="no-speculation reference"),
        plt.Rectangle((0, 0), 1, 1, color=C_INACTIVE, label="no counted draft round"),
        plt.Rectangle((0, 0), 1, 1, color=C_ACTIVE, label="counted draft rounds present"),
        plt.Line2D([0], [0], color="#2f2f2f", lw=0.9,
                   label="min-max across the ten prompts (1 run each)"),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.055),
              ncol=4, fontsize=8.2, frameon=False)
    ax.grid(axis="x", color="#dcdcdc", lw=0.6)
    ax.set_axisbelow(True)
    plt.tight_layout(rect=[0, 0.055, 1, 1])
    _footer(fig, SPREAD_NOTE)
    plt.savefig(OUT_DIR / "plot_mean_by_config.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  wrote {(OUT_DIR / 'plot_mean_by_config.png').relative_to(ROOT)}")


def plot_per_prompt(rows: list[dict], agg: dict[str, dict]) -> None:
    groups = [(cap, title) for cap, title in
              [(300, "300-token cap, vs `baseline`"),
               (1000, "1000-token cap, vs `baseline-1000tok`")]
              if any(agg[c]["max_tokens"] == cap for c in agg)
              and ("baseline" if cap == 300 else "baseline-1000tok") in agg]
    if not groups:
        print("  no group has its matched baseline - skipping heatmap")
        return
    heights = [len([c for c in agg if agg[c]["max_tokens"] == cap]) for cap, _ in groups]
    fig, axes = plt.subplots(
        len(groups), 1, figsize=(11.6, 0.42 * sum(heights) + 3.0),
        gridspec_kw={"height_ratios": heights, "hspace": 0.55},
        constrained_layout=False, squeeze=False)
    axes = [a[0] for a in axes]

    cell = {(r["config"], r["prompt"]): r for r in rows}
    im = None
    for ax, (cap, title) in zip(axes, groups):
        cfgs = [c for c in agg if agg[c]["max_tokens"] == cap]
        cfgs.sort(key=lambda c: -agg[c]["pooled_tok_s"])
        ref_cfg = "baseline-1000tok" if cap == 1000 else "baseline"
        prompts = [p for p in PROMPT_ORDER if (ref_cfg, p) in cell]

        mat = np.full((len(cfgs), len(prompts)), np.nan)
        for i, c in enumerate(cfgs):
            for j, p in enumerate(prompts):
                r, b = cell.get((c, p)), cell.get((ref_cfg, p))
                if r and b and b["tok_s"]:
                    mat[i, j] = 100 * r["tok_s"] / b["tok_s"]

        im = ax.imshow(mat, cmap="RdYlGn", vmin=40, vmax=105, aspect="auto")
        ax.set_xticks(range(len(prompts)))
        ax.set_xticklabels([display_tag(t) for t in prompts],
                           rotation=30, ha="right", fontsize=8.4)
        ax.set_yticks(range(len(cfgs)))
        ax.set_yticklabels(cfgs, fontsize=8.4)
        ax.set_title(title, fontsize=10, pad=6)

        for i, c in enumerate(cfgs):
            for j, p in enumerate(prompts):
                v = mat[i, j]
                if np.isnan(v):
                    continue
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7.6,
                        color="black" if v > 62 else "white")
                if (cell.get((c, p)) or {}).get("draft_n", 0) > 0:
                    ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                           edgecolor="#101010", lw=1.7))

    fig.colorbar(im, ax=axes, label="% of the matched no-speculation baseline",
                 fraction=0.026, pad=0.015)
    fig.suptitle("v1 per-prompt decode rate, normalised to the matched baseline\n"
                 "black outline = this request recorded at least one fully accepted draft round",
                 fontsize=11.5, y=1.0)
    _footer(fig)
    plt.savefig(OUT_DIR / "plot_per_prompt.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  wrote {(OUT_DIR / 'plot_per_prompt.png').relative_to(ROOT)}")


def plot_acceptance_accounting(agg: dict[str, dict]) -> None:
    """Replaces the retracted plot_accept_vs_speed.png (see ERRATA.md item A1)."""
    src = OUT_DIR / "verbose_accounting.json"
    if not src.exists():
        print("  no analysis/verbose_accounting.json - run analysis/verbose_accounting.py first")
        return
    rep = json.loads(src.read_text(encoding="utf-8"))[0]
    d, r, s = rep["drafter_own_counters"], rep["reported"], rep["state_management"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 4.7))

    bars = [
        ("server counter\n(what v1/v2 published)", r["accepted"], r["generated"], "#c0504d"),
        ("drafter token counter\n(same log, next line)", d["draft_tokens_accepted"],
         d["draft_tokens_generated"], "#3f6d9e"),
        ("drafter sequence counter\n(same log, next line)", d["drafts_accepted"],
         d["drafts_generated"], "#6c8c3f"),
    ]
    x = np.arange(len(bars))
    ax1.bar(x, [b[2] for b in bars], color="#d9d9d9", edgecolor="#9a9a9a",
            width=0.56, label="proposed / denominator")
    ax1.bar(x, [b[1] for b in bars], color=[b[3] for b in bars],
            width=0.56, label="accepted / numerator")
    for i, (_, num, den, _c) in enumerate(bars):
        ax1.text(i, den + 4, f"{num}/{den} = {100 * num / den:.1f} %",
                 ha="center", va="bottom", fontsize=9.4, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels([b[0] for b in bars], fontsize=8.4)
    ax1.set_ylabel("draft units")
    ax1.set_ylim(0, max(b[2] for b in bars) * 1.32)
    ax1.set_title("Three acceptance numbers from one run\n"
                  "v2 verbose log, --draft-min 2 --draft-max 32, prompt 1", fontsize=10.5)
    ax1.legend(fontsize=8.2, loc="upper center", bbox_to_anchor=(0.5, -0.21),
               ncol=2, frameon=False)
    ax1.grid(axis="y", color="#e4e4e4", lw=0.6)
    ax1.set_axisbelow(True)

    full = rep["attempts_fully_accepted"]
    part = rep["attempts_partially_accepted"]
    ax2.barh([1], [full], color="#6c8c3f", height=0.5,
             label=f"fully accepted -> reaches the counter ({full})")
    ax2.barh([0], [part], color="#c0504d", height=0.5,
             label=f"partly accepted -> `continue`, discarded, re-verified ({part})")
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(["partial\naccept", "full\naccept"], fontsize=9)
    ax2.set_xlabel("verification rounds")
    ax2.set_xlim(0, max(full, part) * 1.18)
    ax2.set_title("Why the server ratio can only be 1.0\n"
                  "target reports COMMON_CONTEXT_SEQ_RM_TYPE_FULL", fontsize=10.5)
    ax2.legend(fontsize=8.2, loc="upper center", bbox_to_anchor=(0.5, -0.21),
               ncol=1, frameon=False)
    ax2.grid(axis="x", color="#e4e4e4", lw=0.6)
    ax2.set_axisbelow(True)
    fig.suptitle("The published \"100 % draft acceptance\" is a counter artefact, not a measurement",
                 fontsize=12.5, y=0.99)
    plt.tight_layout(rect=[0, 0.135, 1, 0.94])
    fig.text(0.5, 0.058,
             f"Cost of the {part} discarded rounds: {s['checkpoints_created']} state checkpoints "
             f"@ {s['checkpoint_mib_each']} MiB ({s['checkpoint_gib_written']} GiB written, "
             f"{s['checkpoint_gib_read_back']} GiB restored). "
             f"Drafter generate() alone = {rep['drafter_share_of_generation_pct']} % of the "
             f"{rep['generation_wall_ms']} ms generation wall-clock.",
             ha="center", va="bottom", fontsize=8.6, color="#222222")
    fig.text(0.5, 0.005,
             "Source: v2_3090_followup/v2_oleg_suggestions/verbose.log, build b8863-97895129e. "
             "Reconstruct with analysis/verbose_accounting.py. Mechanism: ERRATA.md item A1.",
             ha="center", va="bottom", fontsize=7.4, color="#5a5a5a", style="italic")
    plt.savefig(OUT_DIR / "plot_acceptance_accounting.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  wrote {(OUT_DIR / 'plot_acceptance_accounting.png').relative_to(ROOT)}")


# ---------------------------------------------------------------- main ----

def main() -> None:
    rows = load_all()
    agg = aggregate(rows)
    print(f"loaded {len(rows)} requests from {len(agg)} run labels")
    active = [c for c in agg if agg[c]["counted_draft_tokens"] > 0]
    print(f"  {len(active)} labels recorded at least one fully accepted draft round")
    print(f"  {len(agg) - len(active)} labels recorded none "
          "(baseline, control, or no surviving round)")
    write_csvs(rows, agg)
    plot_mean_by_config(agg)
    plot_per_prompt(rows, agg)
    plot_acceptance_accounting(agg)

    print("\n=== SUMMARY BY CONFIG ===")
    hdr = (f"{'config':24s} {'cap':>5s} {'req-mean':>9s} {'pooled':>8s} "
           f"{'median':>7s} {'min':>7s} {'max':>7s} {'draft-req':>9s} {'draft-tok':>9s}")
    print(hdr)
    print("-" * len(hdr))
    for c in sorted(agg, key=lambda c: -agg[c]["request_mean_tok_s"]):
        a = agg[c]
        print(f"{c:24s} {a['max_tokens']:5d} {a['request_mean_tok_s']:9.1f} "
              f"{a['pooled_tok_s']:8.1f} {a['median_tok_s']:7.1f} {a['min_tok_s']:7.1f} "
              f"{a['max_tok_s']:7.1f} {a['requests_with_draft_rounds']:6d}/{a['requests']:<2d} "
              f"{a['counted_draft_tokens']:9d}")


if __name__ == "__main__":
    main()
