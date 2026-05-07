"""Plot spec-decode bench results.

Reads every *.json under results/ and results/verify/ produced by bench_runner.py,
and emits:
    analysis/plot_mean_by_config.png    — mean tok/s per config (sorted)
    analysis/plot_per_prompt.png        — per-prompt heatmap
    analysis/plot_accept_vs_speed.png   — draft accept rate vs decode rate scatter
    analysis/summary.csv                — all numbers, for the repo

Run: python analysis/plot.py  (from repo root)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULT_DIRS = [ROOT / "results", ROOT / "results/verify"]
OUT_DIR = ROOT / "analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 2026-05-08 update annotation — applied to all v1 charts as a small footer.
# Originally (2026-04-26) said "hardware-class-independent" based on the
# A100 NVLink Δ −11.4 % datapoint. The v3 clean A/B retest in the sibling
# repo (cache-OFF) flipped the vLLM 2× RTX 3090 sign to +27.5 %, and the
# v3.0 README correction (commit 3eef116) reframes the negative finding
# as **engine + spec-method specific to llama.cpp draft-spec on consumer
# Ampere with Q4 target**, not hardware-class-independent. The v3 DFlash
# bench (2026-05-07) confirms the same direction for DFlash specifically.
V22_FOOTER = (
    "Updated 2026-05-08 · negative finding is engine+spec-method specific "
    "to llama.cpp draft-spec / DFlash on consumer Ampere + Q4 target; vLLM "
    "MTP on 2× RTX 3090 PCIe is +27.5 % (sibling repo qwen3.6-vllm-2x3090 "
    "v3/v4). DFlash on 3090: NET LOSS −44.6 % (v3_dflash_2026_05_07/)."
)


def _v22_footer(fig):
    fig.text(
        0.5, 0.005, V22_FOOTER,
        ha="center", va="bottom",
        fontsize=7.6, color="#666666", style="italic",
    )


def load_all() -> pd.DataFrame:
    rows = []
    for d in RESULT_DIRS:
        if not d.exists():
            continue
        for f in sorted(d.glob("*.json")):
            try:
                obj = json.loads(f.read_text(encoding="utf-8"))
            except Exception as e:
                print(f"  skip {f}: {e}", file=sys.stderr)
                continue
            config = obj.get("config", f.stem)
            meta = obj.get("meta", {}) or {}
            for r in obj.get("rows", []):
                rows.append({
                    "config":        config,
                    "source_file":   f.name,
                    "prompt":        r.get("tag"),
                    "tok_s":         r.get("predicted_per_second", 0),
                    "wall_ms":       r.get("wall_ms", 0),
                    "predicted_ms":  r.get("predicted_ms", 0),
                    "predicted_n":   r.get("predicted_n", 0),
                    "prompt_n":      r.get("prompt_tokens", 0),
                    "draft_n":       r.get("draft_n", 0),
                    "draft_acc":     r.get("draft_n_accepted", 0),
                    "max_tokens":    meta.get("max_tokens", 300),
                    "fa":            meta.get("fa", True),
                    "kv_q8":         meta.get("kv_q8", True),
                    "commit":        obj.get("llama_cpp_commit", "?"),
                })
    return pd.DataFrame(rows)


def plot_mean_by_config(df: pd.DataFrame):
    if df.empty: return
    agg = (df.groupby("config")["tok_s"]
             .agg(["mean", "min", "max", "std", "count"])
             .sort_values("mean", ascending=True))
    fig, ax = plt.subplots(figsize=(11, max(4, 0.4 * len(agg))))
    colors = ["#d62728" if m < 130 else "#ff7f0e" if m < 140 else "#2ca02c"
              for m in agg["mean"]]
    ax.barh(
        agg.index, agg["mean"],
        xerr=agg["std"], color=colors, alpha=0.85,
        error_kw=dict(ecolor="#404040", lw=0.9, alpha=0.85, capsize=2.5),
    )
    # Place number labels past the error-bar max so they never overlap the
    # whisker/cap (high-stdev red configs had labels struck through before).
    max_x = max(row["mean"] + (row["std"] if row["std"] == row["std"] else 0) for _, row in agg.iterrows())
    label_pad = 3.0
    for i, (cfg, row) in enumerate(agg.iterrows()):
        std = row["std"] if row["std"] == row["std"] else 0  # NaN-safe
        xpos = row["mean"] + std + label_pad
        ax.text(xpos, i, f"{row['mean']:.1f} (±{std:.1f})",
                va="center", ha="left", fontsize=9, color="#1f1f24")
    ax.set_xlim(0, max_x + 25)
    ax.axvline(x=107, color="gray", linestyle="--", linewidth=1, label="Ollama Q4_K_M (107 tok/s)")
    ax.set_xlabel("Decode speed (tokens / second)")
    ax.set_title("Qwen3.6-35B-A3B UD-Q4_K_XL on RTX 3090 (single GPU, batch=1)")
    ax.legend(loc="lower right")
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    _v22_footer(fig)
    plt.savefig(OUT_DIR / "plot_mean_by_config.png", dpi=140, bbox_inches="tight")
    plt.close()
    print(f"  wrote {OUT_DIR / 'plot_mean_by_config.png'}")


def plot_per_prompt_heatmap(df: pd.DataFrame):
    if df.empty: return
    pivot = df.pivot_table(index="config", columns="prompt", values="tok_s", aggfunc="mean")
    # order prompts consistently
    preferred = ["short_greet", "short_q", "medium_chat", "medium_rec",
                 "reasoning", "long_explain", "code_small",
                 "multi_turn_1", "multi_turn_2", "zh_cn"]
    cols = [c for c in preferred if c in pivot.columns] + \
           [c for c in pivot.columns if c not in preferred]
    pivot = pivot[cols]
    fig, ax = plt.subplots(figsize=(max(8, 0.8 * len(cols)), max(4, 0.3 * len(pivot))))
    im = ax.imshow(pivot.values, cmap="RdYlGn", vmin=60, vmax=145, aspect="auto")
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=30, ha="right")
    ax.set_yticks(range(len(pivot.index))); ax.set_yticklabels(pivot.index)
    for i in range(len(pivot.index)):
        for j in range(len(cols)):
            v = pivot.values[i, j]
            if np.isnan(v): continue
            ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                    color="black" if v > 100 else "white", fontsize=8)
    plt.colorbar(im, ax=ax, label="tok / s")
    ax.set_title("Decode speed per prompt × config")
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    _v22_footer(fig)
    plt.savefig(OUT_DIR / "plot_per_prompt.png", dpi=140, bbox_inches="tight")
    plt.close()
    print(f"  wrote {OUT_DIR / 'plot_per_prompt.png'}")


def plot_accept_vs_speed(df: pd.DataFrame):
    with_draft = df[df["draft_n"] > 0].copy()
    if with_draft.empty:
        print("  no draft stats, skip accept-vs-speed plot")
        return
    with_draft["accept_rate"] = with_draft["draft_acc"] / with_draft["draft_n"].clip(lower=1)
    fig, ax = plt.subplots(figsize=(8, 6))
    for cfg, sub in with_draft.groupby("config"):
        ax.scatter(sub["accept_rate"] * 100, sub["tok_s"], label=cfg, s=60, alpha=0.7)
    ax.axhline(y=135.7, color="gray", linestyle="--", label="baseline 135.7 tok/s")
    ax.set_xlabel("Draft acceptance rate (%)")
    ax.set_ylabel("Decode speed (tok / s)")
    ax.set_title("Anomaly: 100% acceptance does not imply speedup on MoE")
    ax.legend(fontsize=8, loc="lower left")
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    _v22_footer(fig)
    plt.savefig(OUT_DIR / "plot_accept_vs_speed.png", dpi=140, bbox_inches="tight")
    plt.close()
    print(f"  wrote {OUT_DIR / 'plot_accept_vs_speed.png'}")


def write_summary_csv(df: pd.DataFrame):
    if df.empty: return
    df.to_csv(OUT_DIR / "summary.csv", index=False)
    print(f"  wrote {OUT_DIR / 'summary.csv'}")
    # Aggregated
    agg = (df.groupby("config")
             .agg(mean_tok_s=("tok_s", "mean"),
                  min_tok_s=("tok_s", "min"),
                  max_tok_s=("tok_s", "max"),
                  std_tok_s=("tok_s", "std"),
                  runs=("tok_s", "count"),
                  total_draft=("draft_n", "sum"),
                  total_accept=("draft_acc", "sum"))
             .sort_values("mean_tok_s", ascending=False))
    agg["accept_rate_pct"] = np.where(agg["total_draft"] > 0,
                                       100 * agg["total_accept"] / agg["total_draft"].clip(lower=1),
                                       np.nan)
    agg.to_csv(OUT_DIR / "summary_by_config.csv")
    print(f"  wrote {OUT_DIR / 'summary_by_config.csv'}")
    print("\n=== SUMMARY BY CONFIG ===")
    print(agg.to_string(float_format=lambda x: f"{x:.1f}" if not np.isnan(x) else "-"))


if __name__ == "__main__":
    df = load_all()
    print(f"loaded {len(df)} rows from {df['config'].nunique()} configs")
    plot_mean_by_config(df)
    plot_per_prompt_heatmap(df)
    plot_accept_vs_speed(df)
    write_summary_csv(df)
