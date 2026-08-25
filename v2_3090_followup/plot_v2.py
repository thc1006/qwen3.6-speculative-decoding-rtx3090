"""Generate plot_v2_configs.png from results_v2.json.

Audited 2026-08-25 (see ../ERRATA.md). Two label corrections:

  * every non-baseline arm here is an EXTERNAL DRAFT MODEL run (`-md`), so the
    labels now say so. The old "srogmann-style" tag conflated this with v1's
    `--spec-type ngram-mod`, which is a different mechanism and which v2 could
    not run at all: `--spec-type` is registered for llama-server only, and v2
    used llama-cli (ERRATA D6).
  * the footer now states the two facts that decide how these bars may be read:
    one measurement per prompt per config, and a thinking control that did not
    work.

Horizontal bar chart — configs ranked descending by mean gen tok/s.
Design aims:
  - number labels placed past the error-bar max (no collisions)
  - baseline visually anchored with thicker border, no floating label
  - muted editorial palette (FT / Economist style) rather than vivid RGB
  - master-commit cross-check inline in the top-right, not a wide footer
"""
import json
import pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

HERE = pathlib.Path(__file__).resolve().parent
with (HERE / "results_v2.json").open(encoding="utf-8") as f:
    data = json.load(f)

orig = data["runs_original_commit"]
master = data["runs_master_commit"]

# Editorial-ish defaults
rcParams["font.family"] = "DejaVu Sans"
rcParams["axes.edgecolor"] = "#222222"
rcParams["axes.labelcolor"] = "#222222"
rcParams["xtick.color"] = "#555555"
rcParams["ytick.color"] = "#222222"

ORDER = [
    "baseline",
    "srogmann_min48_max64",
    "oleg_draft_2_16",
    "oleg_draft_2_32",
    "oleg_draft_2_64",
    "default_draft_max8",
    "bare_md",
    "default_draft_max16",
    "default_draft_max32",
]
LABELS = {
    "baseline":             "baseline  · no spec-decode",
    "srogmann_min48_max64": "-md  --draft-min 48  --draft-max 64",
    "oleg_draft_2_16":      "-md  --draft-min 2   --draft-max 16",
    "oleg_draft_2_32":      "-md  --draft-min 2   --draft-max 32",
    "oleg_draft_2_64":      "-md  --draft-min 2   --draft-max 64",
    "default_draft_max8":   "default --draft-min=5  --draft-max 8",
    "bare_md":              "bare -md  · all defaults",
    "default_draft_max16":  "default --draft-min=5  --draft-max 16",
    "default_draft_max32":  "default --draft-min=5  --draft-max 32",
}
# Muted palette — deep navy baseline, warm amber for least-bad, soft brick for losses
NAVY   = "#1f3a5f"
AMBER  = "#c97b26"
BRICK  = "#b04a3d"
RULE   = "#d7d2c8"
INK    = "#1f1f24"
MUTED  = "#6e6e76"

def pick_colour(k):
    if k == "baseline":              return NAVY
    if "srogmann" in k:              return AMBER
    return BRICK

rows = []
for k in ORDER:
    s = orig[k]["summary"]
    rows.append({
        "label": LABELS[k],
        "mean":  s["mean_gen"],
        "mn":    s["min_gen"],
        "mx":    s["max_gen"],
        "colour": pick_colour(k),
        "is_baseline": k == "baseline",
    })

fig, ax = plt.subplots(figsize=(12.2, 6.8), dpi=150)
fig.patch.set_facecolor("white")
ax.set_facecolor("#fafaf6")  # warm ivory

ypos = list(range(len(rows)))
means   = [r["mean"] for r in rows]
errs_lo = [r["mean"] - r["mn"] for r in rows]
errs_hi = [r["mx"] - r["mean"] for r in rows]
colours = [r["colour"] for r in rows]
edges   = ["#000000" if r["is_baseline"] else "#2a2a2a" for r in rows]
widths  = [1.6 if r["is_baseline"] else 0.55 for r in rows]
bar_height = 0.62

ax.barh(
    ypos, means, height=bar_height, color=colours, alpha=0.92,
    xerr=[errs_lo, errs_hi], capsize=3,
    error_kw=dict(ecolor=MUTED, lw=0.9, alpha=0.85),
    edgecolor=edges, linewidth=widths,
)

# Number labels placed PAST the error-bar max so they never collide
for i, r in enumerate(rows):
    xpos = r["mean"] + (r["mx"] - r["mean"]) + 2.2
    ax.text(
        xpos, i, f"{r['mean']:.1f}",
        va="center", ha="left",
        fontsize=10.5, color=INK,
        fontweight="bold" if r["is_baseline"] else "regular",
    )

# Soft reference line at baseline (no floating label — the bar itself carries it)
baseline = rows[0]["mean"]
ax.axvline(baseline, color=NAVY, linestyle=(0, (4, 3)), alpha=0.28, linewidth=1.0, zorder=0)

# Y-axis label styling — monospace look but not Courier heavy
ax.set_yticks(ypos)
ax.set_yticklabels([r["label"] for r in rows],
                   fontsize=10.2, family="monospace", color=INK)
ax.invert_yaxis()

ax.set_xlim(0, 170)
ax.set_xticks([0, 20, 40, 60, 80, 100, 120, 140, 160])
ax.set_xlabel("generation rate  (tok / s)",
              fontsize=10.5, color=INK, labelpad=8)

# Title + subtitle, typographic hierarchy — both left-aligned above axes
ax.set_title(
    "Qwen3.6-35B-A3B speculative decoding on RTX 3090   ·   v2 bench\n"
    "$\\mathregular{llama.cpp\\ 97895129e\\ (post\\ PR\\ \\#19493)\\ \\cdot\\ single\\ 3090\\ @\\ stock\\ clocks\\ \\cdot\\ mean\\ of\\ 5\\ prompts\\ \\cdot\\ error\\ bars = min–max}$",
    loc="left", fontsize=13.5, fontweight="bold", color=INK, pad=14,
)
# workaround: subtitle as separate text to get smaller muted second line
ax.text(
    0.0, 1.02,
    "llama.cpp 97895129e (post PR #19493)   ·   single 3090 @ stock clocks   ·   "
    "mean of 5 prompts, one run each   ·   error bars = min–max across prompts",
    transform=ax.transAxes, ha="left", va="bottom",
    fontsize=9.5, color=MUTED,
)
# re-override title to just the headline (subtitle now separate)
ax.set_title("Qwen3.6-35B-A3B speculative decoding on RTX 3090   ·   v2 bench",
             loc="left", fontsize=13.5, fontweight="bold", color=INK, pad=28)

# Grid: vertical gridlines only, subtle
ax.grid(axis="x", linestyle="-", color=RULE, linewidth=0.7, alpha=0.55, zorder=0)
ax.set_axisbelow(True)
for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)
ax.spines["left"].set_color("#888888")
ax.spines["bottom"].set_color("#888888")

# Master-commit cross-check — a single-line footer note plus a compact
# 3-row data summary below the axis label.  No inline overlays (messy).
mb = master["baseline"]["summary"]["mean_gen"]
mo = master["oleg_draft_2_32"]["summary"]["mean_gen"]
ms = master["srogmann_min48_max64"]["summary"]["mean_gen"]
cross_check_note = (
    "cross-checked on llama.cpp master bcb5eeb64 (post PR #22227 "
    "speculative-simple checkpoint)  →  "
    f"baseline {mb:.1f}  ·  --draft-min 2/32 → {mo:.1f}  ·  "
    f"--draft-min 48/64 → {ms:.1f}   "
    "identical within ±0.3 %"
)

# v2.2 update — N=3 trial replication on a fresh standalone 3090 host
# (2026-04-26).  Three configs above × 3 trials × 5 prompts = 15 datapoints
# per config.  Confirms v2.0 numbers are not single-trial flukes.
n3_badge = (
    "N=3 verified 2026-04-26  →  "
    "baseline 139.19  ·  --draft-min 2/32 → 65.24  ·  "
    "--draft-min 48/64 → 85.50   "
    "matches v2.0 within <0.5 pp  ·  run-to-run stdev <0.11 tok/s"
)
fig.text(
    0.5, 0.072, n3_badge,
    ha="center", va="bottom",
    fontsize=8.6, color=NAVY,
    family="monospace", fontweight="bold",
)
fig.text(
    0.5, 0.040, cross_check_note,
    ha="center", va="bottom",
    fontsize=8.6, color=MUTED,
    family="monospace",
)
fig.text(
    0.5, 0.012,
    "audited 2026-08-25 · one run per prompt per config, so bars are not repeat means · "
    "-no-cnv was rejected and /no_think did not disable thinking, so the measured "
    "workload is long chain-of-thought · see ERRATA.md",
    ha="center", va="bottom",
    fontsize=7.8, color="#8a3d3d", family="monospace",
)
fig.text(
    0.995, -0.012,
    "github.com/thc1006/qwen3.6-speculative-decoding-rtx3090   ·   audited v4",
    ha="right", va="bottom",
    fontsize=7.2, color="#9a9a9a", family="monospace",
)

plt.tight_layout(rect=[0.0, 0.10, 1, 1.0])
out = HERE / "plot_v2_configs.png"
plt.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
print(f"Wrote {out}")
