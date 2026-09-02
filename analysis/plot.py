"""Aggregate and plot the v1 speculative-decoding matrix.

Reads every *.json under results/ and results/verify/ produced by bench_runner.py
and emits:

    analysis/summary.csv                    - one row per measured request
    analysis/summary_by_config.csv          - per-config aggregate, all four summaries
    analysis/plot_mean_by_config.png        - request-mean vs pooled decode throughput
    analysis/plot_per_prompt.png            - per-prompt heatmap, % of matched baseline
    analysis/plot_acceptance_accounting.png - what the "100 % acceptance" number means
    analysis/plot_v1_data.json              - what those three figures plot, for --check

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

Run: python analysis/plot.py           (from repo root; writes the files above)
     python analysis/plot.py --check   (re-derives and compares; writes nothing)

Until 2026-09-02 nothing checked these three figures at all: the `charts` job
ran `plot_v4_runs.py --check` and stopped there, so a data change could leave
all three stale and every gate stayed green. `--check` here is that script's
mechanism, for the same reason it gives: a PNG cannot be diffed usefully and
matplotlib output is not byte-stable across versions, so what is compared is
the series each figure plots. The layout guards below run in `--check` too,
because a collision that only fails when somebody opens the PNG is not a check.
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
import matplotlib.collections as mcoll
import matplotlib.colors as mcolors
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.text as mtext
import matplotlib.transforms as mtransforms

# `analysis/` is not on the path when these run from the repository root
import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))
import figstyle                                                    # noqa: E402
figstyle.apply(plt)
import numpy as np
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
RESULT_DIRS = [ROOT / "results", ROOT / "results/verify"]
OUT_DIR = ROOT / "analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CHECK = "--check" in sys.argv
SERIES: dict = {}
SERIES_JSON = OUT_DIR / "plot_v1_data.json"

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

C_REF = figstyle.BLUE      # no-speculation reference
C_INACTIVE = figstyle.GREY  # no draft round recorded
C_ACTIVE = figstyle.VERMILION    # draft rounds recorded

DASH = "—"   # the glyph the README tables already use for an absent cell


def _footer(fig, extra: str = ""):
    """Same reserved, wrapped, standard-size caption as every other figure."""
    figstyle.footer(fig, (FOOTER if not extra else FOOTER + extra))


def record(chart: str, **series) -> None:
    """Register what a chart actually plots, so `--check` can re-derive it.

    Same mechanism and the same reason as `analysis/plot_v4_runs.py`: the PNG
    is not comparable, so the numbers behind it are what CI compares.
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


# -------------------------------------------------------------- guards ----
#
# `analysis/plot_v4_runs.py` grew `_table_guard` and `_view_guard` first and
# these are the same idea, widened by what the 2026-09-02 audit found here.
# The two exclusions in `_fig_texts` are its discoveries and the comment there
# is its comment: both false reports were real, and a guard nobody believes is
# a guard somebody switches off.

def _fig_texts(fig) -> list[tuple]:
    """Every Text the figure actually paints, with the box it paints into.

    Two kinds of Text report a box they never paint. A tick LOCATOR routinely
    puts one tick past the view limit and its label still answers with a
    position out there, where it duly "collides" with whatever is next to it;
    and every tick carries a second, hidden label whose extent is a one-pixel
    box at the origin, which overlaps anything near the origin.
    """
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    skip = set()
    for ax in fig.axes:
        for axis, lim in ((ax.xaxis, sorted(ax.get_xlim())),
                          (ax.yaxis, sorted(ax.get_ylim()))):
            for loc, lab in zip(axis.get_ticklocs(), axis.get_ticklabels()):
                if not lim[0] <= loc <= lim[1]:
                    skip.add(id(lab))
    out = []
    for t in fig.findobj(mtext.Text):
        if not t.get_text().strip() or not t.get_visible() or id(t) in skip:
            continue
        bb = t.get_window_extent(r)
        if bb.width <= 1 or bb.height <= 1:
            continue
        out.append((t, bb))
    return out


def _text_guard(fig, name: str, gap: float = 5.0) -> None:
    """Refuse to publish a figure whose text collides or whose columns are ragged.

    Three faults, all of them shipped in these figures and all of them found by
    reading the PNG rather than the source:

      collision   the dashed baseline rule passed 2.6 px from the first glyph of
                  four labels and the whisker cap abutted the "m" of "mean".
                  A GAP is demanded rather than mere non-overlap, because two
                  boxes that clear each other by a pixel still read as one word.

      ragged      "mean 121.3 (-10.6 %)" against "mean 130.0 (-4.2 %)": the
                  percent field is not a fixed width, so no digit sat under the
                  digit above it. Cells tagged `num:` (a column) or `row:` (a
                  row) must render at the SAME WIDTH, which is what the figure
                  spaces in `figstyle.fig_num` are for.

      drifting    every one of those labels was anchored to `max_tok_s + 2.0`,
                  a DATA position, so the column's left edges staggered across
                  3.9 tok/s and "pooled 109.9" was printed at x = 137. A tagged
                  group must therefore share ONE anchor coordinate. This is not
                  the vacuous check `plot_v4_runs.py` warns about - that one
                  compared x0 against x1 and could not see an alignment change,
                  which is what the width half catches. Sharing an anchor is
                  exactly what a cell placed in data coordinates cannot do.
    """
    boxes = _fig_texts(fig)
    bad = [f"{a.get_text().replace(chr(10), '|')[:44]!r} and "
           f"{b.get_text().replace(chr(10), '|')[:44]!r} are closer than {gap} px"
           for i, (a, x) in enumerate(boxes) for b, y in boxes[i + 1:]
           if mtransforms.Bbox.from_extents(x.x0 - gap, x.y0 - gap,
                                            x.x1 + gap, x.y1 + gap).overlaps(y)]
    groups: dict = defaultdict(list)
    for t, bb in boxes:
        gid = t.get_gid() or ""
        if gid.startswith(("num:", "row:")):
            groups[gid].append((t, bb))
    for gid, cells in sorted(groups.items()):
        vertical = gid.startswith("num:")     # a column: the cells share an x
        widths = {round(bb.width, 1) for t, bb in cells
                  if any(c.isdigit() for c in t.get_text())}
        aligns = {(t.get_ha() if vertical else t.get_va()) for t, _ in cells}
        if len(widths) > 1:
            bad.append(f"{gid} is ragged: its numbers render at widths "
                       f"{sorted(widths)}, so their digits cannot line up")
        if len(aligns) != 1:
            bad.append(f"{gid} mixes alignments {sorted(aligns)}")
        anchors = set()
        for t, bb in cells:
            if vertical:
                a = t.get_ha()
                anchors.add(round(bb.x0 if a == "left" else bb.x1 if a == "right"
                                  else (bb.x0 + bb.x1) / 2, 1))
            else:
                a = t.get_va()
                anchors.add(round(bb.y0 if a in ("bottom", "baseline")
                                  else bb.y1 if a == "top"
                                  else (bb.y0 + bb.y1) / 2, 1))
        if max(anchors) - min(anchors) > 0.6:
            bad.append(
                f"{gid} does not line up: its cells are anchored across "
                f"{max(anchors) - min(anchors):.1f} px of "
                f"{'x' if vertical else 'y'}, so it is not a "
                f"{'column' if vertical else 'row'}")
    if bad:
        raise SystemExit(f"  {name}: the figure is not readable\n    "
                         + "\n    ".join(bad))


def _marks(ax) -> list[tuple[str, float, float, float, float]]:
    """The data extent of every artist on `ax` that stands for a measurement.

    Not a list maintained by hand - that is how `plot_v4_runs.py`'s first
    collision check came to miss the title. Every Line2D has to declare itself
    as a mark or as a rule (a grid line, an axvline, a fit drawn to the edge on
    purpose) by its gid, so a line added later fails loudly instead of being
    quietly skipped.
    """
    out = []
    for p in ax.patches:
        if (p.get_gid() or "") == "on-fill":     # sits on a cell, not on the axes
            continue
        b = p.get_bbox()
        out.append((f"patch {p.get_gid() or ''}", b.x0, b.x1, b.y0, b.y1))
    for ln in ax.lines:
        gid = ln.get_gid() or ""
        if gid == "rule":
            continue
        if gid != "mark":
            raise SystemExit(
                f"  a Line2D on this axes has gid {gid!r}; every line must "
                "declare itself 'mark' (a measurement) or 'rule' (a reference "
                "line drawn to the edge on purpose)")
        d = ln.get_xydata()
        if len(d):
            out.append(("line", float(np.min(d[:, 0])), float(np.max(d[:, 0])),
                        float(np.min(d[:, 1])), float(np.max(d[:, 1]))))
    for c in ax.collections:
        # PathCollection only: `errorbar` leaves a LineCollection whose
        # get_offsets() answers with its default [[0, 0]] rather than anything
        # it drew, which is two phantom points at the origin on every figure
        # with error bars.
        if isinstance(c, mcoll.PathCollection):
            for q in c.get_offsets():
                out.append(("point", float(q[0]), float(q[0]),
                            float(q[1]), float(q[1])))
    return out


def _range_guard(ax, name: str, axis: str = "x", slack: float = 0.10) -> None:
    """Refuse an axis that hides a mark, or that reserves range for text.

    Both halves are defects these figures shipped. The bar chart set
    `xlim(0, max + 44)` so that the labels drawn past the bars would fit: 24 %
    of the range existed only to hold text, and the same trick cost the
    acceptance figure 24 % of one panel and 15 % of the other. Text belongs
    outside the data area, and once it is out there the axis has no reason to
    be longer than what it draws.

    The other half is `plot_v4_runs.py`'s `_view_guard`: an axis whose limits
    came from somewhere other than its own marks can clip one, and a clipped
    mark is invisible rather than obviously wrong.
    """
    lo, hi = sorted(ax.get_xlim() if axis == "x" else ax.get_ylim())
    setattr(ax, "_range_guarded", True)
    marks = _marks(ax)
    if not marks:
        return
    lows = [m[1 if axis == "x" else 3] for m in marks]
    highs = [m[2 if axis == "x" else 4] for m in marks]
    outside = [m[0] for m in marks
               if (m[1 if axis == "x" else 3] < lo - 1e-9
                   or m[2 if axis == "x" else 4] > hi + 1e-9)]
    bad = []
    if outside:
        bad.append(f"{len(outside)} of {len(marks)} marks fall outside the "
                   f"{axis} limits [{lo:.4g}, {hi:.4g}]: {sorted(set(outside))}")
    span = hi - lo
    for edge, blank in (("top", hi - max(highs)), ("bottom", min(lows) - lo)):
        # The bottom of a rate axis is a meaningful zero and stays where it is;
        # only a blank END is text-space in disguise.
        if edge == "top" and span and blank / span > slack:
            bad.append(f"{blank / span * 100:.1f} % of the {axis} range is blank "
                       f"past the longest mark ({max(highs):.4g} of {hi:.4g}); "
                       f"the allowance is {slack * 100:.0f} %")
    if bad:
        raise SystemExit(f"  {name}: the {axis} axis is not honest\n    "
                         + "\n    ".join(bad))


def _contrast_guard(fig, name: str, floor: float = 3.0) -> None:
    """WCAG 2.2 1.4.11: an informative mark needs 3:1 against what is behind it.

    Measured, never assumed, and the measurement is the point. The pooled bars
    were drawn `alpha=0.55`: the fill is named `#D55E00` in the source and puts
    2.07:1 on the page, and the alpha also faded the dark edge that was added
    to carry this very criterion, from 12.63:1 to 3.24:1.

    3.24 still clears 3:1, so that figure was not in breach - which is exactly
    why this has to be measured rather than argued. The fix moves the alpha to
    the face so the edge is solid again, and what it buys is margin: a quarter
    of a point of headroom on the one channel that was carrying the criterion,
    against ten. A fill OR its edge may carry the 3:1 - that is the remedy the
    criterion's own understanding document describes, and it is why every
    filled shape in this repository has a dark border.
    """
    paper = fig.get_facecolor()
    # The paper itself is a Rectangle, and so is every axes background and
    # every legend frame. They are what a mark is measured AGAINST, so
    # measuring them against themselves reports 1.00:1 on every figure -
    # which is what the first run of this guard did.
    ground = {id(fig.patch)}
    for ax in fig.axes:
        ground.add(id(ax.patch))
        if ax.get_legend() is not None:
            ground.add(id(ax.get_legend().get_frame()))
    bad = []
    for p in fig.findobj(mpatches.Rectangle):
        if not p.get_visible() or id(p) in ground:
            continue
        if (p.get_gid() or "") == "on-fill":   # sits on a cell; checked there
            continue
        b = p.get_bbox()
        if not (b.width and b.height):
            continue
        fc, ec = p.get_facecolor(), p.get_edgecolor()
        best, how = 0.0, []
        for what, c, on in (("fill", fc, p.get_fill()),
                            ("edge", ec, p.get_linewidth() > 0)):
            if not on or c[3] == 0:
                continue
            r = figstyle.contrast_ratio(figstyle.over(c, c[3], paper), paper)
            how.append(f"{what} {r:.2f}")
            best = max(best, r)
        if how and best < floor:
            bad.append(f"a filled mark reaches only {best:.2f}:1 against the "
                       f"paper ({', '.join(how)}); 1.4.11 wants {floor}:1")
    # Lines too, and for the same reason a stem in `plot_head_to_head` had its
    # alpha removed: a whisker carries the spread, so it is a graphical object
    # a reader needs. #2f2f2f is 13.53:1 solid and 2.94:1 at alpha 0.5, and it
    # was drawn here at 0.8.
    for ln in fig.findobj(mlines.Line2D):
        if not ln.get_visible() or (ln.get_gid() or "") not in ("mark", "rule"):
            continue
        if ln.get_linewidth() <= 0 or ln.get_linestyle() in ("None", " ", ""):
            continue
        c = mcolors.to_rgba(ln.get_color(), ln.get_alpha())
        ratio = figstyle.contrast_ratio(figstyle.over(c, c[3], paper), paper)
        if ratio < floor:
            bad.append(f"a {ln.get_gid()} line reaches only {ratio:.2f}:1 against "
                       f"the paper; 1.4.11 wants {floor}:1")
    if bad:
        raise SystemExit(f"  {name}: a mark a reader needs is not visible enough\n    "
                         + "\n    ".join(sorted(set(bad))))


def _cover_guard(fig, name: str) -> None:
    """Refuse a legend that is drawn on top of a mark.

    `plot_v4_runs.py` grew this one after giving a legend a white background
    put it over the two lowest points in a panel - the finding that panel
    exists to show. A mark can be inside the view limits and still invisible.
    Every legend in this file hangs below its axes, which is a choice that has
    to keep holding: `bbox_to_anchor` is in axes fractions, so a figure that
    grows a row moves the legend relative to the bars without anyone touching
    the number.
    """
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    bad = []
    for ax in fig.axes:
        leg = ax.get_legend()
        if leg is None or not leg.get_visible():
            continue
        lb = leg.get_window_extent(r)
        boxes = [(f"patch {p.get_gid() or ''}", p.get_window_extent(r))
                 for p in ax.patches if (p.get_gid() or "") != "on-fill"]
        for ln in ax.lines:
            if (ln.get_gid() or "") == "mark":
                boxes.append(("whisker", ln.get_window_extent(r)))
        hit = [n for n, b in boxes if b.width and b.height and lb.overlaps(b)]
        if hit:
            bad.append(f"the legend covers {len(hit)} marks: {sorted(set(hit))}")
    if bad:
        raise SystemExit(f"  {name}: the key is drawn over the data\n    "
                         + "\n    ".join(bad))


def _publish(fig, name: str) -> None:
    """Guard, then write. Nothing is written that has not passed the guards.

    `_range_guard` needs to be told WHICH axis carries the values, so it is the
    one guard a plot function has to call for itself - and a guard somebody has
    to remember is a guard that covers whatever was written the day the list
    was made. This ratchets it: an axes that drew a mark and was never
    range-guarded stops the figure here.
    """
    _text_guard(fig, name)
    _contrast_guard(fig, name)
    _cover_guard(fig, name)
    missed = [ax for ax in fig.axes
              if ax.get_label() != "<colorbar>"
              and _marks(ax) and not getattr(ax, "_range_guarded", False)]
    if missed:
        raise SystemExit(f"  {name}: {len(missed)} axes drew marks and were never "
                         "passed to _range_guard, so nothing checked that the "
                         "axis shows them or that it is not padded out for text")
    if not CHECK:   # --check verifies; it must not dirty the tree
        plt.savefig(OUT_DIR / name, dpi=figstyle.DPI, bbox_inches="tight")
    plt.close(fig)
    if not CHECK:
        print(f"  wrote {(OUT_DIR / name).relative_to(ROOT)}")


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

    def dm(c):
        return 100 * (agg[c]["request_mean_tok_s"] / base["request_mean_tok_s"] - 1)

    def dp(c):
        return 100 * (agg[c]["pooled_tok_s"] / base["pooled_tok_s"] - 1)

    record("mean_by_config", reference="baseline",
           rows=[{"config": c,
                  "request_mean_tok_s": agg[c]["request_mean_tok_s"],
                  "request_mean_delta_pct": dm(c),
                  "pooled_tok_s": agg[c]["pooled_tok_s"],
                  "pooled_delta_pct": dp(c),
                  "min_tok_s": agg[c]["min_tok_s"],
                  "max_tok_s": agg[c]["max_tok_s"],
                  "requests_with_draft_rounds": agg[c]["requests_with_draft_rounds"],
                  "requests": agg[c]["requests"]} for c in cfgs])

    # ---- the layout ------------------------------------------------------
    # A table of rows with the two bars drawn in one column of it, which is the
    # shape `plot_head_to_head` was rebuilt into for the same reason. Every row
    # here carried TWO run-on labels of two quantities each, both anchored to
    # `max_tok_s + 2.0` - the top of the min-max whisker, which is neither
    # bar's value - so "pooled 109.9 (-19.0 %)" was printed at x = 137 tok/s,
    # 27 tok/s to the right of the number it reports, and the labels' left
    # edges staggered across 3.9 tok/s because that anchor is data. Then
    # `xlim(0, max + 44)` reserved 24 % of the axis to hold the text.
    #
    # Now each quantity is a column at a fixed FIGURE fraction, the axis is as
    # long as the longest thing drawn on it, and the dashed baseline rule -
    # which used to pass 2.6 px from the first glyph of four labels - has the
    # axes to itself.
    X_ARM = 0.008
    AX_L, AX_R = 0.145, 0.575
    X_MEAN, X_DMEAN, X_POOL, X_DPOOL, X_DRAFT = 0.655, 0.730, 0.815, 0.890, 0.998

    y = np.arange(len(cfgs))
    h = 0.38
    fig, ax = plt.subplots(figsize=(13.8, max(5.0, 0.62 * len(cfgs))))
    fig.subplots_adjust(left=AX_L, right=AX_R, top=0.885, bottom=0.135)
    # x in FIGURE fractions, y in DATA: one x per column, shared by the header
    # and every cell, which is what makes a column a column.
    tr = mtransforms.blended_transform_factory(fig.transFigure, ax.transData)

    def cell(x, yy, s, ha="right", gid=None, **kw):
        t = ax.text(x, yy, s, transform=tr, ha=ha, va="center", clip_on=False,
                    size=10.4, **kw)
        t.set_gid(gid)
        return t

    for i, c in enumerate(cfgs):
        a = agg[c]
        colour = (C_REF if c == "baseline"
                  else C_ACTIVE if a["requests_with_draft_rounds"] else C_INACTIVE)
        ax.barh(y[i] + h / 2, a["request_mean_tok_s"], height=h,
                edgecolor=figstyle.EDGE, linewidth=figstyle.EDGE_LW, color=colour)
        # The alpha goes on the FACE, not on the artist. `alpha=0.55` on the
        # patch fades the border along with the fill, and the border is what
        # carries 1.4.11 here: measured, it took the edge from 12.63:1 to
        # 3.24:1 and the fill itself to 2.07:1. 3.24 clears the 3:1 the
        # criterion asks, so this was a quarter of a point of headroom rather
        # than a breach; face-only alpha gives it back the whole ten. The hatch
        # is drawn in the edge colour and is solid with it.
        ax.barh(y[i] - h / 2, a["pooled_tok_s"], height=h,
                edgecolor=figstyle.EDGE, linewidth=figstyle.EDGE_LW,
                facecolor=(*figstyle.rgb(colour), 0.55), hatch="///")
        ax.plot([a["min_tok_s"], a["max_tok_s"]], [y[i] + h / 2] * 2,
                color="#2f2f2f", lw=0.9, solid_capstyle="butt", gid="mark")
        for x in (a["min_tok_s"], a["max_tok_s"]):
            ax.plot([x, x], [y[i] + h / 2 - 0.09, y[i] + h / 2 + 0.09],
                    color="#2f2f2f", lw=0.9, gid="mark")

        cell(X_ARM, y[i], c, ha="left", family="DejaVu Sans Mono", color="#1f1f24")
        cell(X_MEAN, y[i], figstyle.fig_num(a["request_mean_tok_s"], 5, 1),
             gid="num:mean", color="#1f1f24")
        cell(X_DMEAN, y[i], figstyle.fig_num(dm(c), 5, 1, True),
             gid="num:mean-delta", color="#1f1f24")
        cell(X_POOL, y[i], figstyle.fig_num(a["pooled_tok_s"], 5, 1),
             gid="num:pooled", color="#4a4a52")
        cell(X_DPOOL, y[i], figstyle.fig_num(dp(c), 5, 1, True),
             gid="num:pooled-delta", color="#4a4a52")
        # The count of requests that recorded a draft round used to be the
        # second line of the y tick AND two of the legend's four entries, in
        # the same words. Worse, the baseline's own line read "no counted draft
        # round" while its bar is blue and the legend's swatch for that string
        # is grey, so the row and the key disagreed about the row's colour. It
        # is a number, so it is a column, and the legend now describes only
        # what the colour means.
        cell(X_DRAFT, y[i],
             f"{figstyle.fig_num(a['requests_with_draft_rounds'], 2, 0)}/{a['requests']}",
             gid="num:draft-rounds", color="#3a3a42")

    hy = len(cfgs) - 0.30
    hdr = dict(size=10.0, color="#3a3a42", weight="bold", linespacing=1.25)
    for x, s, ha in ((X_ARM, "config", "left"),
                     (X_MEAN, "request-mean\ntok/s", "right"),
                     (X_DMEAN, "vs base\n%", "right"),
                     (X_POOL, "pooled\ntok/s", "right"),
                     (X_DPOOL, "vs base\n%", "right"),
                     (X_DRAFT, "req. with a\ndraft round", "right")):
        ax.text(x, hy, s, transform=tr, ha=ha, va="bottom", clip_on=False, **hdr)

    ax.axvline(base["request_mean_tok_s"], color=figstyle.BLUE, ls="--", lw=1.1,
               gid="rule",
               label=f"no-speculation baseline, {base['request_mean_tok_s']:.1f} tok/s")
    ax.set_yticks([])
    ax.set_ylim(-0.8, len(cfgs) - 0.2)
    # As long as the longest thing drawn on it and no longer: the text is
    # outside the axes now, so the axis has nothing to make room for.
    ax.set_xlim(0, max(agg[c]["max_tok_s"] for c in cfgs) * 1.02)
    ax.set_xlabel("decode rate (tokens / second)  -  higher is faster")
    ax.set_title("v1 300-token matrix: request-mean (solid) vs pooled throughput (hatched)\n"
                 "Qwen3.6-35B-A3B UD-Q4_K_XL, one RTX 3090, single request, greedy",
                 pad=46)

    swatch = dict(edgecolor=figstyle.EDGE, linewidth=figstyle.EDGE_LW)
    # Interleaved, because a legend with `ncol=3` fills by COLUMN: this order
    # puts the three colours across the top row and what the two bars in a row
    # are across the second, rather than mixing the two questions down a column.
    handles = [
        mpatches.Patch(facecolor=C_REF, label="no-speculation reference", **swatch),
        mpatches.Patch(facecolor="#c8c8c8", label="solid: request-mean", **swatch),
        mpatches.Patch(facecolor=C_INACTIVE,
                       label="speculative arm, no counted draft round", **swatch),
        mpatches.Patch(facecolor="#c8c8c8", hatch="///", label="hatched: pooled",
                       **swatch),
        mpatches.Patch(facecolor=C_ACTIVE,
                       label="speculative arm, counted draft rounds", **swatch),
        mlines.Line2D([0], [0], color="#2f2f2f", lw=0.9,
                      label="min-max across the ten prompts (1 run each)"),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.075),
              ncol=3, frameon=False)
    ax.grid(axis="x", color="#dcdcdc", lw=0.6)
    ax.set_axisbelow(True)
    _range_guard(ax, "plot_mean_by_config", "x")
    _footer(fig, SPREAD_NOTE)
    _publish(fig, "plot_mean_by_config.png")


def parity_half_width(values, centre: float = 100.0,
                      keep: float = 0.75, sharp: float = 4.0) -> float:
    """Where a scale centred on `centre` should stop resolving and start clipping.

    159 of the 190 cells (83.7 %) sit between 95 and 105, which was the top
    15 % of a vmin=40, vmax=105 scale: 84 % of the matrix rendered as one
    indistinguishable blue and the whole ngmod contrast - 89 against 98 against
    100 - was invisible. The values are bimodal, so the honest place to break
    the scale is between the two modes, and that is a measurement rather than a
    taste: sort |v - centre|, take the largest gap that still leaves `keep` of
    the cells inside, and stop halfway across it. Here that gap is 16.6 wide
    (11.5 to 28.0), more than twice the next largest, and it keeps 170 of 190.

    Two conditions, and the second was added because the first is not enough.
    `keep` stops a unimodal set from picking a gap near zero and clipping
    almost everything, which would be this figure's own defect with the sign
    flipped - but with `keep` alone an evenly spread set still clipped its top
    quarter, because SOME gap is always the largest. So the winning gap must
    also be `sharp` times the mean gap, which is what "the data is bimodal"
    means quantitatively. Here the mean gap is 0.30 and the winning one is
    16.55, a factor of 55; on an evenly spread set the factor is 1 and the
    scale keeps the whole range and clips nothing.
    """
    d = np.sort(np.abs(np.asarray(values, dtype=float) - centre))
    d = d[~np.isnan(d)]
    if d.size < 2:
        return float(max(d.max(), 1.0)) if d.size else 1.0
    gaps = np.diff(d)
    ok = [k for k in range(len(gaps)) if (k + 1) / d.size >= keep]
    if not ok:
        return float(d.max())
    k = max(ok, key=lambda i: gaps[i])
    mean_gap = (d[-1] - d[0]) / (d.size - 1)
    if not mean_gap or gaps[k] < sharp * mean_gap:
        return float(d.max())
    return float(min((d[k] + d[k + 1]) / 2, d.max()))


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
    cell = {(r["config"], r["prompt"]): r for r in rows}

    mats, panels = [], []
    for cap, title in groups:
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
        mats.append(mat)
        panels.append((cap, title, cfgs, prompts))

    # ---- the scale -------------------------------------------------------
    # The diverging map is what `figstyle` defends and none of that changes:
    # `RdBu` and not `RdBu_r`, so the alarming colour stays on the alarming
    # end, and ColorBrewer's red-blue rather than RdYlGn because about eight
    # per cent of men cannot separate red from green.
    #
    # What changes is the two numbers around it. `vmin=40, vmax=105` put the
    # map's white at (40 + 105) / 2 = 72.5 % of the baseline, so a scale whose
    # whole justification is "diverging, centred on the same as the baseline"
    # was centred on 27 % slower than the baseline, and parity - the one value
    # the reader is asked to compare against - rendered as deep blue. The scale
    # is now symmetric about 100 by construction, which is what makes the
    # midpoint honest, and its half-width is measured from the data.
    allv = np.concatenate([m[~np.isnan(m)].ravel() for m in mats]) if mats else np.array([])
    half = parity_half_width(allv)
    vmin, vmax = 100.0 - half, 100.0 + half
    below = int((allv < vmin).sum())
    above = int((allv > vmax).sum())
    extend = ("both" if below and above else "min" if below
              else "max" if above else "neither")
    record("per_prompt", parity_half_width=half, vmin=vmin, vmax=vmax,
           clipped_below=below, clipped_above=above,
           panels=[{"cap": cap, "configs": cfgs, "prompts": prompts,
                    "matrix": [[None if np.isnan(v) else float(v) for v in row]
                               for row in mat]}
                   for (cap, _t, cfgs, prompts), mat in zip(panels, mats)])

    # `hspace` is a fraction of the MEAN panel height, and the two panels here
    # are 15 rows and 4, so 0.62 of their mean left a 600 px band of empty
    # paper between them. It has to clear the upper panel's rotated tick
    # labels, which are 210 px tall at 45 degrees, and no more. 0.30 puts
    # them on the lower panel's title, which the guard reports as three
    # collisions; 0.40 is the first value that clears with room to spare.
    fig, axes = plt.subplots(
        len(groups), 1, figsize=(12.6, 0.42 * sum(heights) + 3.0),
        gridspec_kw={"height_ratios": heights, "hspace": 0.40},
        constrained_layout=False, squeeze=False)
    axes = [a[0] for a in axes]

    im = None
    ink_bad: list[str] = []
    for ax, (cap, title, cfgs, prompts), mat in zip(axes, panels, mats):
        im = ax.imshow(mat, cmap=figstyle.DIVERGING, vmin=vmin, vmax=vmax,
                       aspect="auto")
        ax.set_xticks(range(len(prompts)))
        # 45 degrees and anchored, not 30: at 30 the rotated boxes of
        # medium_chat/medium_rec, long_explain/code_small and
        # multi_turn_1/multi_turn_2 overlapped by 9 to 13 px at 300 DPI.
        ax.set_xticklabels([display_tag(t) for t in prompts],
                           rotation=45, ha="right", rotation_mode="anchor")
        ax.set_yticks(range(len(cfgs)))
        ax.set_yticklabels(cfgs)
        ax.set_title(title, pad=6)

        for i, c in enumerate(cfgs):
            for j, p in enumerate(prompts):
                v = mat[i, j]
                if np.isnan(v):
                    continue
                fill = im.cmap(im.norm(v))
                # the ink is chosen from what the cell actually RENDERS as,
                # not from the value: this read `"black" if v > 62`, and RdBu
                # puts the high end in deep blue, so every 100 and the single
                # 104 -- the darkest cells on the figure -- were printed in
                # black. The scale is exempt from 1.4.11 as essential; the
                # number over it is text and 1.4.3 applies to it.
                ink = figstyle.on_fill(fill)
                if figstyle.contrast_ratio(ink, fill) < 4.5:
                    ink_bad.append(f"the number in the {c}/{p} cell reaches only "
                                   f"{figstyle.contrast_ratio(ink, fill):.2f}:1")
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", color=ink)
                if (cell.get((c, p)) or {}).get("draft_n", 0) > 0:
                    # Not "#101010" any more. The outline is a graphical object
                    # a reader needs, so 1.4.11 asks 3:1 of it against the cell
                    # it is drawn on, and black measured 2.06:1 on the deepest
                    # red and 2.63:1 on the deepest blue -- which under the old
                    # limits was the colour of 100. Chosen per cell like the
                    # ink, it is at least 4.58:1 either side of the crossover.
                    edge = figstyle.on_fill(fill)
                    if figstyle.contrast_ratio(edge, fill) < 3.0:
                        ink_bad.append(f"the draft-round outline on {c}/{p} reaches "
                                       f"only {figstyle.contrast_ratio(edge, fill):.2f}:1")
                    ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                           edgecolor=edge, lw=1.7, gid="on-fill"))
    if ink_bad:
        raise SystemExit("  plot_per_prompt: a mark is not legible on its cell\n    "
                         + "\n    ".join(sorted(set(ink_bad))))

    # `aspect` is what sets the bar's LENGTH once `fraction` has set its width,
    # and at the default 20 a bar this thin came out 55 % of the panels' height,
    # floating beside the upper heatmap instead of describing both of them.
    cb = fig.colorbar(im, ax=axes, label="% of the matched no-speculation baseline",
                      fraction=0.026, pad=0.015, aspect=34, extend=extend)
    cb.ax.axhline(100.0, color="#1f1f24", lw=1.0, gid="rule")
    fig.suptitle("v1 per-prompt decode rate, normalised to the matched baseline\n"
                 "outlined cell = this request recorded at least one fully accepted "
                 "draft round", y=1.0)
    _footer(fig, f" The colour scale is symmetric about parity, 100 = the matched "
                 f"baseline, and resolves +/- {half:.1f} points of it; the "
                 f"{below + above} cells outside that range are drawn in the end "
                 f"colour and marked by the bar's arrow. Every cell prints its own "
                 f"value, so nothing is carried by colour alone.")
    _publish(fig, "plot_per_prompt.png")


def plot_acceptance_accounting(agg: dict[str, dict]) -> None:
    """Replaces the retracted plot_accept_vs_speed.png (see ERRATA.md item A1)."""
    src = OUT_DIR / "verbose_accounting.json"
    if not src.exists():
        print("  no analysis/verbose_accounting.json - run analysis/verbose_accounting.py first")
        return
    rep = json.loads(src.read_text(encoding="utf-8"))[0]
    d, r, s = rep["drafter_own_counters"], rep["reported"], rep["state_management"]
    full = rep["attempts_fully_accepted"]
    part = rep["attempts_partially_accepted"]
    total = rep["verification_attempts"]

    # ---- what the caption may say ----------------------------------------
    # The caption read "Cost of the 20 discarded rounds: 33 state checkpoints",
    # and 33 is `checkpoints_created`, which is not the cost of the discarded
    # rounds at all: `checkpoints_restored` is 20. The figure's own green bar
    # is also 33 - `attempts_fully_accepted` - so one number labelled two
    # quantities in one figure and the caption attached it to the third.
    #
    # The log settles which is which. Reading
    # v2_3090_followup/v2_oleg_suggestions/verbose.log as an event stream of
    # "created speculative checkpoint", "restoring speculative checkpoint" and
    # `update_slots: n_draft=N, accepted=M`, there are 33 creations, 33 full
    # accepts, 20 partials and 20 restores; every creation opens a segment
    # containing exactly one full accept, every partial is immediately followed
    # by a restore, and nothing is unmatched. So one checkpoint is created per
    # verification ROUND and one is restored per PARTIAL acceptance, which is
    # why 33 = the fully accepted rounds and 20 = the discarded ones, and why
    # the number the caption wanted was 20.
    #
    # These identities are the caption's whole content, so they are asserted
    # rather than trusted: a re-run of verbose_accounting.py that broke one
    # would fail here instead of publishing a caption that had gone false.
    if total != full + part:
        raise SystemExit(f"  plot_acceptance_accounting: {total} verification attempts "
                         f"is not {full} full + {part} partial")
    if s["checkpoints_restored"] != part:
        raise SystemExit(f"  plot_acceptance_accounting: {s['checkpoints_restored']} "
                         f"restores against {part} partial acceptances; the caption "
                         "says a restore is what a partial acceptance costs")
    if s["checkpoints_created"] != full:
        raise SystemExit(f"  plot_acceptance_accounting: {s['checkpoints_created']} "
                         f"checkpoints created against {full} rounds; the caption says "
                         "one checkpoint is created per round")

    # Short lines, and "(same log, next line)" said once in the title rather
    # than under two of the three bars. At 11 pt "drafter sequence counter" is
    # 605 px wide against a 500 px tick pitch, so the three category labels ran
    # into each other -- which is the collision the canvas was widened for once
    # already, come back because the labels grew instead of the panel.
    bars = [
        ("server\ncounter\n(v1 and v2\npublished this)", r["accepted"], r["generated"]),
        ("drafter\ntoken counter", d["draft_tokens_accepted"],
         d["draft_tokens_generated"]),
        ("drafter\nsequence counter", d["drafts_accepted"],
         d["drafts_generated"]),
    ]
    record("acceptance_accounting",
           counters=[{"name": n.replace("\n", " "), "accepted": a, "proposed": b,
                      "pct": 100 * a / b} for n, a, b in bars],
           attempts=total, fully_accepted=full, partially_accepted=part,
           checkpoints_created=s["checkpoints_created"],
           checkpoints_restored=s["checkpoints_restored"],
           gib_written=s["checkpoint_gib_written"],
           gib_read_back=s["checkpoint_gib_read_back"])

    # 12.2 x 4.7 was too small once the type came up to the standard's floor:
    # the three two-line category labels ran into each other and both legends
    # landed on top of the labels below them. Angling the labels made it worse,
    # because rotated two-line text is taller still and the panel had nowhere to
    # go. The canvas is the thing that was wrong, so the canvas is what changed.
    # `tight_layout` cannot be used here any more and the geometry is set
    # directly instead. It measures the legend that hangs 0.46 axes below ax1
    # and the title pad that holds the label rows, and pays for both out of the
    # axes: the panel came out 466 px tall on a 1920 px canvas, which squeezed
    # the two label rows into 33 px of space they need 100 for.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.0, 7.0))
    fig.subplots_adjust(left=0.075, right=0.945, top=0.70, bottom=0.30,
                        wspace=0.34)

    x = np.arange(len(bars))
    # #d9d9d9 is 1.3:1 against white and its #9a9a9a border 2.8:1, so neither
    # the bar nor its edge met the 3:1 that 1.4.11 asks of a graphic a reader
    # needs. The palette grey is 4.54:1 and the shared edge 12.63:1.
    ax1.bar(x, [b[2] for b in bars], color="#d9d9d9",
            edgecolor=figstyle.EDGE, linewidth=figstyle.EDGE_LW,
            width=0.56, label="proposed / denominator")
    # ONE colour, because the legend claims one. This drew each bar's accepted
    # portion in its own hue while the single legend swatch showed the first of
    # them, so a reader was told "accepted = vermilion" beside a blue accepted
    # bar and a green one. The three counters are already named on the x axis;
    # the colour was carrying nothing the labels did not, and was contradicting
    # the key.
    ax1.bar(x, [b[1] for b in bars], color=figstyle.VERMILION,
            edgecolor=figstyle.EDGE, linewidth=figstyle.EDGE_LW,
            width=0.56, label="accepted / numerator")
    # Two rows above the axes, x in DATA (which bar this is about) and y in
    # AXES fractions (the same height for all three). "115/115 = 100.0 %",
    # "115/214 = 53.7 %" and "33/81 = 40.7 %" were one run-on string each,
    # three quantities packed together, and each was anchored to its own bar
    # top - so they were drawn at 119, 218 and 85 draft units, a zig-zag in the
    # data coordinate space that could not be read across.
    rowtr = mtransforms.blended_transform_factory(ax1.transData, ax1.transAxes)
    for i, (_name, num, den) in enumerate(bars):
        t = ax1.text(i, 1.135, f"{figstyle.fig_num(num, 3, 0)} / "
                               f"{figstyle.fig_num(den, 3, 0)}",
                     transform=rowtr, ha="center", va="bottom", clip_on=False,
                     color="#3a3a42", size=10.6)
        t.set_gid("row:accepted-of-proposed")
        t = ax1.text(i, 1.018, f"{figstyle.fig_num(100 * num / den, 5, 1)} %",
                     transform=rowtr, ha="center", va="bottom", clip_on=False,
                     fontweight="bold", size=12)
        t.set_gid("row:acceptance-pct")
    ax1.set_xticks(x)
    ax1.set_xticklabels([b[0] for b in bars])
    # A little more room at both ends than `bar` leaves by default: the second
    # line of the first category label is 22 characters wide and centred on
    # x = 0, which put its left edge within five pixels of the "0" on the y
    # axis.
    ax1.set_xlim(-0.62, len(bars) - 1 + 0.62)
    ax1.set_ylabel("draft units")
    ax1.set_ylim(0, max(b[2] for b in bars) * 1.02)
    ax1.set_title("Three acceptance numbers from one run\n"
                  "consecutive lines of one v2 verbose log\n"
                  "--draft-min 2 --draft-max 32, prompt 1", pad=72)
    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.30),
               ncol=2, frameon=False)
    ax1.grid(axis="y", color="#e4e4e4", lw=0.6)
    ax1.set_axisbelow(True)

    ax2.barh([1], [full], color=figstyle.GREEN, height=0.62,
             edgecolor=figstyle.EDGE, linewidth=figstyle.EDGE_LW)
    ax2.barh([0], [part], color=figstyle.VERMILION, height=0.62,
             edgecolor=figstyle.EDGE, linewidth=figstyle.EDGE_LW)
    ax2.set_yticks([0, 1])
    # The legend here said "fully accepted -> reaches the counter (33)" and
    # "partly accepted -> `continue`, discarded, re-verified (20)" beside ticks
    # that already read "full accept" and "partial accept": two of its words
    # were the tick's words, and the only content it added was the two values,
    # which were printed nowhere on the bars. A reader had to travel to the key
    # and back to read a bar. The fate of each class belongs on its tick and
    # the value belongs in a column.
    ax2.set_yticklabels(["partial accept\n`continue`, discarded,\nre-verified",
                         "full accept\nreaches the counter"])
    ax2.set_xlabel(f"verification rounds  ({total} attempts in this run)")
    ax2.set_xlim(0, max(full, part) * 1.02)
    ax2.set_ylim(-0.58, 1.58)
    ax2.set_title("Why the server ratio can only be 1.0\n"
                  "target reports COMMON_CONTEXT_SEQ_RM_TYPE_FULL", pad=16)
    ax2.grid(axis="x", color="#e4e4e4", lw=0.6)
    ax2.set_axisbelow(True)
    coltr = mtransforms.blended_transform_factory(ax2.transAxes, ax2.transData)
    for yy, v in ((1, full), (0, part)):
        t = ax2.text(1.02, yy, figstyle.fig_num(v, 2, 0), transform=coltr,
                     ha="left", va="center", clip_on=False, size=11,
                     color="#1f1f24")
        t.set_gid("num:rounds")
    ax2.text(1.02, 1.50, "rounds", transform=coltr, ha="left", va="bottom",
             clip_on=False, size=10.0, color="#3a3a42", weight="bold")

    fig.suptitle("The published \"100 % draft acceptance\" is a counter artefact, not a measurement", y=0.99)
    # No bottom band is reserved any more. Reserving 13.5 % of the height and
    # then writing the caption at a fixed y=0.058 left the legends, which hang
    # below the axes, ending a third of the canvas above it: the figure was
    # published with an empty band down its middle. `figstyle.footer` measures
    # where the lowest artist actually is, which is the whole reason it exists.
    _range_guard(ax1, "plot_acceptance_accounting left", "y")
    _range_guard(ax2, "plot_acceptance_accounting right", "x")
    figstyle.footer(
        fig,
        f"{total} verification attempts in {s['checkpoints_created']} rounds. One "
        f"checkpoint is created per round ({s['checkpoints_created']} created @ "
        f"{s['checkpoint_mib_each']} MiB = {s['checkpoint_gib_written']} GiB written) "
        f"and one is restored per partial acceptance, so the cost of the {part} "
        f"discarded rounds is {s['checkpoints_restored']} restores, "
        f"{s['checkpoint_gib_read_back']} GiB read back. Drafter generate() alone "
        f"= {rep['drafter_share_of_generation_pct']} % of the "
        f"{rep['generation_wall_ms']} ms generation wall-clock. Source: "
        f"v2_3090_followup/v2_oleg_suggestions/verbose.log, build "
        f"b8863-97895129e. Reconstruct with analysis/verbose_accounting.py. "
        f"Mechanism: ERRATA.md item A1.")
    _publish(fig, "plot_acceptance_accounting.png")


# ---------------------------------------------------------------- main ----

def main() -> None:
    rows = load_all()
    agg = aggregate(rows)
    print(f"loaded {len(rows)} requests from {len(agg)} run labels")
    active = [c for c in agg if agg[c]["counted_draft_tokens"] > 0]
    print(f"  {len(active)} labels recorded at least one fully accepted draft round")
    print(f"  {len(agg) - len(active)} labels recorded none "
          "(baseline, control, or no surviving round)")
    if not CHECK:
        write_csvs(rows, agg)
    plot_mean_by_config(agg)
    plot_per_prompt(rows, agg)
    plot_acceptance_accounting(agg)

    if CHECK:
        if not SERIES_JSON.exists():
            sys.exit(f"{SERIES_JSON} is missing; run this script without --check first")
        want = json.loads(SERIES_JSON.read_text(encoding="utf-8"))
        if want != SERIES:
            bad = sorted({k for k in set(want) | set(SERIES)
                          if want.get(k) != SERIES.get(k)})
            sys.exit(f"the committed v1 charts are stale: {bad} no longer match the "
                     f"data. Re-run `python analysis/plot.py`.")
        print(f"  charts match the data ({len(SERIES)} series)")
        return
    SERIES_JSON.write_text(json.dumps(SERIES, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print(f"  wrote {SERIES_JSON.relative_to(ROOT)}")

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
