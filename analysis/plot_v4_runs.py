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
import os
import sys
import statistics as st
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.collections as mcoll
import matplotlib.text as mtext
import matplotlib.transforms as mtransforms

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "v4_audit_2026_08_25" / "data"
OUT = ROOT / "analysis"

# Okabe and Ito, via analysis/figstyle.py. The pair this replaced was a red
# and a green telling two arm families apart with nothing else distinguishing
# them, which about eight per cent of men cannot separate.
# `analysis/` is not on the path when these run from the repository root
import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))
import figstyle                                                    # noqa: E402
figstyle.apply(plt)
C_REF = figstyle.BLUE          # no speculation
C_ACTIVE = figstyle.VERMILION  # matched-vocabulary drafter
C_DFLASH = figstyle.GREEN      # DFlash
C_DFLASH2 = figstyle.SKY       # DFlash, second configuration
C_INACTIVE = figstyle.GREY     # drafter-free n-gram methods

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


def _footer(fig, extra="", base=None, width=150):
    """Reserved space, wrapped text, at the standard's minimum size.

    This drew at y=0.004 with `wrap=True`, and the figure was saved with
    `bbox_inches="tight"`, so the caption was written on top of the x axis label
    and then clipped at both edges. Two of the five figures shipped that way.
    """
    # `width` because the wrap column decides the figure's width under
    # `bbox_inches="tight"`, and the width decides the type size where the figure
    # is read. 150 was chosen for a 13 inch canvas; a 9 inch one needs less.
    figstyle.footer(fig, (FOOTER if base is None else base) + extra, width=width)


_FIGSP = "\u2007"   # FIGURE SPACE. Measured: in DejaVu Sans its advance is one
                    # digit wide, and the digits are themselves tabular, so a
                    # padded number sits in the same column as an unpadded one.
                    # An ordinary space is half a digit and silently breaks the
                    # column, which is the whole point of the padding.
_MINUS = "\u2212"  # the axis ticks and the README tables use it, and it is the
                    # same advance as "+", so the sign occupies a column too;
                    # ASCII hyphen-minus is 43 % of that width.


def _fig_num(v: float, width: int, dp: int, signed: bool) -> str:
    """A number that lines up under the one above it."""
    s = f"{v:+.{dp}f}" if signed else f"{v:.{dp}f}"
    return s.replace("-", _MINUS).rjust(width, _FIGSP)


def _table_guard(fig, name: str, gap: float = 5.0) -> None:
    """Refuse to publish a figure whose text collides or whose columns are ragged.

    Two collisions shipped in the first version of the head-to-head table and
    neither was caught by the check written alongside it, because that check
    walked a list of cells maintained by hand and the title was never added to
    it. So this walks every Text the figure actually contains. It also demands a
    GAP rather than mere non-overlap: `draft/gen` and `acceptance` cleared each
    other by a pixel and still read as one word.

    The column half of this check was vacuous in its first form. It compared the
    anchor edge of each cell, taking x0 for a left-aligned one and x1 for a
    right-aligned one; those are the same point by construction, so a cell whose
    alignment had been changed reported flush. A fault injected on purpose -- one
    numeric cell left-aligned in a right-aligned column -- was not caught. What
    actually has to hold is that the numbers in a column are the SAME RENDERED
    WIDTH, which is what the figure-space padding is for: equal width against a
    shared anchor is what puts every digit in its own column. Cells are found by
    the gid `_cell` gives them rather than by position.

    Tested in both directions on each half: the collision half reports four pairs
    when a column is moved onto its neighbour and none otherwise; the width half
    reports the column when one cell's padding is dropped and none otherwise.
    """
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    # Two kinds of Text report a box they never paint, and both produced false
    # reports here before they were excluded. A tick LOCATOR routinely puts one
    # tick past the view limit -- this figure's y ticks run to 300 against a limit
    # of 255.6 -- and its label object still answers with a position above the
    # axes, where it duly "collided" with the title. And every tick carries a
    # second, hidden label object whose extent is a one pixel box at the origin,
    # which overlaps anything else near the origin. A guard that reports either is
    # a guard nobody believes the third time.
    skip = set()
    for a in fig.axes:
        for axis, lim in ((a.xaxis, sorted(a.get_xlim())),
                          (a.yaxis, sorted(a.get_ylim()))):
            for loc, lab in zip(axis.get_ticklocs(), axis.get_ticklabels()):
                if not lim[0] <= loc <= lim[1]:
                    skip.add(id(lab))
    texts = [t for t in fig.findobj(mtext.Text)
             if t.get_text().strip() and t.get_visible() and id(t) not in skip
             and t.get_window_extent(r).width > 1
             and t.get_window_extent(r).height > 1]
    boxes = [(t.get_text().replace("\n", "|")[:44], t.get_window_extent(r))
             for t in texts]
    bad = [f"{a!r} and {b!r} are closer than {gap} px"
           for i, (a, x) in enumerate(boxes) for b, y in boxes[i + 1:]
           if mtransforms.Bbox.from_extents(x.x0 - gap, x.y0 - gap,
                                            x.x1 + gap, x.y1 + gap).overlaps(y)]
    cols: dict = defaultdict(list)
    for t in texts:
        gid = t.get_gid() or ""
        if gid.startswith("num:"):
            cols[gid].append((t.get_text(),
                              round(t.get_window_extent(r).width, 1),
                              t.get_ha()))
    for gid, cs in sorted(cols.items()):
        widths = {w for txt, w, _ in cs if any(ch.isdigit() for ch in txt)}
        aligns = {a for _, _, a in cs}
        if len(widths) != 1:
            bad.append(f"{gid} is ragged: the numbers render at widths "
                       f"{sorted(widths)}, so their digits cannot line up")
        if len(aligns) != 1:
            bad.append(f"{gid} mixes alignments {sorted(aligns)}")
    if bad:
        raise SystemExit(f"  {name}: the figure is not readable\n    "
                         + "\n    ".join(bad))


def _view_guard(ax, name: str, eps: float = 1e-9) -> None:
    """Refuse to publish an axes that does not show a point it drew.

    `plot_acceptance_threshold` set its x limits from the two numbers it also used
    to draw the fit line over, and two of its sixty fitted points sat left of
    them, at 9.5 % and 13.6 % accepted. They were scattered, clipped, and never
    mentioned: the legend went on saying thirty points and thirty points while
    the figure showed fifty-eight, and the two it dropped were the
    lowest-acceptance pair, which carry the most leverage on the slope the whole
    figure is about.

    Only marks are checked. A fit line, a grid line, an axhline and an axvline are
    all drawn to the edge on purpose, so a Line2D with no marker is not data here.
    """
    xlo, xhi = sorted(ax.get_xlim())
    ylo, yhi = sorted(ax.get_ylim())
    # PathCollection only. `errorbar` leaves a LineCollection per series, and a
    # LineCollection's `get_offsets()` returns its default [[0, 0]] rather than
    # anything it drew, so an earlier version of this reported two phantom points
    # at the origin on every figure with error bars. A guard that cries wolf is a
    # guard somebody switches off.
    pts = []
    for c in ax.collections:
        if isinstance(c, mcoll.PathCollection):
            pts += [(q[0], q[1]) for q in c.get_offsets()]
    for ln in ax.lines:
        # `ln.get_transform() is ax.transData` and nothing else: an axhline or an
        # axvline is BLENDED, one axis data and the other the axes fraction 0 to 1,
        # so its xydata compared against a data limit is a comparison of two
        # different things. `_ink_guard` had exactly this bug and it made that
        # guard blind to the fault it was written for.
        if (ln.get_marker() not in ("", " ", "None", None)
                and ln.get_transform() is ax.transData):
            pts += [(x, y) for x, y in ln.get_xydata()]
    out = [(x, y) for x, y in pts
           if x != x or y != y                      # NaN is not "inside"
           or not (xlo - eps <= x <= xhi + eps and ylo - eps <= y <= yhi + eps)]
    if out:
        raise SystemExit(
            f"  {name}: {len(out)} of {len(pts)} plotted points fall outside "
            f"x[{xlo:.4g}, {xhi:.4g}] y[{ylo:.4g}, {yhi:.4g}] and are not drawn:\n    "
            + "\n    ".join(f"({x:.4g}, {y:.4g})" for x, y in out[:8]))


def _cover_guard(ax, name: str) -> None:
    """Refuse to publish an axes whose legend sits on top of a point it drew.

    `_view_guard` asks whether a point is inside the axes. A point can be inside
    the axes and still invisible, because something opaque was put over it. Giving
    `plot_two_levels`'s right panel a white legend background -- to stop its two
    dashed swatches reading as two more dashed marks in the data -- put that
    background over blocks 4 and 5 of the arm the panel is about, at -4.45 and
    -4.66, which are the lowest two points on it and the transition the panel
    exists to show. The legend was moved; this is what would have caught it.
    """
    leg = ax.get_legend()
    if leg is None:
        return
    fig = ax.get_figure()
    fig.canvas.draw()
    box = leg.get_window_extent(fig.canvas.get_renderer())
    pts = []
    for c in ax.collections:
        if isinstance(c, mcoll.PathCollection):
            pts += [(q[0], q[1]) for q in c.get_offsets()]
    for ln in ax.lines:
        if (ln.get_marker() not in ("", " ", "None", None)
                and ln.get_transform() is ax.transData):
            pts += [(x, y) for x, y in ln.get_xydata()]
    hidden = [d for d, sxy in zip(pts, ax.transData.transform(pts).tolist())
              if box.contains(*sxy)] if pts else []
    if hidden:
        raise SystemExit(
            f"  {name}: the legend covers {len(hidden)} of {len(pts)} plotted "
            f"points. Move it.\n    "
            + "\n    ".join(f"({x:.4g}, {y:.4g})" for x, y in hidden[:8]))


def _ink_guard(ax, name: str) -> None:
    """Refuse to publish an axes where a label's background erases a line.

    The three guards above all pass on a figure where an opaque label box is laid
    across the curve the figure is about. `plot_dflash_sweep` had two: the box
    behind "-14.8%" took a segment out of the green line and the box behind
    "+17.3%" took one out of the orange, and at 300 DPI it is visible as a pale
    break in the stroke. No text collides, no point is outside the axes, and no
    legend covers a point, so nothing else here would have said a word.

    A label may sit ON a line. What it may not do is paint over it: this looks for
    a Text whose bbox patch is opaque enough to hide a stroke and whose box a line
    segment passes through.
    """
    fig = ax.get_figure()
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    # Sampled along the segment, not at its ends and middle. The first version of
    # this tested the two endpoints and the midpoint, and passed: the green line
    # runs from n_max 4 straight to n_max 8 as ONE segment, so its endpoints are
    # at the two markers and its midpoint is halfway between them, and the label
    # that erases it sits at neither. A guard that samples three points on a
    # segment a thousand pixels long is a guard that agrees with whatever it is
    # shown.
    pts = []
    for ln in ax.lines:
        if ln.get_linestyle() in ("None", " ", "") or ln.get_linewidth() <= 0:
            continue
        # the LINE's own transform, not ax.transData. `axhline` and `axvline` are
        # blended: one axis is data and the other is the axes fraction 0 to 1, so
        # transforming their xydata as data put the reference rule at x = 0 to 1
        # on an axis spanning 16 to 58, which is off the chart. This guard could
        # not see a collision with a reference line at all, and the fault it was
        # written to catch was a three line annotation with the baseline rule
        # through its middle.
        xy = ln.get_transform().transform(ln.get_xydata())
        for i in range(len(xy) - 1):
            (x0, y0), (x1, y1) = xy[i], xy[i + 1]
            n = max(2, int(max(abs(x1 - x0), abs(y1 - y0)) / 4) + 1)
            pts += [(x0 + (x1 - x0) * k / n, y0 + (y1 - y0) * k / n)
                    for k in range(n + 1)]
    # Both halves. The first form of this checked only text WITH an opaque
    # background, on the theory that the defect is a box painting over a stroke.
    # The other half is a line drawn THROUGH text that has no background, which is
    # not a lost stroke but a struck-through sentence, and it is the harder of the
    # two to read. A three line annotation had the baseline rule through its
    # middle and nothing said so.
    #
    # Inset vertically before testing, because a label anchored ON a line is a
    # convention, not a fault: `no speculation, 133 tok/s` sits at va="bottom" with
    # the rule at its lower edge and must not fire.
    bad = []
    for t in ax.texts:
        if not t.get_text().strip() or not t.get_visible():
            continue
        bb = t.get_bbox_patch()
        opaque = bb is not None and (bb.get_alpha() or 1.0) >= 0.35
        box = (bb if opaque else t).get_window_extent(r)
        if box.height <= 2 or box.width <= 2:
            continue
        # The inset applies to the UNBACKGROUNDED case only. An opaque box erases
        # whatever is under it anywhere inside it, so insetting that one weakened
        # the half this guard was written for: the dflash labels stopped being
        # caught, and the both-ways sweep is what said so. Text with no background
        # is struck through only when a line crosses its middle, and a label
        # anchored ON a rule by convention must not fire.
        test = box if opaque else mtransforms.Bbox.from_extents(
            box.x0, box.y0 + 0.22 * box.height, box.x1, box.y1 - 0.22 * box.height)
        hit = sum(1 for q in pts if test.contains(*q))
        if hit:
            what = ("covers" if opaque else "is struck through by")
            bad.append(f"{t.get_text().splitlines()[0]!r} {what} a line "
                       f"({hit} sampled points)")
    if bad:
        raise SystemExit(f"  {name}: a label and a line are drawn over each other\n    "
                         + "\n    ".join(sorted(set(bad))))


def _guards(fig, name: str) -> None:
    """Every guard, over every axes, for every figure.

    Hung one at a time they were hung unevenly: `plot_head_to_head` had the text
    one and none of the other three, and `plot_dflash_sweep` had the view one on
    its upper panel only. One call, so a figure cannot be given a subset by
    accident, and so a new figure gets all four by writing one line.
    """
    for ax in fig.axes:
        _view_guard(ax, name)
        _cover_guard(ax, name)
        _ink_guard(ax, name)
    _table_guard(fig, name)


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
                    textcoords="offset points", xytext=(0, 9), ha="center", color=figstyle.text_colour(C_REF),
                      bbox=box)
        ax.annotate(f"{spec_y[i]:.0f}", (i, spec_y[i] - spec_e[i]),
                    textcoords="offset points", xytext=(0, -15), ha="center", color=figstyle.text_colour(C_ACTIVE),
                      bbox=box)

    # The ratio row. It used to be annotated at (i, midpoint of the two series),
    # which put it in the data coordinate space of an axis labelled "aggregate
    # throughput", so each box read as a mark at 70, 91 and 104 tokens a second.
    # And because the midpoint between a rising line and a flat one rises, the
    # three boxes ASCENDED left to right while the ratios they report fall, 0.28
    # to 0.18 to 0.16: a reader who took the shape before the text took away the
    # opposite of the finding. They are now a row at one height, in axes
    # coordinates, where they have no vertical reading at all.
    _ratio = [spec_y[i] / base_y[i] for i in range(len(levels))]
    _rowtr = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    ax.text(0.5 * (len(levels) - 1), 0.995,
            "spec-draft-n8 as a fraction of no speculation",
            transform=_rowtr, ha="center", va="top", size=10, color="#3f3f46")
    for i in range(len(levels)):
        ax.text(i, 0.920, f"{_ratio[i]:.2f}x", transform=_rowtr, ha="center",
                va="top", color="#3f3f46",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#d0d0d0",
                          lw=0.7))

    ax.set_xlim(-0.35, len(levels) - 0.65)
    ax.set_xticks(list(x))
    # The tick used to read "N / client requests in flight: w, w, w", which said
    # "client requests in flight" on the tick, again under it and again on the
    # axis. The observed widths are not redundant though: they are what the
    # caption means by "verified from request timestamps", so they stay, and
    # only the repeated words go.
    ax.set_xticklabels([f"{c}\nobserved: {', '.join(str(w) for w in widths[i])}"
                        for i, c in enumerate(levels)])
    ax.set_xlabel("concurrent client requests in flight")
    ax.set_ylabel("aggregate throughput (generated tokens / wall-clock second)")
    # 1.18 left no room above the highest point and the ratio row landed on the
    # "180" label, four pixels into it, with the row's own header touching the
    # values. The c=8 baseline carries an SD of 15.2, so the mark to clear is
    # 195.2, not 180. Chosen by measuring the boxes, not by eye.
    ax.set_ylim(0, max(base_y) * 1.42)
    # The two ratios in the subtitle were typed in as "0.28x" and "0.16x". They
    # happened to be right, and nothing would have said so if a re-measurement had
    # moved them.
    ax.set_title("Aggregate throughput against concurrent client requests\n"
                 f"the no-speculation arm gains, the external-drafter arm does not: "
                 f"{_ratio[0]:.2f}x at {levels[0]}, {_ratio[-1]:.2f}x at {levels[-1]}",
                 pad=22)
    # Below the axes: the only large empty region is where the c=1 labels sit.
    # -0.16 put the legend on the x axis label. The tick labels also spelled
    # "client requests in flight: N" under every tick, so the same fact was
    # written three times: on the tick, under it, and on the axis.
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, frameon=False)
    ax.grid(axis="y", color="#dcdcdc", lw=0.6)
    ax.set_axisbelow(True)
    plt.tight_layout(rect=[0, 0.075, 1, 1])
    _footer(fig, " Error bars are the run-to-run SD of three repeats. The x axis is "
                 "CONCURRENT CLIENT REQUESTS, verified from request timestamps; the "
                 "server's decode batch width was not instrumented.")
    _guards(fig, "batching")
    if not CHECK:  # --check verifies the numbers, it must not dirty the tree
        plt.savefig(OUT / "plot_batching.png", dpi=figstyle.DPI, bbox_inches="tight")
    plt.close()
    if not CHECK:
        print(f"  wrote {(OUT / 'plot_batching.png').relative_to(ROOT)}")


# ------------------------------------------------------------ runs J, K ----
def plot_dflash_sweep() -> None:
    series = []
    for label, pattern, colour in (
            ("run J:  -c 16384, fitter default margin", "matrix_J2_*/*__rep*.json", C_DFLASH),
            # not C_DFLASH2: sky blue measures 2.31:1 against white, and a line
            # has no border to carry the 3:1 that WCAG 2.2's 1.4.11 asks of a
            # graphic a reader needs. Vermilion is 3.87:1 and is green's partner
            # in the Okabe and Ito set.
            ("run K:  -c 8192, --fit-target 2048", "matrix_K1_sweep_*/*__rep*.json",
             figstyle.VERMILION)):
        arms = load(pattern)
        if "baseline" not in arms:
            continue
        b = agg(arms["baseline"])[0]
        # `agg` returns (mean, stdev) and only the mean was read here, so this
        # chart drew a mean of three repeats with no error bar while the chart
        # sixty lines above it drew the same quantity with one, from the same
        # helper. A mean without its spread is the omission the standards for
        # scientific figures name first.
        pts = sorted((int(a.rsplit("n", 1)[1]), 100 * (agg(arms[a])[0] / b - 1),
                      100 * sum(x["draft_n_accepted"] for r in arms[a] for x in r["rows"])
                      / max(1, sum(x["draft_n"] for r in arms[a] for x in r["rows"])),
                      100 * agg(arms[a])[1] / b)
                     for a in arms if a.startswith("spec-dflash-n"))
        if pts:
            series.append((label, pts, colour, b))
    record("dflash_sweep",
           series=[{"label": lb, "points": pt, "baseline": bs}
                   for lb, pt, _c, bs in series])
    if not series:
        print("  no DFlash data - skipping sweep chart")
        return

    # 9.2 wide put run J's n_max 4 label into run K's n_max 3 label: on a log
    # base 2 axis those two are close together, and both labels are wide.
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(11.4, 7.4), sharex=True,
                                  gridspec_kw={"height_ratios": [2.1, 1]})
    # The two runs land almost on top of each other at n_max 4 and 8 - which is
    # the replication, and also a label collision. Push each series' labels to a
    # different distance rather than letting them overprint.
    for k, (label, pts, colour, b) in enumerate(series):
        # circle and square, not two circles: the two runs were separated by
        # colour alone, and a second channel costs nothing.
        #
        # Run J's circle is OPEN and larger. The two runs replicate at n_max 4 and
        # 8, which is why both are drawn, and where they replicate they are 1.31
        # and 0.98 points apart on an axis where a filled marker covers about 1.8:
        # the later series simply covered the earlier one and only a crescent of it
        # showed, so the one thing the figure exists to show was the one thing
        # hidden. An open ring with the other run's marker inside it is what
        # agreement should look like.
        fill = (dict(mfc="none", mew=1.8, ms=9.5, zorder=4) if k == 0
                else dict(ms=6.5, zorder=3))
        ax.errorbar([p[0] for p in pts], [p[1] for p in pts],
                    yerr=[p[3] for p in pts], marker="os"[k % 2], lw=2.0, capsize=4,
                    color=colour, label=f"{label}   (its baseline: {b:.1f} tok/s)",
                    **fill)
        ax2.plot([p[0] for p in pts], [p[2] for p in pts], marker="os"[k % 2], lw=1.8,
                 color=colour, **fill)
        # One offset per SERIES, not one per sign. The offsets were 24 points up
        # for a positive value and 31 down for a negative one, which on this axis
        # is 7 to 9 percentage points, so "+18.7%" was drawn level with the 26
        # gridline; and because the two series used different distances in the same
        # direction, "-15.8%" was drawn ABOVE "-14.8%", a pair whose vertical order
        # contradicted the values it carries.
        #
        # Every label in a series now sits the same distance from its own point, so
        # within a series the order of the labels is the order of the values, and
        # the two series go to opposite sides, so they cannot collide even at
        # n_max 4 and 8 where the runs replicate. Level with the point was tried
        # first and does not fit: on a log base 2 axis the gap between 3 and 4 is
        # 296 pixels and two of these labels are 422.
        # Up and to the RIGHT for run J, down and to the LEFT for run K: both
        # curves fall from left to right, so those are the two sides the curve is
        # not on. Straight above and straight below, the labels' white boxes took
        # segments out of the lines they annotate.
        dx, dy, va, ha = (16, 14, "bottom", "left") if k == 0 \
            else (-16, -14, "top", "right")
        for n, d, _acc, _sd in pts:
            ax.annotate(f"{d:+.1f}%", (n, d), textcoords="offset points",
                        xytext=(dx, dy), ha=ha, va=va,
                          color=figstyle.text_colour(colour),
                        bbox=dict(boxstyle="round,pad=0.15", fc="white",
                                  ec="none", alpha=0.85))

    ax.axhline(0, color=C_REF, ls="--", lw=1.2)
    # On the zero line itself, not in a corner: the corner it used to sit in is
    # exactly where the worst-performing point lands.
    # ...and at the LEFT end of that line it then sat on run K's n_max 1 label,
    # which is the leftmost point on the chart. The right end of the zero line is
    # empty on this data: the largest n_max is the deepest point of the cliff.
    # No white box. Anchored va="bottom" on the rule, the box's own padding put it
    # OVER the rule and erased ninety-one sampled points of the line's right end.
    # It sits in empty paper there and never needed one.
    ax.annotate("no speculation ", (0.988, 0.0),
                xycoords=ax.get_yaxis_transform(), textcoords="offset points",
                xytext=(0, 4), ha="right", va="bottom",
                color=figstyle.text_colour(C_REF))
    ys = [d for _, pp, _, _ in series for _, d, _a, _s in pp] + [0.0]
    pad = max(6.0, 0.12 * (max(ys) - min(ys)))
    # 1.9 and 1.6 reserved 29.6 % of the vertical range for labels thrown that far
    # from their points. They sit beside them now.
    ax.set_ylim(min(ys) - 1.0 * pad, max(ys) + 1.0 * pad)
    ax.set_ylabel("change in aggregate throughput\nagainst the same run's baseline (%)")
    # The subtitle read "two runs, independently" beside a claim about a plateau at
    # draft length 2 to 4. Run J has points at 4, 8 and 16 only: it contributes one
    # point to that plateau and cannot corroborate it. What the two runs do
    # independently is agree WHERE THEY OVERLAP, which is what it says now, and
    # both the plateau's extent and the overlap are read off the data rather than
    # typed.
    _shared = sorted(set.intersection(*({q[0] for q in pp}
                                        for _l, pp, _c, _bb in series)))
    _best = max(q[1] for _l, pp, _c, _bb in series for q in pp)
    _plateau = sorted({q[0] for _l, pp, _c, _bb in series for q in pp
                       if q[1] > 0.9 * _best})
    ax.set_title(
        f"DFlash self-speculation: a plateau at draft length "
        f"{min(_plateau)}-{max(_plateau)}, then a cliff\n"
        f"one binary, one placement policy, three repeats per point; the two runs "
        f"agree where they overlap, at {' and '.join(str(n) for n in _shared)}")
    ax.legend(loc="lower left", frameon=False)
    ax.grid(color="#dcdcdc", lw=0.6)
    ax.set_axisbelow(True)

    ax2.set_ylabel("draft tokens\naccepted (%)")
    ax2.set_xlabel("--spec-draft-n-max  (maximum tokens the drafter may propose)")
    ax2.set_xscale("log", base=2)
    # sorted(set(...)): the two runs share n_max 4 and 8, so this list had them
    # twice and matplotlib drew both tick labels, one exactly on the other.
    ax2.set_xticks(sorted({q[0] for _l, pts, _c, _b in series for q in pts}))
    ax2.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    # Room at both ends for the labels, which now sit to one side of their points:
    # at the left edge run K's n_max 1 label ran into the y axis tick labels. The
    # axis is log base 2, so the margin is a ratio, not an offset.
    _ns = [q[0] for _l, pp, _c, _b in series for q in pp]
    ax2.set_xlim(min(_ns) / 1.45, max(_ns) * 1.25)
    ax2.grid(color="#dcdcdc", lw=0.6)
    ax2.set_axisbelow(True)

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    _footer(fig, " Error bars are the run-to-run SD of the three repeats at that "
                 "draft length, on the quantity the line plots; the lower panel "
                 "is a pooled ratio and has none. "
                 " Runs J and K differ in context and fitter margin, so only their "
                 "deltas against their own baselines are comparable.")
    _guards(fig, "dflash_sweep")
    if not CHECK:  # --check verifies the numbers, it must not dirty the tree
        plt.savefig(OUT / "plot_dflash_sweep.png", dpi=figstyle.DPI, bbox_inches="tight")
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

    fit = [("thinking on", "matrix_L_thinkon_*/*__rep*.json", figstyle.BLUE),
           ("thinking off", "matrix_L_thinkoff_*/*__rep*.json", figstyle.VERMILION)]
    xs, ys = [], []
    fig, ax = plt.subplots(figsize=(9.4, 6.0))
    for label, pat, colour in fit:
        # Only the DFlash arms inform the line. spec-draft-n8 is a separate
        # draft model paying a full forward pass per token, so its cost per unit
        # of acceptance is a different quantity; it appears in the out-of-sample
        # markers instead, where it is the worst miss.
        d = series(pat, only="dflash")
        xs += [p[0] for p in d]; ys += [p[1] for p in d]
        ax.scatter([p[0] for p in d], [p[1] for p in d], s=34, color=colour, edgecolors=figstyle.EDGE, linewidths=figstyle.EDGE_LW,
                   alpha=0.8,
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

    # The thinking-on half on its own. Its crossing was written into this
    # figure's caption and into the audit README as the literal "45.4 %", with
    # nothing deriving it: the two other crossings on the same line are both
    # recorded and both cross-checked by analysis/verify_claims.py, and this one
    # was a number somebody had typed. It is computed here now, and it comes out
    # at 45.41, so the published figure was right and unguarded.
    _on = series(fit[0][1], only="dflash")
    xs_on = [q[0] for q in _on]
    ys_on = [q[1] for q in _on]
    mxo, myo = st.mean(xs_on), st.mean(ys_on)
    slope_on = (sum((a - mxo) * (b - myo) for a, b in zip(xs_on, ys_on))
                / sum((a - mxo) ** 2 for a in xs_on))
    brk_on = -(myo - slope_on * mxo) / slope_on

    record("acceptance_threshold", fitted=sorted(zip(xs, ys)),
           out_of_sample=sorted(oos), slope=slope, intercept=inter,
           break_even=brk, r=num / den,
           break_even_length_matched=brk_lm, slope_length_matched=slope_lm,
           break_even_thinking_on=brk_on, slope_thinking_on=slope_on,
           n_fitted=len(xs), n_fitted_length_matched=len(xs_lm),
           n_fitted_thinking_on=len(xs_on))

    # `lo, hi` was 15 and 95 and served two purposes at once: the range the fit
    # line is DRAWN over, and the axis limits. The axis then lost data. Two of the
    # sixty fitted points sit at 9.5 % and 13.6 % accepted, left of 15, so the
    # figure drew fifty-eight while its own legend said thirty and thirty, and the
    # two it dropped are the lowest-acceptance pair, the points with the most
    # leverage on the slope. The two purposes are now separate: the line is drawn
    # over the range it was fitted on, which is also the only range it is entitled
    # to, and the axis is set from everything the figure plots.
    lo, hi = min(xs), max(xs)
    ax.plot([lo, hi], [slope * lo + inter, slope * hi + inter], color="#7a7a82",
            lw=1.3, ls="-", label=f"least squares on run L  (r = {num / den:+.3f})")
    ax.axhline(0, color=figstyle.BLUE, ls="--", lw=1.1)
    # a green line and a blue line meaning different things is the pair to avoid
    # The figure used to draw ONE crossing and label it 48.2 %, while its own
    # caption said the confound moves it to 46.5, that the clean half gives 45.4,
    # and that the threshold should be read as 45 to 48. The number a reader took
    # away was the one drawn. All three crossings are the same fit on three
    # subsets of the same points, so the honest mark is the band they span, with
    # the full-sample line still drawn inside it.
    #
    # The band's FILL cannot be what carries it. At the opacity a fill has to have
    # to sit under the data it measures 1.1:1 against the paper, and WCAG 1.4.11
    # asks 3:1 of any mark that carries information. So the two crossings at the
    # edges are drawn as lines at full opacity, which measure 3.06:1, and the fill
    # between them is only an aid.
    _lo_b, _hi_b = min(brk, brk_lm, brk_on), max(brk, brk_lm, brk_on)
    ax.axvspan(_lo_b, _hi_b, color=figstyle.PURPLE, alpha=0.13, lw=0, zorder=0)
    for _b in (_lo_b, _hi_b):
        ax.axvline(_b, color=figstyle.PURPLE, ls=":", lw=1.6)
    # anchored at the TOP of the y range, this ran straight through the third
    # line of the legend, which is also upper left. The bottom of the same
    # vertical line is empty on every rendering of this figure.
    ax.annotate(f"break-even, {_lo_b:.1f} to {_hi_b:.1f} % accepted "
                f"(full sample {brk:.1f})",
                (_hi_b, ax.get_ylim()[0]),
                textcoords="offset points", xytext=(6, 12),
                  color=figstyle.text_colour(figstyle.PURPLE))
    _x = xs + [p[0] for p in oos]
    _pad = 0.04 * (max(_x) - min(_x))
    ax.set_xlim(min(_x) - _pad, max(_x) + _pad)
    ax.set_xlabel("draft tokens accepted (%)")
    ax.set_ylabel("change in decode rate against the matched baseline (%)")
    ax.set_title("Acceptance decides the sign, and only the sign\n"
                 "line fitted on run L alone; runs J and K are the out-of-sample test")
    ax.legend(loc="upper left", frameon=False)
    ax.grid(color="#dcdcdc", lw=0.6)
    ax.set_axisbelow(True)
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    # This figure spans both workloads and five repeats, so the shared footer
    # ("thinking on, three repeats") would be wrong on it.
    _footer(fig, f" Acceptance is llama.cpp's server-side counter; it under-reports on "
                 "arms that take speculative checkpoints (ERRATA A13), which is every "
                 "external-drafter point here and none of the fitted ones. "
                 "Half the fitted points are run L's thinking-off half, where the arms "
                 "generated DIFFERENT numbers of tokens (ERRATA A17): refitting without "
                 f"that confound moves the crossing to {brk_lm:.1f} % over "
                 f"{len(xs_lm)} points, and the thinking-on half alone gives "
                 f"{brk_on:.1f} % over {len(xs_on)}. The shaded band spans those "
                 f"three fits; the dotted line is the full-sample one. "
                 f"Read the threshold as 45-48 %.",
            base="2026-08-26, llama.cpp 3737e4137, one RTX 3090, "
                 "Qwen3.6-35B-A3B-UD-Q4_K_XL, greedy, ten prompts. Fitted points: run L, "
                 "both workloads, five repeats per arm, per prompt. Out-of-sample points: "
                 "runs J and K, whole arms, three repeats. "
                 "Controls and caveats: ERRATA.md, v4_audit_2026_08_25/README.md.")
    _guards(fig, "acceptance_threshold")
    if not CHECK:  # --check verifies the numbers, it must not dirty the tree
        plt.savefig(OUT / "plot_acceptance_threshold.png", dpi=figstyle.DPI, bbox_inches="tight")
    plt.close()
    if not CHECK:
        print(f"  wrote {(OUT / 'plot_acceptance_threshold.png').relative_to(ROOT)}")


# ------------------------------------------------------------- runs A, B ----
def plot_acceptance_correlation() -> None:
    """The retraction, as a figure. This repository was built around an anomaly.

    v1 published "100 % draft acceptance and still slower, therefore an MoE
    pathology". The legacy binary counted 194 draft tokens across the whole
    workload and reported all 194 accepted; the corrected one counts 16 590 and
    finds 29.7 % accepted. With numbers that mean what they say, decode rate
    tracks acceptance at r = +0.998 across the ten prompts and there is nothing
    left to explain. ERRATA A7.

    The README states this in one sentence and had no figure for it, while the
    finding it retracts had one. That asymmetry is what this exists to remove.

    Run A is NOT plotted. It is a different binary that drafted 85 times less, so
    a point from it on these axes would be read as comparable and is not; its two
    counts are in the annotation instead.
    """
    def _load(d):
        a: dict = defaultdict(list)
        for f in sorted(glob.glob(str(DATA / d / "*__rep*.json"))):
            a[json.load(open(f, encoding="utf-8"))["arm"]].append(
                json.load(open(f, encoding="utf-8")))
        return a
    A, B = _load("A_bcb5eeb64_legacy"), _load("B_master_3737e4137")
    if "draft-max8-matched" not in B or "baseline" not in B:
        print("  no run B data - skipping the correlation chart")
        return
    def _counts(arms, arm):
        return (sum(x["draft_n"] for r in arms[arm] for x in r["rows"]),
                sum(x["draft_n_accepted"] for r in arms[arm] for x in r["rows"]))
    a_dn, a_da = _counts(A, "draft-max8-matched")
    b_dn, b_da = _counts(B, "draft-max8-matched")

    per: dict = defaultdict(lambda: defaultdict(list))
    acc: dict = {}
    for arm, runs in B.items():
        for r in runs:
            for x in r["rows"]:
                per[x["tag"]][arm].append(x["predicted_per_second"])
                if x["draft_n"] and arm == "draft-max8-matched":
                    acc[x["tag"]] = (x["draft_n_accepted"], x["draft_n"])
    pts = sorted((100 * acc[t][0] / acc[t][1],
                  st.mean(per[t]["draft-max8-matched"]), t) for t in per)
    base = st.mean([v for t in per for v in per[t]["baseline"]])
    xs = [q[0] for q in pts]
    ys = [q[1] for q in pts]
    mx, my = st.mean(xs), st.mean(ys)
    slope = (sum((a - mx) * (b - my) for a, b in zip(xs, ys))
             / sum((a - mx) ** 2 for a in xs))
    inter = my - slope * mx
    r = (sum((a - mx) * (b - my) for a, b in zip(xs, ys))
         / ((sum((a - mx) ** 2 for a in xs)
             * sum((b - my) ** 2 for b in ys)) ** 0.5))
    record("acceptance_correlation",
           points=[{"prompt": t, "accepted_pct": x, "tok_s": y} for x, y, t in pts],
           baseline_tok_s=base, r=r, slope=slope, intercept=inter,
           legacy_drafted=a_dn, legacy_accepted=a_da,
           corrected_drafted=b_dn, corrected_accepted=b_da)

    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    fig.subplots_adjust(left=0.095, right=0.985, top=0.735, bottom=0.145)
    ax.plot([min(xs), max(xs)],
            [slope * min(xs) + inter, slope * max(xs) + inter],
            color="#7a7a82", lw=1.4, zorder=2)
    ax.scatter(xs, ys, s=64, color=C_ACTIVE, edgecolors=figstyle.EDGE,
               linewidths=figstyle.EDGE_LW, zorder=4)
    # code_small and reasoning land on each other, two points apart in acceptance
    # and equal in rate, so they go to opposite sides. Naming both is the point:
    # the two prompts that accept most are the two that are hardest.
    LAB = {"medium_chat": (0, -18, "center"), "code_small": (-9, -18, "right"),
           "reasoning": (9, -18, "left")}
    for x, y, t in pts:
        if t in LAB:
            dx, dy, ha = LAB[t]
            ax.annotate(t, (x, y), textcoords="offset points", xytext=(dx, dy),
                        ha=ha, size=10, color="#3f3f46")
    ax.axhline(base, color=C_REF, ls="--", lw=1.3, zorder=1)
    ax.text(0.985, base, f"no speculation, {base:.0f} tok/s ",
            transform=ax.get_yaxis_transform(), ha="right", va="bottom",
            size=11, color=figstyle.text_colour(C_REF))
    # below the rule, not across it: the rule spans the axes and any text that
    # crosses it is struck through, which `_ink_guard` now refuses.
    ax.annotate(
        f"The legacy binary counted {a_dn} draft tokens in this workload and\n"
        f"called all {a_da} accepted. The corrected one counts "
        + f"{b_dn:,}".replace(",", " ")
        + f" and finds\n{100 * b_da / b_dn:.1f} % accepted. With real numbers the "
        f"speed follows the\nacceptance: r = {r:+.3f}.",
        (0.03, 0.80), xycoords="axes fraction", ha="left", va="top", size=11,
        color="#1f1f24")
    ax.set_xlim(min(xs) - 4, max(xs) + 6)
    ax.set_ylim(0, base * 1.14)
    ax.set_xlabel("draft tokens accepted (%)", size=11)
    ax.set_ylabel("decode rate (tokens / second)", size=11)
    ax.tick_params(labelsize=11)
    ax.grid(color="#dcdcdc", lw=0.6)
    ax.set_axisbelow(True)
    for s_ in ("right", "top"):
        ax.spines[s_].set_visible(False)
    fig.text(0.008, 0.975,
             "The anomaly this repository was built around was a counter",
             ha="left", va="top", size=14)
    fig.text(0.008, 0.912,
             "one point per prompt, run B on llama.cpp 3737e4137, three repeats "
             "each", ha="left", va="top", size=11, color="#3f3f46")
    _footer(fig,
            base="2026-04-21 workload re-run on llama.cpp 3737e4137. Run A is the "
                 "archived legacy binary bcb5eeb64, whose acceptance counter is "
                 "the subject of ERRATA A7 and which is not plotted here because "
                 "it drafted eighty-five times less. Speculation is slower than "
                 "no speculation at every acceptance rate reached, and the reason "
                 "is ordinary: the best prompt accepts about half its draft "
                 "tokens and pays the draft path for all of them.",
            extra="", width=96)
    _guards(fig, "acceptance_correlation")
    if not CHECK:  # --check verifies the numbers, it must not dirty the tree
        plt.savefig(OUT / "plot_acceptance_correlation.png", dpi=figstyle.DPI,
                    bbox_inches="tight")
    plt.close()
    if not CHECK:
        print(f"  wrote {(OUT / 'plot_acceptance_correlation.png').relative_to(ROOT)}")


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
    # The run the documents are built on, named explicitly rather than picked by
    # a glob that a later run could win: a chart that silently switches source
    # would disagree with the table beside it. Override with BENCH_PLOT_RUN.
    # `tag` was set here and never read, so the footer said "Run O2" whichever
    # run was actually plotted - including the fallback, which is not a Latin
    # square.
    want = os.environ.get("BENCH_PLOT_RUN", "matrix_O2_latin_*")
    src = sorted(glob.glob(str(DATA / want)))
    if not src:
        src = sorted(glob.glob(str(DATA / "matrix_O_headtohead_*")))
    if not src:
        print("  no head-to-head data - skipping chart")
        return
    run = Path(src[-1])
    tag = run.name.split("_")[1]
    arms = load(f"{run.name}/*__rep*.json")

    # Whether every arm visited every position is a property of the run, not of
    # its name. `t_start` is time.perf_counter() inside the one driver process,
    # so it orders the arm-runs within a block.
    _blocks: dict = defaultdict(list)
    for _rs in arms.values():
        for _r in _rs:
            if _r.get("rows"):
                _blocks[_r["repeat"]].append(
                    (min(x["t_start"] for x in _r["rows"]), _r["arm"]))
    _pos: dict = defaultdict(list)
    for _rep in sorted(_blocks):
        for _i, (_, _a) in enumerate(sorted(_blocks[_rep])):
            _pos[_a].append(_i + 1)
    _n = len(_pos)
    _per = (len(next(iter(_pos.values()))) / _n) if _pos else 0
    balanced = bool(_pos) and _per == int(_per) and all(
        sorted(v) == sorted(list(range(1, _n + 1)) * int(_per)) for v in _pos.values())
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
    # Grouped by what the draft path IS, not by a causal claim: all three load a
    # separate drafter GGUF through -md. The short name is printed in a column of
    # its own rather than looked up in a legend: a legend costs a look away from
    # the row and back, and the drafter is a property of the row. This figure had
    # one, and the comment where it used to be records it covering two arms'
    # labels at "lower right" and two more in the right margin.
    FAMILY = {
        "spec-dflash": ("DFlash head",  C_DFLASH),
        "spec-mtp":    ("MTP head",     figstyle.PURPLE),
        "spec-draft":  ("0.8 B model",  C_ACTIVE),
        "ngram":       ("n-gram",       C_INACTIVE),
        "baseline":    ("none",         C_REF),
    }
    # The spread this arm shows ACROSS invocations, which is the caveat the
    # documents put on the headline and which the figure used to leave out
    # entirely: it drew a 1.6 point interval around a value that is the second
    # highest of twelve, and its caption called every interval narrow. ERRATA A16.
    _tl = SERIES.get("two_levels") or {}
    _by: dict = defaultdict(list)
    for _t, _v in zip(_tl.get("tags", []), _tl.get("values", [])):
        _by[_t].append(_v)
    _SP_ARM = "spec-dflash-n2"
    _SP_MEANS = {_t: sum(_v) / len(_v) for _t, _v in _by.items()}
    _SP_LO = min(_SP_MEANS.values()) if _SP_MEANS else 0.0
    _SP_HI = max(_SP_MEANS.values()) if _SP_MEANS else 0.0
    def fam(a):
        for k in FAMILY:
            if a.startswith(k):
                return k
        return "ngram"
    rows = []
    for a, runs in arms.items():
        dn = sum(x["draft_n"] for r in runs for x in r["rows"])
        da = sum(x["draft_n_accepted"] for r in runs for x in r["rows"])
        ng = sum(x["predicted_n"] for r in runs for x in r["rows"])
        # draft tokens proposed per token generated. Without it the acceptance
        # figure reads as success for an arm that hardly ever drafts:
        # ngram-map-k4v-m8 shows 50 % from 216 proposals over 27 000 tokens.
        rows.append((a, pooled(runs), (100*da/dn) if dn else None, fam(a),
                     (dn / ng) if ng else None))
    record("head_to_head", run=run.name, position_balanced=balanced,
           rows=[{"arm": a, "pooled": pl, "acceptance_pct": ac,
                  "draft_per_generated": dg, "family": fm}
                 for a, pl, ac, fm, dg in rows])
    rows.sort(key=lambda r: r[1])

    # ---- the layout ------------------------------------------------------
    # A forest plot, sized for the only place it is read: embedded in README.md,
    # where GitHub's content column is about 864 CSS pixels. The previous version
    # was authored 13.2 inches wide, so at that column its 10.4 point labels
    # rendered at 9.7 CSS pixels. DejaVu Sans has an x height of about 0.55, which
    # puts those labels at 0.161 degrees at fifty centimetres and 0.114 at arm's
    # length, against the 0.15 to 0.3 degree critical print size Legge and Bigelow
    # review in Journal of Vision 11(5):8. Below that band reading slows. The
    # canvas is 9.0 inches now and the labels are 11 point, which is 15 pixels at
    # that column. DPI does not enter into it.
    #
    # Two things paid for the width. The figure carried seven text columns and six
    # of them repeat the table eleven lines above it in the same document, which
    # the ICMJE recommendations of January 2026 forbid in terms: "do not duplicate
    # data in graphs and tables". Only `drafter` is not in that table, and by this
    # docstring it is the finding. And the estimate was a 7.5 point filled dot,
    # which on the old panel was 2.47 percentage points wide in data units while
    # the widest interval in the set is 1.59 and the narrowest 0.23: NONE of the
    # eight intervals was visible, at any DPI, and SUGI 31 paper 139-31 names the
    # harm exactly, that a reader takes the width of the symbol for the width of
    # the interval. A vertical tick cannot do that. The panel went from 37 % of
    # the figure to 61 %, and six of the eight intervals are now drawn wider than
    # the mark that sits in them.
    #
    # No axis break and no log scale. Correll, Bertini and Franconeri (CHI 2020)
    # measured that marking a break does not de-bias a reader, and a log scale
    # would compress the six informative arms from 44.8 % of the axis to 27.6 %,
    # because the outliers here are on the side where the ratio approaches zero.
    FIG_W = 9.0
    X_ARM, X_DRAFT = 0.008, 0.246
    AX_L, AX_R = 0.380, 0.985
    XLO, XHI = -80.0, 32.0
    TICK_LW = 1.1

    fig, ax = plt.subplots(figsize=(FIG_W, 5.0))
    fig.subplots_adjust(left=AX_L, right=AX_R, top=0.715, bottom=0.265)
    tr = mtransforms.blended_transform_factory(fig.transFigure, ax.transData)

    for i, (a, p_, acc, f, dpg) in enumerate(rows):
        label, colour = FAMILY[f]
        c = ci.get(a)
        if a == _SP_ARM:
            # the twelve invocation means as MARKS, not a band: the README says in
            # terms that this range is "a bound on what was observed, not a
            # confidence interval", and a band with hard edges reads as a
            # distribution. Drawn above the row so they cannot be taken for the
            # interval that belongs to it.
            for _m in sorted(_SP_MEANS.values()):
                ax.plot([_m, _m], [i + 0.22, i + 0.40], color=colour, lw=1.1,
                        alpha=0.9, solid_capstyle="butt", zorder=3)
            ax.plot([_SP_LO, _SP_HI], [i + 0.31, i + 0.31], color=colour, lw=0.8,
                    zorder=2)
        if c:
            lo, hi = c["ci95_t_pct"]
            # no end caps: at one to four pixels the two caps merge back into the
            # blob the tick was chosen to avoid
            ax.plot([lo, hi], [i, i], color=colour, lw=3.0,
                    solid_capstyle="butt", zorder=3)
            ax.plot([c["point_pct"]] * 2, [i - 0.20, i + 0.20], color="#1f1f24",
                    lw=TICK_LW, solid_capstyle="butt", zorder=5)
        else:
            ax.plot([0], [i], "D", ms=6.5, color="white", mec=colour, mew=1.6,
                    zorder=5)
        ax.text(X_ARM, i, a, transform=tr, ha="left", va="center", clip_on=False,
                family="DejaVu Sans Mono", size=11,
                color=figstyle.text_colour(colour))
        ax.text(X_DRAFT, i, label, transform=tr, ha="left", va="center",
                clip_on=False, size=11, color="#3a3a42")

    ax.axvline(0, color=C_REF, ls="--", lw=1.2, zorder=1)
    ax.set_ylim(-0.9, len(rows) - 0.05 + 0.5)
    ax.set_yticks([])
    ax.set_xlim(XLO, XHI)
    for s_ in ("left", "right", "top"):
        ax.spines[s_].set_visible(False)
    ax.grid(axis="x", color="#dcdcdc", lw=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=11)
    # one row under the ticks: direction at the two ends, the quantity in the
    # middle. Cochrane's technical supplement on graphs of statistical data asks
    # for the direction of effect below the plot (3.9); the middle label already
    # names the quantity, so the ends only have to say which way is which.
    for _fx, _s, _ha, _sz, _c in (
            (0.0, "slower", "left", 10, "#3f3f46"),
            (0.5, "change against no speculation (%)", "center", 11, "#1f1f24"),
            (1.0, "faster", "right", 10, "#3f3f46")):
        ax.text(_fx, -0.16, _s, transform=ax.transAxes, ha=_ha, va="top",
                size=_sz, color=_c)
    fig.text(0.008, 0.975,
             "Eight speculative configurations against one baseline",
             ha="left", va="top", size=14)
    _wci = ci[_SP_ARM]["ci95_t_pct"][1] - ci[_SP_ARM]["ci95_t_pct"][0]
    fig.text(0.008, 0.925,
             f"bar: the 95 % t interval over run {tag}'s {n_blocks} blocks. Tick: "
             f"the point estimate.\nTicks above the top row: {len(_SP_MEANS)} "
             f"invocations of {_SP_ARM} in one day. They span\n"
             f"{_SP_HI - _SP_LO:.1f} points against that arm's {_wci:.1f}, and it "
             f"is the only arm measured repeatedly.",
             ha="left", va="top", size=11, color="#3f3f46")
    _w = [c["ci95_t_pct"][1] - c["ci95_t_pct"][0] for c in ci.values()]
    _footer(fig,
            base=f"2026-08-26, llama.cpp 3737e4137, one RTX 3090, "
                 f"Qwen3.6-35B-A3B-UD-Q4_K_XL, greedy, thinking on, ten prompts, "
                 f"run {tag}. Intervals are {min(_w):.2f} to {max(_w):.2f} points "
                 f"wide; the two 0.8 B arms' are at the tick's width. Values and "
                 f"caveats: the table and the note above this figure, and "
                 f"ERRATA.md.",
            extra="", width=96)
    _guards(fig, "head_to_head")
    if not CHECK:  # --check verifies the numbers, it must not dirty the tree
        plt.savefig(OUT / "plot_head_to_head.png", dpi=figstyle.DPI, bbox_inches="tight")
    plt.close()
    if not CHECK:
        print(f"  wrote {(OUT / 'plot_head_to_head.png').relative_to(ROOT)}")



# ------------------------------------------------------- runs M1-U6 ----
def plot_two_levels() -> None:
    """The A16 finding, which had no figure.

    Every block of every comparable run, for the one arm that moves. Left: the
    43 values, ordered by clock, coloured by which level they sit in. Right:
    run O3 alone, where the transition happens inside one run and no other arm
    goes with it.
    """
    TGT = "707a55a8a4397ecde44de0c499d3e68c1ad1d240d1da65826b4949d1043f4450"
    runs = []
    for d in sorted(glob.glob(str(DATA / "matrix_*"))):
        mp = os.path.join(d, "manifest.json")
        if not os.path.isfile(mp):
            continue
        m = json.load(open(mp, encoding="utf-8"))
        if not (m.get("think") == "on" and m.get("concurrency") == 1
                and m.get("prompt_set", "v1") == "v1" and str(m.get("ctx")) == "8192"
                and str(m.get("fit_target")) == "3072"
                and m.get("target_sha256") == TGT
                and "spec-dflash-n2" in (m.get("arms") or {})
                # A16 is "twelve times IN ONE DAY", and `verify_claims.py` uses
                # exactly this rule. Run T4 satisfies everything else and ran on
                # 2026-08-27; it is A16's addendum, not one of the 43 blocks.
                # The two rules being separate copies is how they drifted apart
                # once already, so `tests/` asserts they still agree.
                and str(m.get("created", "")).startswith("2026-08-26")):
            continue
        runs.append((m["created"], os.path.basename(d)))
    if not runs:
        print("  no comparable runs - skipping the two-level chart")
        return
    runs.sort()

    def blocks(run, arm):
        out = {}
        for f in glob.glob(str(DATA / run / f"{arm}__rep*.json")):
            r = json.load(open(f, encoding="utf-8"))
            out[r["repeat"]] = (1000 * sum(x["predicted_n"] for x in r["rows"])
                                / sum(x["predicted_ms"] for x in r["rows"]))
        return out

    xs, ys, tags = [], [], []
    for created, run in runs:
        b, a = blocks(run, "baseline"), blocks(run, "spec-dflash-n2")
        tag = run.split("_")[1]
        for k in sorted(b):
            if k in a:
                xs.append(len(xs))
                ys.append(100 * (a[k] / b[k] - 1))
                tags.append(f"{tag}")
    SPLIT = 23.0
    hi = [y for y in ys if y >= SPLIT]
    lo = [y for y in ys if y < SPLIT]
    record("two_levels", n=len(ys), split=SPLIT,
           high_n=len(hi), high_mean=st.mean(hi), high_sd=st.stdev(hi),
           low_n=len(lo), low_mean=st.mean(lo), low_sd=st.stdev(lo),
           values=[round(y, 3) for y in ys], tags=tags)

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.2, 5.4),
                                  gridspec_kw={"width_ratios": [2.05, 1]})
    # A green and a red told the two groups apart and nothing else did: same
    # marker, same size, colour carrying the whole distinction, which is the
    # pair about eight per cent of men cannot separate. Blue against vermilion
    # from the Okabe and Ito set, and a different marker for each group, so the
    # split survives being printed in grey as well.
    C_HI, C_LO = figstyle.BLUE, figstyle.VERMILION
    M_HI, M_LO = "o", "^"
    for x, y, t in zip(xs, ys, tags):
        ax.scatter(x, y, s=52, color=(C_HI if y >= SPLIT else C_LO), edgecolors=figstyle.EDGE, linewidths=figstyle.EDGE_LW,
                   marker=(M_HI if y >= SPLIT else M_LO),
                   zorder=3)
    # the SD of each group was computed, serialised into plot_data.json and then
    # never drawn: a mean line with no spread, in a figure whose whole subject is
    # how far apart the two groups are
    for val, sd, c, lab in (
            # The three quantities a reader compares between these two rows are
            # the count, the mean and the SD, and they sat behind leading phrases
            # of different widths ("above the +23 % split," against "below it,"),
            # so none of the three lined up. They lead now and the prose trails:
            # the fixed words are identical between the rows and the numbers are
            # padded to equal width, so the three columns line up without the
            # legend having to become a table.
            (st.mean(hi), st.stdev(hi), C_HI,
             f"{len(hi):2d} blocks   mean {_fig_num(st.mean(hi), 5, 1, True)} %"
             f"   SD {st.stdev(hi):.2f}   above the {SPLIT:+.0f} % split"),
            (st.mean(lo), st.stdev(lo), C_LO,
             f"{len(lo):2d} blocks   mean {_fig_num(st.mean(lo), 5, 1, True)} %"
             f"   SD {st.stdev(lo):.2f}   below it")):
        ax.axhspan(val - sd, val + sd, color=c, alpha=0.12, lw=0)
        ax.axhline(val, color=c, ls="--", lw=1.2, alpha=0.8, label=lab)
    seen, ticks, labels = set(), [], []
    for x, t in zip(xs, tags):
        if t not in seen:
            seen.add(t); ticks.append(x); labels.append(t)
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.set_xlabel("block, ordered by clock; ticks mark the first block of each run")
    ax.set_ylabel("spec-dflash-n2 against the baseline in the same block (%)")
    ax.set_title(f"One arm, {len(ys)} blocks, one day: {min(ys):+.1f} % to {max(ys):+.1f} %, "
                 f"clustered by run\n"
                 f"draft_n is 2441 and acceptance 72.3 % in every one of them")
    # A frame, and a background. Without one the two dashed swatches are two more
    # dashed marks inside the data area, at a height that means nothing, in the
    # same style and colour as the two that do mean something.
    ax.legend(loc="lower left", frameon=True, facecolor="white",
              edgecolor="#d0d0d0", framealpha=0.94)
    ax.grid(axis="y", color="#dcdcdc", lw=0.6)
    ax.set_axisbelow(True)

    o3 = sorted(glob.glob(str(DATA / "matrix_O3_latin_*")))
    if o3:
        run = Path(o3[-1]).name
        man = json.load(open(DATA / run / "manifest.json", encoding="utf-8"))
        # NOT C_LO. Vermilion means "below the +23 % split" in the panel two
        # inches to the left, and this line is one arm, not a level: the same hue
        # carried two meanings in two panels drawn side by side and neither legend
        # said so. Green is what this arm's family is drawn in everywhere else in
        # this file.
        _others = 0
        for arm in man["arms"]:
            v = blocks(run, arm)
            if 0 not in v:
                continue
            rel = [100 * (v[k] / v[0] - 1) for k in sorted(v)]
            focus = arm == "spec-dflash-n2"
            _others += 0 if focus else 1
            ax2.plot(sorted(v), rel, marker="o" if focus else ".",
                     lw=2.0 if focus else 0.9,
                     color=C_DFLASH if focus else figstyle.GREY,
                     zorder=3 if focus else 1,
                     label="spec-dflash-n2" if focus else None)
        # The grey lines were never named: not in a legend, not in the title, not
        # in the caption. How many there are is counted, not typed.
        ax2.plot([], [], marker=".", lw=0.9, color=figstyle.GREY,
                 label=f"the other {_others} arms in run O3")
        ax2.axhline(0, color="#8a8a92", lw=0.8)
        ax2.set_xlabel("block within run O3")
        ax2.set_ylabel("change from that arm's own first block (%)")
        # Three lines, not two: over a panel 1047 pixels wide the two-line form
        # rendered 1340 wide, overflowed its own panel by 146 pixels on the left,
        # and landed on that panel's topmost y tick label.
        ax2.set_title("O3: the transition happens\ninside one run, and no\n"
                      "other arm goes with it")
        # BELOW the panel. At lower left the box covered blocks 4 and 5, the two
        # lowest points on it and the transition it is about; at lower right it
        # still reached them, because the second entry is long and the panel is
        # narrow. Outside the axes it cannot cover a point at all, and it cannot be
        # read as a mark either, so it does not need the frame the left panel's
        # does.
        ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=1,
                   frameon=False)
        ax2.grid(color="#dcdcdc", lw=0.6)
        ax2.set_axisbelow(True)
    plt.tight_layout(rect=[0, 0.07, 1, 1])
    _footer(fig,
            base="2026-08-26, llama.cpp 3737e4137, one RTX 3090, "
                 "Qwen3.6-35B-A3B-UD-Q4_K_XL, greedy, thinking on, ten prompts, "
                 "--fit-target 3072. Every block is one arm-run of the arm against "
                 "the no-speculation arm-run in the same block. ERRATA A16.",
            extra=" The shaded band around each dashed line is one standard "
                    "deviation of that group's blocks."
                    " The no-speculation baseline over these runs holds 115.72-117.25 "
                  "tok/s, a CV of 0.42 %. Every run produced byte-identical output. "
                  "The +23 % split is where the second-widest gap in the sorted values "
                  "sits and leaves 11 of 12 runs whole; it is a reading aid, not a fitted "
                  "boundary. The widest gap, 2.06 pp, isolates run U3 at the bottom.")
    _guards(fig, "two_levels")
    if not CHECK:
        plt.savefig(OUT / "plot_two_levels.png", dpi=figstyle.DPI, bbox_inches="tight")
    plt.close()
    if not CHECK:
        print(f"  wrote {(OUT / 'plot_two_levels.png').relative_to(ROOT)}")


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
    plot_acceptance_correlation()
    # before the head to head, which reads its recorded values: the caveat on the
    # headline is the spread this run measures, and a figure that draws the
    # narrow interval without it is the false precision ERRATA A16 exists about.
    plot_two_levels()
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
