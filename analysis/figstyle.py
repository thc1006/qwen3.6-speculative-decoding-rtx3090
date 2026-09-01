"""One place for what every figure in this repository looks like.

Measured against published standards for scientific figures on 2026-09-01, the
eight live figures failed four of them at once:

    raster resolution   150 DPI written, 300 the minimum for print
    type size           7.2 to 11.5 pt, against 10 to 12 for labels and 14 to
                        16 for titles; no figure had a title at the standard
    colour              a red (#c0504d) and a green (#4a7c59) told two arm
                        families apart with nothing else distinguishing them,
                        and the per-prompt heatmap used RdYlGn, which is the
                        textbook example of what not to use: about eight per
                        cent of men cannot separate those hues
    layout              the caption was drawn at the foot of the figure and
                        `bbox_inches="tight"` then clipped it at both edges,
                        while the x axis label was written underneath it, so
                        two lines of text overlapped on two figures

Vector output is deliberately NOT added. The standards call it preferred rather
than required and 300 DPI raster meets the stated minimum, and nine more files
that no check compares against the data would be nine more places for the tree
to drift from what it publishes, which is the defect this repository spends most
of its machinery on.

The palette is Okabe and Ito's, designed so that all of its hues stay separable
under the common forms of colour vision deficiency. Colour is never the only
channel here regardless: bars carry their value as text, the heatmap prints its
number in every cell, and the batching chart uses distinct markers.
"""
from __future__ import annotations

DPI = 300

RC = {
    "figure.dpi": DPI,
    "savefig.dpi": DPI,
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.titlesize": 14,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "savefig.bbox": None,
}

# Okabe and Ito, https://jfly.uni-koeln.de/color/
BLUE = "#0072B2"       # no speculation, the reference
VERMILION = "#D55E00"  # the external 0.8 B draft model
GREEN = "#009E73"      # DFlash
SKY = "#56B4E9"        # DFlash, second configuration
GREY = "#767676"       # drafter-free n-gram. #999999, the Okabe and Ito grey,
                       # measures 2.85:1 against white and WCAG 2.2's 1.4.11
                       # wants 3:1 for a graphical object a reader needs in
                       # order to understand the chart. Grey is neutral, so
                       # darkening it costs no hue separation; 4.54:1 here.
ORANGE = "#E69F00"     # spare
PURPLE = "#CC79A7"     # MTP: two greens, one for DFlash and one for MTP, are
                       # separable to normal vision and not to a deuteranope,
                       # and these two arms are the pair a reader most wants to
                       # tell apart. Reddish purple is far from both.
BLACK = "#000000"

# WCAG 2.2, 1.4.11 Non-text Contrast, wants 3:1 between a graphical object and
# the colour next to it. Measured against white, this palette gives 5.19 (blue),
# 3.87 (vermilion), 3.42 (green) and 3.06 (purple), which pass, and 2.31 (sky),
# which does not. Rather than distort a hue that was chosen for its separation
# under colour vision deficiency, every filled shape gets a dark edge: the
# colour adjacent to the fill is then the edge, and the edge is 12.63:1. This is
# the remedy the criterion's own understanding document describes.
#
# The criterion exempts "heatmaps and other situations where changing the
# colours changes the meaning", so the diverging scale is out of its scope; the
# numbers printed in those cells are text and are covered by 1.4.3 instead.
# 1.4.3 Contrast (Minimum) wants 4.5:1 for type below 18 pt, and this palette was
# chosen for FILLS. Against white it measures 3.87:1 (vermilion), 3.42 (green),
# 3.06 (reddish purple) and 2.31 (sky): a value label printed in its own series
# colour is text, and fails a criterion the bar it sits beside passes. Recolouring
# the series would cost the separation the palette exists for, so each hue keeps a
# darkened twin used for TYPE ONLY. Measured against white: 5.34, 5.48, 5.58, 5.85.
# Blue (5.19) and the darkened grey (4.54) already pass and are their own twins.
TEXT = {
    VERMILION: "#B04E00",
    GREEN: "#00785A",
    PURPLE: "#A14A78",
    SKY: "#1A6B96",
    BLUE: BLUE,
    GREY: GREY,
    BLACK: BLACK,
}


def text_colour(fill: str) -> str:
    """The type colour to use for a label that belongs to `fill`."""
    return TEXT.get(fill, fill)


def on_fill(rgba) -> str:
    """Black or white, whichever a reader can actually read on this cell.

    The heatmap prints a number in every cell and the cells run from deep red to
    deep blue. One fixed ink cannot serve both ends: black on the darkest cells
    of either arm is the pair this figure had, and 44, the worst measurement in
    the matrix, was the least legible number on it. The colour scale itself is
    exempt from 1.4.11 as essential, but what is printed over it is text and 1.4.3
    applies, so the ink is chosen per cell at the standard crossover luminance.
    """
    r, g, b = (float(c) for c in rgba[:3])

    def lin(c):
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    lum = 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b)
    return BLACK if lum > 0.179 else "#FFFFFF"

EDGE = "#333333"
EDGE_LW = 0.8

# Diverging, for a scale centred on "the same as the baseline". ColorBrewer's
# red-blue is safe under deuteranopia and protanopia; RdYlGn is not.
#
# `RdBu`, NOT `RdBu_r`. The reversed one was tried first and inverted the
# meaning of the figure: it puts red at the HIGH end, so a heatmap of "per cent
# of the baseline" drew 100 %, which is the good outcome, in deep red and 44 %,
# which is the worst measurement in the matrix, in blue. Colourblind-safe and
# backwards is not an improvement on unsafe and right. This way the alarming
# colour is on the alarming end.
DIVERGING = "RdBu"

FOOT_SIZE = 10       # the caption, at the standard's floor rather than 7.2
FOOT_BOTTOM = 0.20   # of the figure height, reserved so nothing is clipped


def apply(plt) -> None:
    plt.rcParams.update(RC)


def footer(fig, text: str, width: int = 150) -> None:
    """Draw the caption just below whatever the figure already occupies.

    Three wrong versions preceded this one, each fixing the last one's damage.
    Drawn inside the figure at y=0.004 it landed on top of the x axis label and
    `bbox_inches="tight"` then clipped it at both edges. Given reserved space
    with `subplots_adjust` and no tight bbox, nothing outside the axes was on
    the canvas any more: the title lost a character, the legend covered two
    arms' labels, and the caption ran off both sides. Put back outside the
    figure at a fixed y=-0.04 with the tight bbox restored, it was finally
    correct and readable, and left a band of empty paper between itself and the
    legends, because a fixed offset cannot know how far below the axes a legend
    reached.

    So the offset is measured rather than chosen. The figure is drawn once, the
    lowest edge of everything already on it is found in figure coordinates, and
    the caption goes a fixed small pad below THAT.
    """
    import textwrap
    body = "\n".join(textwrap.wrap(" ".join(text.split()), width))
    fig.canvas.draw()
    inv = fig.transFigure.inverted()
    lows = []
    for ax in fig.axes:
        for a in [ax, ax.title, ax.xaxis.label, *ax.get_xticklabels()] + (
                [ax.get_legend()] if ax.get_legend() else []):
            try:
                bb = a.get_window_extent(fig.canvas.get_renderer())
            except Exception:
                continue
            if bb.height and bb.width:
                lows.append(inv.transform((0, bb.y0))[1])
    y = (min(lows) if lows else 0.0) - 0.035
    fig.text(0.5, y, body, ha="center", va="top",
             fontsize=FOOT_SIZE, color="#3f3f46")


def save(plt, fig, path) -> None:
    """Every figure is written the same way, once, here.

    `bbox_inches="tight"` is not optional: the legends sit outside the axes and
    the caption sits below the figure, and without it neither is on the canvas.
    """
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
