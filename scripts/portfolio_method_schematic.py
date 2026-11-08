#!/usr/bin/env python3
"""Method schematic for the portfolio card: the multimodal mixture-of-experts
model this repo develops, evaluated under the subject-safe protocol. Card-first
layout: type floor 9.5pt (most text 10+), no section mini-titles — the eyebrow
carries the title and the chip flows speak for themselves. Architecture and
protocol values come from the frozen artifacts
(src/pd_drawing_analysis/moe.py, fusion.py, results/moe/*.json); no metrics
are drawn. Fieldnotes theme.

Note: band heights are budgeted in points (units x axes-height) so every
label fits; labels are drawn line-by-line with explicit anchors.

Usage: python scripts/portfolio_method_schematic.py [out.png]
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch
from PIL import Image, ImageFilter  # ships with matplotlib

# The site's loaded fonts: Source Sans 3 is the body voice, IBM Plex Mono
# the label/caption voice. Both are installed locally; matplotlib would
# silently fall back to DejaVu Sans without them.
plt.rcParams["font.sans-serif"] = ["Source Sans 3", "Helvetica Neue", "Arial",
                                   "DejaVu Sans"]
plt.rcParams["font.monospace"] = ["IBM Plex Mono", "Courier New",
                                  "DejaVu Sans Mono"]

SURFACE = "#e4e4d7"  # the gallery card-art light green
INK = "#25251f"
INK2 = "#52514e"
MUTED = "#6e6d66"
SAGE = "#89927a"
WINE = "#482728"
CRIMSON = "#a43b36"  # the placeholder artSVG red — the card's identity red
SUMI = "#1E342E"
CREAM = "#fbf9f4"

plt.rcParams.update({
    "font.family": "sans-serif",
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "text.color": INK,
})


def axes_height_in(ax) -> float:
    """Axes height in inches — for converting point sizes to data units."""
    return ax.get_position().height * ax.figure.get_figheight()


def multiline(ax, x, y, lines, fs, ha="center", va="center", **kw):
    """Draw line-by-line: each line its own single-line text. (x, y) is the
    block anchor; va applies to the block as a whole."""
    pitch = (fs * 1.30 / 72.0) / axes_height_in(ax)
    n = len(lines)
    offsets = [(n - 1) / 2 - i for i in range(n)]
    if va == "top":
        offsets = [(n - 1) - i for i in range(n)]
    elif va == "bottom":
        offsets = [-i for i in range(n)]
    for line, dy in zip(lines, offsets):
        ax.text(x, y + dy * pitch, line, fontsize=fs, ha=ha, va="center", **kw)


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        Path(__file__).resolve().parents[1] / "figures" / "method_schematic.png")
    out.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(6.42, 3.60))  # canvas is padded to the exact
    # .project-art aspect at save time (see below)
    gs = fig.add_gridspec(2, 1, height_ratios=(1.45, 1.15), hspace=0.55,
                          left=0.03, right=0.985, top=0.86, bottom=0.10)

    # ---- band 1: the model — one expert per modality, one learned gate ----
    ax = fig.add_subplot(gs[0])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    def chip(a, x, y, w, h, lines, fc, ec, tc=CREAM, fs=9.5):
        a.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008",
                                   facecolor=fc, edgecolor=ec, linewidth=1.5,
                                   zorder=3))
        multiline(a, x + w / 2, y + h / 2, lines, fs, color=tc,
                  fontweight="bold")

    def arrow(a, x0_, y0_, x1_, y1_, rad=0.0):
        a.annotate("", xy=(x1_, y1_), xytext=(x0_, y0_),
                   arrowprops=dict(arrowstyle="->", color=INK2, lw=1.9,
                                   connectionstyle=f"arc3,rad={rad}"))

    # inputs (left column) -> experts (middle) -> gate -> prediction
    chip(ax, 0.000, 0.585, 0.210, 0.315,
         ["tablet recording", "25 features"], SURFACE, SUMI, INK, fs=9.5)

    # The image input is drawn as the thing it is: an actual spiral scan from
    # the Kaggle arm (V01HE02, training/healthy). The datasets are untracked,
    # so the word-mark chip below is the fallback when the scan is absent.
    scan = (Path(__file__).resolve().parents[1] / "KaggleSpiral" / "drawings"
            / "spiral" / "training" / "healthy" / "V01HE02.png")
    thumb = plt.imread(scan) if scan.exists() else None
    if thumb is not None:
        # Display prep for thumbnail size: the pencil strokes are a thin,
        # faint minority of the scan, so at 30px they antialias into the
        # paper and vanish. Soft-threshold strokes dark on white, thicken
        # them (MinFilter dilates the dark mass), and LANCZOS-resample here
        # — matplotlib's bilinear pass skips a line this thin. Same drawing,
        # displayed the way an icon of it has to be.
        lum = thumb[..., :3].mean(axis=2)
        ramp = np.clip((lum - 0.55) / 0.25, 0.0, 1.0)  # strokes -> 0
        pil = Image.fromarray((ramp * 255).astype(np.uint8))
        pil = pil.filter(ImageFilter.MinFilter(5))
        thumb = np.asarray(pil.resize((76, 76), Image.LANCZOS),
                           dtype=float) / 255.0
    ax.add_patch(FancyBboxPatch((0.000, 0.075), 0.210, 0.315,
                                boxstyle="round,pad=0.008", facecolor=SURFACE,
                                edgecolor=CRIMSON, linewidth=1.5, zorder=3))
    if thumb is None:
        multiline(ax, 0.000 + 0.210 / 2, 0.075 + 0.315 / 2,
                  ["drawing image", "8,112-d features"], 9.5,
                  color=INK, fontweight="bold")
    else:
        # square in INCHES: extent is in data units, so width must be the
        # thumbnail's physical height divided by the axes width in inches
        y0_, y1_ = 0.0975, 0.3675
        h_in = (y1_ - y0_) * axes_height_in(ax)
        w_in = ax.get_position().width * fig.get_figwidth()
        x0_, x1_ = 0.015, 0.015 + h_in / w_in
        # aspect="auto": without it imshow forces aspect="equal" and
        # re-anchors the whole axes, collapsing the band's layout
        if thumb.ndim == 2:
            ax.imshow(thumb, extent=(x0_, x1_, y0_, y1_), cmap="gray",
                      aspect="auto", zorder=4, interpolation="bilinear")
        else:
            ax.imshow(thumb, extent=(x0_, x1_, y0_, y1_),
                      aspect="auto", zorder=4, interpolation="bilinear")
        ax.add_patch(plt.Rectangle((x0_, y0_), x1_ - x0_, y1_ - y0_,
                                   facecolor="none", edgecolor=INK,
                                   linewidth=0.8, zorder=5))
        # text moves right of the thumbnail; the image replaces the words
        xc = (x1_ + 0.203) / 2
        multiline(ax, xc, 0.075 + 0.315 / 2, ["8,112-d", "features"], 9.5,
                  color=INK, fontweight="bold")
    chip(ax, 0.305, 0.585, 0.195, 0.315, ["tabular expert"], SUMI, SUMI, fs=10)
    chip(ax, 0.305, 0.075, 0.195, 0.315, ["image expert"], CRIMSON, CRIMSON, fs=10)
    chip(ax, 0.585, 0.330, 0.195, 0.315,
         ["softmax gate", "sees both inputs"], SAGE, SAGE, fs=9.5)
    chip(ax, 0.820, 0.330, 0.165, 0.315,
         ["one prediction", "per person"], SUMI, SUMI, fs=9.5)

    arrow(ax, 0.212, 0.7425, 0.302, 0.7425)
    arrow(ax, 0.212, 0.2325, 0.302, 0.2325)
    arrow(ax, 0.502, 0.7425, 0.581, 0.545, rad=-0.25)
    arrow(ax, 0.502, 0.2325, 0.581, 0.430, rad=0.25)
    arrow(ax, 0.782, 0.4875, 0.817, 0.4875)

    ax.text(0.5, -0.065,
            "one gate over both modalities · early / late / stacked fusion alongside",
            fontsize=9.5, color=MUTED, ha="center", va="center",
            fontfamily="IBM Plex Mono")
    # attribution + honesty: the scan is an NC-ND dataset image, and the
    # two-modality layout is exercised on the synthetic paired fixtures
    # (the two real datasets here have different participants)
    ax.text(0.5, -0.165,
            "modality pairing: synthetic fixtures · sample scan: kaggle "
            "parkinson's drawings · CC BY-NC-ND 4.0",
            fontsize=7.5, color=MUTED, ha="center", va="center",
            fontfamily="IBM Plex Mono")

    ax.set_title(" ".join("MULTIMODAL MIXTURE OF EXPERTS"),
                 fontsize=14, color=INK2, loc="left", pad=30)

    # ---- band 2: the subject-safe protocol that evaluates it ---------------
    ax2 = fig.add_subplot(gs[1])
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis("off")

    chip(ax2, 0.000, 0.28, 0.220, 0.56, ["102 images →", "class-aware ids"],
         SURFACE, SUMI, INK, fs=9.5)
    chip(ax2, 0.265, 0.28, 0.300, 0.56,
         ["StratifiedGroupKFold", "5 outer × 3 inner folds", "group = participant"],
         SAGE, SAGE, CREAM, fs=9.5)
    chip(ax2, 0.605, 0.28, 0.190, 0.56, ["pipeline fitted", "inside folds"],
         SURFACE, SUMI, INK, fs=9.5)
    chip(ax2, 0.830, 0.28, 0.155, 0.56, ["one vote", "per person"],
         SUMI, SUMI, fs=9.5)
    for x0_, x1_ in ((0.222, 0.262), (0.567, 0.602), (0.797, 0.827)):
        ax2.annotate("", xy=(x1_, 0.56), xytext=(x0_, 0.56),
                     arrowprops=dict(arrowstyle="->", color=INK2, lw=1.7))

    ax2.text(0.5, 0.05,
             "fold boundaries assert disjoint train / eval subjects",
             fontsize=10, color=WINE, ha="center", va="center",
             fontfamily="IBM Plex Mono")

    # The gallery card box (.project-art, 320.7 x 160) shows previews with
    # object-fit: cover, so the tight render is padded to the box's exact
    # aspect — otherwise the centered crop amputates the title and the
    # caption line. The wings are the same #e4e4d7 as the card background,
    # making the padding invisible.
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=SURFACE, bbox_inches="tight",
                pad_inches=0.2)  # the card's cover crop eats ~12px per end
    buf.seek(0)
    img = plt.imread(buf)
    h, w = img.shape[:2]
    target = 320.7 / 160
    wing = matplotlib.colors.to_rgba(SURFACE)
    if w < h * target:  # too tall: widen symmetrically
        pad = int(round(h * target)) - w
        wings = np.full((h, 1, 4), wing)
        img = np.concatenate([wings.repeat(pad // 2, 1), img,
                              wings.repeat(pad - pad // 2, 1)], axis=1)
    elif h < w / target:  # too wide: pad top and bottom
        pad = int(round(w / target)) - h
        wings = np.full((1, w, 4), wing)
        img = np.concatenate([wings.repeat(pad // 2, 0), img,
                              wings.repeat(pad - pad // 2, 0)], axis=0)
    plt.imsave(out, img)
    print(f"written: {out} ({img.shape[1]}x{img.shape[0]}, "
          f"aspect {img.shape[1] / img.shape[0]:.3f})")


if __name__ == "__main__":
    main()
