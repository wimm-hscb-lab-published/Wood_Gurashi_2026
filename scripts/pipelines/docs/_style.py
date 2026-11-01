"""Shared drawing style for the package README schematics.

Each ``make_<package>_figures.py`` script in this folder imports these helpers and writes PNGs into
``<package>/docs/``. Run a script with the mosaic conda environment (~/miniconda3/envs/mosaic), e.g.::

    python docs/make_DNA_Utils_figures.py

Colour roles (same as the notebook docs): blue = kept / passes, orange = flagged / removed,
yellow = manual input, grey = neutral / in-between.
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8a8985"
GRID = "#e6e5e1"
BLUE, BLUE_T = "#2a78d6", "#cde2fb"
ORANGE, ORANGE_T = "#eb6834", "#fbdccd"
YELLOW, YELLOW_T = "#eda100", "#fcebc2"
AQUA, AQUA_T = "#1baf7a", "#c9eedf"
GRAY_T = "#f0efec"

PIPELINES = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "text.color": INK,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
})


def canvas(width, height, xmax, ymax, title=None):
    """Blank axes in data units (0..xmax, 0..ymax) for box-and-arrow diagrams."""
    fig, ax = plt.subplots(figsize=(width, height))
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, ymax)
    ax.axis("off")
    if title:
        ax.set_title(title, loc="left", fontsize=11, weight="bold", pad=8)
    return fig, ax


def box(ax, x, y, w, h, text="", fc=SURFACE, ec=MUTED, weight="normal", size=9.5, ls="-", ha="center"):
    """Rounded box centred at (x, y); text centred, or left-aligned with ha='left'."""
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc=fc, ec=ec, lw=1.4, ls=ls))
    if text:
        tx = x if ha == "center" else x - w / 2 + 0.18
        ax.text(tx, y, text, ha=ha, va="center", fontsize=size, weight=weight, color=INK, linespacing=1.4)


def arrow(ax, points, label=None, label_xy=None, ha="center"):
    """Poly-line arrow through ``points`` with an arrowhead on the last segment."""
    for a, b in zip(points[:-2], points[1:-1]):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=INK2, lw=1.3, solid_capstyle="butt")
    ax.add_patch(FancyArrowPatch(points[-2], points[-1], arrowstyle="-|>", mutation_scale=12, lw=1.3,
                                 color=INK2, shrinkA=0, shrinkB=0))
    if label:
        ax.text(*label_xy, label, ha=ha, va="center", fontsize=9, color=INK2, style="italic")


def node(ax, x, y, text, r=0.28):
    """Small circular junction (e.g. 'any', 'all')."""
    ax.add_patch(Circle((x, y), r, fc=SURFACE, ec=INK2, lw=1.3))
    ax.text(x, y, text, ha="center", va="center", fontsize=8.8, weight="bold")


def note(ax, x, y, text, size=8.6):
    ax.text(x, y, text, fontsize=size, color=INK2, va="bottom", linespacing=1.4)


def save(fig, package, name):
    """Write ``<package>/docs/<name>`` at 110 dpi and close the figure."""
    out = os.path.join(PIPELINES, package, "docs")
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, name)
    fig.savefig(path, dpi=110, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    return path
