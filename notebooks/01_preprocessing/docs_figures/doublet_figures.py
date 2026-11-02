"""Draws the schematics used in 01_Preprocessing_Docs.md.

Usage: python doublet_figures.py <output_dir> <path/to/pipelines>
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

OUT = sys.argv[1]
PIPELINES = sys.argv[2]
sys.path.insert(0, PIPELINES)

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8a8985"
BLUE, BLUE_T = "#2a78d6", "#cde2fb"
ORANGE, ORANGE_T = "#eb6834", "#fbdccd"
GRAY_T = "#f0efec"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "text.color": INK,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK2,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})


def box(ax, x, y, w, h, text, fc=SURFACE, ec=MUTED, weight="normal", size=9.5):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc=fc, ec=ec, lw=1.4))
    ax.text(x, y, text, ha="center", va="center", fontsize=size, weight=weight, color=INK, linespacing=1.35)


def arrow(ax, p0, p1, label=None, label_xy=None):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=12, lw=1.3, color=INK2,
                                 shrinkA=0, shrinkB=0))
    if label:
        lx, ly = label_xy
        ax.text(lx, ly, label, ha="center", va="center", fontsize=9, color=INK2, style="italic")


# ── 1. Reassignment: score bands + decision tree ──────────────────────────────
fig = plt.figure(figsize=(10.5, 5.6))
gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 2.3], hspace=0.25)

ax = fig.add_subplot(gs[0])
T, B = 0.36, 0.10  # band drawn wider than ±0.02 so it is visible
xmax = 0.80
ax.add_patch(Rectangle((0, 0), T - B, 1, fc=BLUE_T, ec=BLUE, lw=1.4))
ax.add_patch(Rectangle((T - B, 0), 2 * B, 1, fc=GRAY_T, ec=MUTED, lw=1.4, hatch="///"))
ax.add_patch(Rectangle((T + B, 0), xmax - T - B, 1, fc=ORANGE_T, ec=ORANGE, lw=1.4))
ax.text((T - B) / 2, 0.5, "keeps the donor label", ha="center", va="center", fontsize=10.5, weight="bold")
ax.text(T, 0.5, "band:\ncorrelation\ntest", ha="center", va="center", fontsize=9, linespacing=1.2,
        bbox=dict(fc=GRAY_T, ec="none", pad=1.5))
ax.text((T + B + xmax) / 2, 0.5, "Mixed", ha="center", va="center", fontsize=10.5, weight="bold")
ax.axvline(T, ymin=-0.08, ymax=1.08, color=INK, ls="--", lw=1.4, clip_on=False)
ax.set_xlim(0, xmax)
ax.set_ylim(0, 1)
ax.set_yticks([])
ax.set_xticks([T - B, T + B])
ax.set_xticklabels(["threshold − 0.02", "threshold + 0.02"], fontsize=9)
ax.text(T, 1.1, "threshold", ha="center", va="bottom", fontsize=9, color=INK2)
for s in ["left", "right", "top"]:
    ax.spines[s].set_visible(False)
ax.set_title("Final doublet call by score (frac_changed_bidir)  —  band width not to scale",
             loc="left", fontsize=11, weight="bold", pad=22)

ax = fig.add_subplot(gs[1])
ax.set_xlim(0, 11.6)
ax.set_ylim(0.3, 4.4)
ax.axis("off")
ax.text(0.05, 4.25, "Cells inside the band", fontsize=11, weight="bold", va="top")

yq, yo = 2.85, 0.85
box(ax, 0.95, yq, 1.5, 0.85, "cell in\nthe band", fc=GRAY_T)
box(ax, 3.3, yq, 2.55, 1.25, "Mixed cluster exists\nand corr(cell, Mixed)\ncan be computed?")
box(ax, 6.2, yq, 2.3, 1.25, "corr(cell, donor)\ncan be computed?")
box(ax, 9.45, yq, 2.55, 1.25, "corr(cell, Mixed)\n≥\ncorr(cell, donor)?")

box(ax, 3.3, yo, 2.0, 0.7, "keeps donor label", fc=BLUE_T, ec=BLUE, weight="bold")
box(ax, 6.2, yo, 1.5, 0.7, "Mixed", fc=ORANGE_T, ec=ORANGE, weight="bold")
box(ax, 8.55, yo, 1.5, 0.7, "Mixed", fc=ORANGE_T, ec=ORANGE, weight="bold")
box(ax, 10.6, yo, 2.0, 0.7, "keeps donor label", fc=BLUE_T, ec=BLUE, weight="bold")

arrow(ax, (1.72, yq), (2.0, yq))
arrow(ax, (4.6, yq), (5.03, yq), "yes", (4.82, yq + 0.2))
arrow(ax, (7.37, yq), (8.15, yq), "yes", (7.76, yq + 0.2))
arrow(ax, (3.3, yq - 0.63), (3.3, yo + 0.36), "no", (3.08, 1.85))
arrow(ax, (6.2, yq - 0.63), (6.2, yo + 0.36), "no", (5.98, 1.85))
arrow(ax, (8.95, yq - 0.63), (8.6, yo + 0.36), "yes", (8.5, 1.85))
arrow(ax, (9.95, yq - 0.63), (10.5, yo + 0.36), "no", (10.5, 1.85))

fig.text(0.125, 0.0,
         "Correlations use only variants observed in both profiles and need ≥ 40 of them (min_shared); "
         "otherwise they cannot be computed.\n"
         "Donor profile = median AF of the donor's cells; Mixed profile = median AF of the cells labelled Mixed in Patient imputation.",
         fontsize=8.5, color=INK2, va="top", linespacing=1.4)

fig.savefig(os.path.join(OUT, "doublet_reassignment.png"), dpi=110, bbox_inches="tight", pad_inches=0.15)
plt.close(fig)

# ── 2. Per-donor elbow cutoff on simulated scores ────────────────────────────
from DNA_Utils.qc_filters import piecewise_elbow_threshold

rng = np.random.default_rng(3)
scores = np.concatenate([rng.gamma(1.2, 0.025, 420), rng.uniform(0.18, 0.55, 30)])
scores = np.clip(scores, 0, 1)
kw = dict(min_frac=0.01, max_frac=0.45, slope_ratio_min=3.0)
thr, k, info = piecewise_elbow_threshold(scores, **kw)
boot = []
for _ in range(300):
    t, _, _ = piecewise_elbow_threshold(scores[rng.integers(0, len(scores), len(scores))], **kw)
    if np.isfinite(t):
        boot.append(t)
lo, hi = np.quantile(boot, [0.025, 0.975])

y = np.sort(scores)[::-1]
x = np.arange(len(y))
m1, b1 = np.polyfit(x[:k], y[:k], 1)
m2, b2 = np.polyfit(x[k:], y[k:], 1)

fig, ax = plt.subplots(figsize=(10.5, 4.6))
ax.axvspan(0, k, color=ORANGE_T, lw=0, zorder=0)
ax.axhspan(lo, hi, color=GRAY_T, lw=0, zorder=0)
ax.plot(x, y, color=BLUE, lw=2, zorder=3, label="cells sorted by score")
ax.plot(x[:k + 25], m1 * x[:k + 25] + b1, color=INK2, lw=1.3, ls=(0, (4, 3)), zorder=4)
ax.plot(x[max(k - 40, 0):], m2 * x[max(k - 40, 0):] + b2, color=INK2, lw=1.3, ls=(0, (4, 3)), zorder=4)
ax.axhline(thr, color=INK, lw=1.4, ls="--", zorder=4)
ax.axvline(k, color=INK, lw=1.2, ls=":", zorder=4)

ax.annotate("steep head: fitted line 1\n(≥ 3× steeper than line 2)", xy=(14, m1 * 14 + b1), xytext=(95, 0.40),
            fontsize=9.5, color=INK, arrowprops=dict(arrowstyle="-", color=MUTED, lw=1))
ax.annotate("flat tail: fitted line 2", xy=(300, m2 * 300 + b2), xytext=(240, 0.21),
            fontsize=9.5, color=INK, arrowprops=dict(arrowstyle="-", color=MUTED, lw=1))
ax.text(len(y) - 5, thr + 0.006, f"threshold = score at split point ({thr:.2f})", ha="right", va="bottom", fontsize=9.5)
ax.text(len(y) - 5, lo - 0.006, "95% bootstrap CI (300 resamples)", ha="right", va="top", fontsize=9, color=INK2)
ax.text(k + 8, 0.585, "◀ flagged: score ≥ threshold", fontsize=9.5, weight="bold", va="top")
ax.text(k + 8, 0.548, f"split point k = {k} cells", fontsize=9.5, va="top")

ax.set_xlim(0, len(y))
ax.set_ylim(0, 0.6)
ax.set_xlabel("cells, sorted from highest to lowest score")
ax.set_ylabel("frac_changed_bidir")
for s in ["right", "top"]:
    ax.spines[s].set_visible(False)
ax.grid(axis="y", color="#e6e5e1", lw=0.8, zorder=0)
ax.set_title("Per-donor cutoff: two-line elbow fit  —  simulated scores, 450 cells",
             loc="left", fontsize=11, weight="bold", pad=10)
fig.savefig(os.path.join(OUT, "doublet_elbow_cutoff.png"), dpi=110, bbox_inches="tight", pad_inches=0.15)
print("thr", round(thr, 3), "k", k, "ci", round(lo, 3), round(hi, 3), "ratio", round(info["ratio"], 1))
