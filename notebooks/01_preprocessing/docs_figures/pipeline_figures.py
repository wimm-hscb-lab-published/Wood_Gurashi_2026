"""Draws the schematics used in 01_Preprocessing_Docs.md.

Usage: python pipeline_figures.py <output_dir>
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

OUT = sys.argv[1]

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8a8985"
BLUE, BLUE_T = "#2a78d6", "#cde2fb"
ORANGE, ORANGE_T = "#eb6834", "#fbdccd"
YELLOW, YELLOW_T = "#eda100", "#fcebc2"
GRAY_T = "#f0efec"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "text.color": INK,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
})


def box(ax, x, y, w, h, text, fc=SURFACE, ec=MUTED, weight="normal", size=9.5, ls="-", ha="center", color=INK):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc=fc, ec=ec, lw=1.4, ls=ls))
    tx = x if ha == "center" else x - w / 2 + 0.18
    ax.text(tx, y, text, ha=ha, va="center", fontsize=size, weight=weight, color=color, linespacing=1.4)


def arrow(ax, pts, label=None, label_xy=None, ha="center"):
    for a, b in zip(pts[:-2], pts[1:-1]):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=INK2, lw=1.3, solid_capstyle="butt")
    ax.add_patch(FancyArrowPatch(pts[-2], pts[-1], arrowstyle="-|>", mutation_scale=12, lw=1.3, color=INK2,
                                 shrinkA=0, shrinkB=0))
    if label:
        ax.text(*label_xy, label, ha=ha, va="center", fontsize=9, color=INK2, style="italic")


def canvas(w, h, xmax, ymax, title):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, ymax)
    ax.axis("off")
    ax.set_title(title, loc="left", fontsize=11, weight="bold", pad=8)
    return fig, ax


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=110, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


# ── 1. Pipeline overview ─────────────────────────────────────────────────────
fig, ax = canvas(10.5, 3.9, 12.4, 4.3, "Preprocessing at a glance")
W, H = 2.2, 1.0
xs = [1.25, 3.7, 6.15, 8.6, 11.05]
y1, y2 = 3.1, 1.25
row1 = [("1  Load H5", "variant filter", None),
        ("2  Annotate", "variants", None),
        ("3  PCA → UMAP", "→ clustering*", None),
        ("4  Merge clusters", "by AF correlation", "multi"),
        ("5  Assign donors", "clusters → donors", "manual")]
row2 = [("6  Doublets", "→ Mixed", "multi"),
        ("7  Low quality", "→ Low_quality", None),
        ("8  Filter", "keep clean cells", None),
        ("9  Final patient", "labels", "manual"),
        ("10  Export", "per-donor H5", None)]
for row, y in [(row1, y1), (row2, y2)]:
    for x, (t1, t2, kind) in zip(xs, row):
        fc = YELLOW_T if kind == "manual" else SURFACE
        ec = YELLOW if kind == "manual" else MUTED
        box(ax, x, y, W, H, "", fc=fc, ec=ec, ls=(0, (4, 2.5)) if kind == "multi" else "-")
        ax.text(x, y + 0.17, t1, ha="center", va="center", fontsize=9.5, weight="bold")
        ax.text(x, y - 0.2, t2, ha="center", va="center", fontsize=8.6, color=INK2)
for row_y in (y1, y2):
    for a, b in zip(xs[:-1], xs[1:]):
        arrow(ax, [(a + W / 2 + 0.02, row_y), (b - W / 2 - 0.02, row_y)])
arrow(ax, [(xs[-1], y1 - H / 2), (xs[-1], (y1 + y2) / 2), (xs[0], (y1 + y2) / 2), (xs[0], y2 + H / 2)])

ax.add_patch(FancyBboxPatch((0.2, 0.12), 0.45, 0.28, boxstyle="round,pad=0.01,rounding_size=0.06", fc=YELLOW_T, ec=YELLOW, lw=1.2))
ax.text(0.78, 0.26, "manual input (edit the label map)", va="center", fontsize=8.8, color=INK2)
ax.add_patch(FancyBboxPatch((4.1, 0.12), 0.45, 0.28, boxstyle="round,pad=0.01,rounding_size=0.06", fc=SURFACE, ec=MUTED, lw=1.2, ls=(0, (4, 2.5))))
ax.text(4.68, 0.26, "multiplexed runs only", va="center", fontsize=8.8, color=INK2)
ax.text(7.6, 0.26, "* clustering runs for multiplexed runs only", va="center", fontsize=8.8, color=INK2)
save(fig, "overview_preprocessing.png")

# ── 2. min_mut_prct_cells clamp ──────────────────────────────────────────────
N = np.unique(np.round(np.logspace(np.log10(100), np.log10(20000), 1500)).astype(int))
req = np.array([max(5, min(20, round(0.01 * n))) for n in N])
fig, ax = plt.subplots(figsize=(10.5, 3.9))
ax.axvspan(100, 500, color=GRAY_T, lw=0, zorder=0)
ax.axvspan(2000, 20000, color=GRAY_T, lw=0, zorder=0)
ax.plot(N, req, color=BLUE, lw=2, zorder=3)
for n in [300, 1000, 2000, 5000, 10000]:
    r = max(5, min(20, round(0.01 * n)))
    ax.plot(n, r, "o", ms=8, color=BLUE, mec=SURFACE, mew=2, zorder=4)
    lab = f"{r} cells = {100 * r / n:.2f}%"
    if n == 300:
        ax.text(n, r + 1.2, lab, ha="center", va="bottom", fontsize=8.8)
    elif n == 2000:
        ax.text(n * 1.07, r - 1.2, lab.replace(" = ", "\n= "), ha="left", va="top", fontsize=8.8, linespacing=1.25)
    elif n == 1000:
        ax.text(n * 1.07, r - 0.8, lab, ha="left", va="top", fontsize=8.8)
    else:
        ax.text(n, r - 1.2, lab.replace(" = ", "\n= "), ha="center", va="top", fontsize=8.8, linespacing=1.25)
ax.text(np.sqrt(100 * 500), 24.3, "floor: at least 5 cells", ha="center", fontsize=9.5, weight="bold")
ax.text(np.sqrt(500 * 2000), 24.3, "1% of cells", ha="center", fontsize=9.5, weight="bold")
ax.text(np.sqrt(2000 * 20000), 24.3, "cap: at most 20 cells", ha="center", fontsize=9.5, weight="bold")
ax.set_xscale("log")
ax.set_xlim(100, 20000)
ax.set_ylim(0, 26.5)
ax.set_xticks([100, 300, 500, 1000, 2000, 5000, 10000, 20000])
ax.set_xticklabels(["100", "300", "500", "1,000", "2,000", "5,000", "10,000", "20,000"])
ax.minorticks_off()
ax.set_yticks([0, 5, 10, 15, 20])
ax.set_xlabel("N_CELLS (log scale)")
ax.set_ylabel("cells that must carry a variant")
ax.grid(axis="y", color="#e6e5e1", lw=0.8, zorder=0)
for s in ["right", "top"]:
    ax.spines[s].set_visible(False)
ax.set_title("min_mut_prct_cells: cells required to keep a variant  (labels show the resulting percentage)",
             loc="left", fontsize=11, weight="bold", pad=10)
save(fig, "min_mut_prct_cells.png")

# ── 3. Cluster-merge loop ────────────────────────────────────────────────────
fig, ax = canvas(10.5, 4.6, 11.6, 4.6, "Demultiplexing: merging correlated clusters")
yt, yb = 3.45, 1.25
box(ax, 1.0, yt, 1.75, 0.95, "clusters from\ngraph-community", fc=GRAY_T)
box(ax, 3.85, yt, 3.0, 1.05, "cluster × cluster correlation\nof mean AF profiles")
box(ax, 7.4, yt, 2.4, 1.05, "strongest pair\ncorr ≥ 0.90?")
box(ax, 10.5, yt, 2.0, 0.85, "stop: converged", fc=BLUE_T, ec=BLUE, weight="bold")
box(ax, 7.4, yb, 3.6, 1.35, "merge every pair with\ncorr > max(0.90, strongest − 0.02)\n(average linkage on 1 − corr);\nrelabel cells, plot new matrix")
box(ax, 3.85, yb, 2.6, 1.05, "≤ N_DONORS + 2\nclusters left?")
box(ax, 1.0, yb, 1.75, 0.85, "stop: target\nreached", fc=BLUE_T, ec=BLUE, weight="bold")

arrow(ax, [(1.9, yt), (2.33, yt)])
arrow(ax, [(5.37, yt), (6.18, yt)])
arrow(ax, [(8.62, yt), (9.48, yt)], "no", (9.05, yt + 0.2))
arrow(ax, [(7.4, yt - 0.55), (7.4, yb + 0.7)], "yes", (7.6, 2.35), ha="left")
arrow(ax, [(5.58, yb), (5.17, yb)])
arrow(ax, [(2.53, yb), (1.9, yb)], "yes", (2.2, yb + 0.2))
arrow(ax, [(3.85, yb + 0.55), (3.85, yt - 0.55)], "no: next iteration (max 10)", (4.05, 2.35), ha="left")

ax.text(0.05, 0.12, "Also stops early if only one cluster is left or no cluster correlation can be computed.",
        fontsize=8.6, color=INK2)
save(fig, "cluster_merge_loop.png")

# ── 4. Low-quality gates ─────────────────────────────────────────────────────
fig, ax = canvas(10.5, 5.6, 11.6, 5.9, "Low-quality removal: a cell is flagged if any active gate fires (per donor)")
gx, gw = 4.15, 7.9
gates = [
    (4.75, "Gate 1 · noisy genotype at germline-stable loci",
     "score = z(median |AF − donor median|) + z(1 − corr with donor) + 0.7·z(missing)\n"
     "flag if score ≥ max(median + 5 robust SD, 99.5th percentile), so at most ~0.5% of cells"),
    (3.05, "Gate 2 · global dropout",
     "dropout = fraction of variants with AF ≤ 1% or missing\n"
     "flag if dropout > min(median + 3.5 robust SD, 97.5th pct) and ≥ 2 binomial SE above it,\n"
     "or if dropout ≥ max(0.50, donor 95th percentile)"),
    (1.35, "Gate 4 · combined moderate risk",
     "flag if dropout and low-AF fraction are both ≥ the donor's 95th percentile\n"
     "and both ≥ 2 binomial SE above it  (low-AF = informative loci with AF < 5% or missing)"),
]
for y, head, body in gates:
    h = 1.45 if body.count("\n") == 2 else 1.2
    box(ax, gx, y, gw, h, "")
    ax.text(gx - gw / 2 + 0.18, y + h / 2 - 0.22, head, fontsize=9.6, weight="bold", va="top")
    ax.text(gx - gw / 2 + 0.18, y + h / 2 - 0.55, body, fontsize=8.6, color=INK2, va="top", linespacing=1.45)
    arrow(ax, [(gx + gw / 2 + 0.02, y), (8.5, y), (8.82, 3.05)] if y != 3.05 else [(gx + gw / 2 + 0.02, y), (8.67, y)])
ax.add_patch(Circle((8.95, 3.05), 0.28, fc=SURFACE, ec=INK2, lw=1.3))
ax.text(8.95, 3.05, "any", ha="center", va="center", fontsize=8.8, weight="bold")
arrow(ax, [(9.23, 3.05), (9.55, 3.05)])
box(ax, 10.55, 3.05, 1.95, 1.0, "Low_quality\n(removed later\nwith Mixed)", fc=ORANGE_T, ec=ORANGE, weight="bold", size=9)
ax.text(0.2, 0.1,
        "Off in this notebook: Gate 3, low AF at informative loci (use_low_af_gate=False), and LOH protection "
        "(protect_localized_loh=False).\nMixed cells are skipped and keep their label.",
        fontsize=8.5, color=INK2)
save(fig, "low_quality_gates.png")
print("done")
