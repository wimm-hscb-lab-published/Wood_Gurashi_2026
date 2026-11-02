"""Draws the schematics used in README.md (02_protein_filtering).

Usage: python protein_figures.py [output_dir]      (default: this folder)
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))

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


def box(ax, x, y, w, h, text="", fc=SURFACE, ec=MUTED, weight="normal", size=9.5, ls="-"):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc=fc, ec=ec, lw=1.4, ls=ls))
    if text:
        ax.text(x, y, text, ha="center", va="center", fontsize=size, weight=weight, linespacing=1.4)


def arrow(ax, pts, label=None, label_xy=None, ha="center"):
    for a, b in zip(pts[:-2], pts[1:-1]):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=INK2, lw=1.3)
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


# ── 1. Overview ──────────────────────────────────────────────────────────────
fig, ax = canvas(10.5, 2.9, 12.4, 3.0, "Protein filtering at a glance (per experiment)")
W, H, y = 2.6, 1.15, 1.75
steps = [
    ("1  Load", "per-patient H5\nfrom 01", None),
    ("2  Protein filter", "Q75 + 3 × IQR\nupper cut-off", None),
    ("3  Export", "filtered H5\nper patient", None),
    ("4  BAM split", "multiplexed runs,\nif a BAM exists", "optional"),
]
xs = [1.5, 4.6, 7.7, 10.8]
for x, (t1, t2, kind) in zip(xs, steps):
    box(ax, x, y, W, H, ls=(0, (4, 2.5)) if kind else "-")
    ax.text(x, y + 0.27, t1, ha="center", va="center", fontsize=9.6, weight="bold")
    ax.text(x, y - 0.2, t2, ha="center", va="center", fontsize=8.6, color=INK2, linespacing=1.3)
for a, b in zip(xs[:-1], xs[1:]):
    arrow(ax, [(a + W / 2 + 0.02, y), (b - W / 2 - 0.02, y)])
ax.add_patch(FancyBboxPatch((0.2, 0.35), 0.45, 0.28, boxstyle="round,pad=0.01,rounding_size=0.06", fc=SURFACE, ec=MUTED, lw=1.2, ls=(0, (4, 2.5))))
ax.text(0.78, 0.49, "skipped for single-donor experiments, or when no BAM file is found", va="center", fontsize=8.8, color=INK2)
save(fig, "overview_protein_filtering.png")

# ── 2. IQR gate on simulated totals ──────────────────────────────────────────
rng = np.random.default_rng(7)
totals = np.concatenate([rng.lognormal(np.log(1200), 0.45, 900), rng.lognormal(np.log(9000), 0.35, 25)])
q25, q75 = np.percentile(totals, [25, 75])
iqr = q75 - q25
thr = q75 + 3 * iqr
removed = int((totals > thr).sum())

fig, ax = plt.subplots(figsize=(10.5, 3.9))
bins = np.logspace(np.log10(totals.min()), np.log10(totals.max()), 60)
ax.hist(totals[totals <= thr], bins=bins, color=BLUE, alpha=0.85, label=f"kept ({len(totals) - removed} cells)")
ax.hist(totals[totals > thr], bins=bins, color=ORANGE, alpha=0.9, label=f"removed ({removed} cells)")
ax.axvspan(thr, totals.max() * 1.1, color=ORANGE_T, lw=0, zorder=0)
for v, lab, ls in [(q25, f"Q25 = {q25:,.0f}", ":"), (q75, f"Q75 = {q75:,.0f}", ":"), (thr, f"threshold = Q75 + 3 × IQR = {thr:,.0f}", "--")]:
    ax.axvline(v, color=INK, lw=1.3 if ls == "--" else 1.0, ls=ls)
ax.text(q25 * 0.97, ax.get_ylim()[1] * 0.92, f"Q25\n{q25:,.0f}", ha="right", va="top", fontsize=8.8)
ax.text(q75 * 1.03, ax.get_ylim()[1] * 0.92, f"Q75\n{q75:,.0f}", ha="left", va="top", fontsize=8.8)
ax.text(thr * 1.04, ax.get_ylim()[1] * 0.92, f"threshold\nQ75 + 3 × IQR\n= {thr:,.0f}", ha="left", va="top", fontsize=8.8, weight="bold")
ax.set_xscale("log")
ax.set_xlim(totals.min() * 0.9, totals.max() * 1.1)
ax.set_xlabel("total protein reads per cell (log scale)")
ax.set_ylabel("cells")
for s in ["right", "top"]:
    ax.spines[s].set_visible(False)
ax.legend(frameon=False, loc="upper left", fontsize=9)
ax.set_title("Protein filtering: cells above Q75 + 3 × IQR are removed (simulated data, 925 cells)",
             loc="left", fontsize=11, weight="bold", pad=10)
save(fig, "protein_iqr_gate.png")
print(f"IQR figure: Q25={q25:.0f} Q75={q75:.0f} IQR={iqr:.0f} thr={thr:.0f} removed={removed}")

# ── 3. BAM demultiplexing ────────────────────────────────────────────────────
fig, ax = canvas(10.5, 5.3, 12.0, 5.6, "BAM demultiplexing")
yt, yb = 4.35, 1.6
box(ax, 1.05, yt, 1.8, 0.9, "multiplexed\nexperiment?")
box(ax, 3.85, yt, 2.6, 1.0, "BAM file found?\nraw_data/bam/<EXP>/\n(multiplexed/ first)", size=8.8)
box(ax, 6.75, yt, 1.9, 0.9, "pysam\navailable?")
box(ax, 1.05, 2.75, 1.8, 0.75, "skipped", fc=GRAY_T, weight="bold")
box(ax, 3.85, 2.75, 2.6, 0.75, "skipped (message)", fc=GRAY_T, weight="bold")
box(ax, 6.75, 2.75, 1.9, 0.75, "stops with\nImportError", fc=ORANGE_T, ec=ORANGE, weight="bold", size=9)
arrow(ax, [(1.95, yt), (2.53, yt)], "yes", (2.24, yt + 0.2))
arrow(ax, [(5.17, yt), (5.78, yt)], "yes", (5.47, yt + 0.2))
arrow(ax, [(1.05, yt - 0.47), (1.05, 3.14)], "no", (1.25, 3.68), ha="left")
arrow(ax, [(3.85, yt - 0.52), (3.85, 3.14)], "no", (4.05, 3.65), ha="left")
arrow(ax, [(6.75, yt - 0.47), (6.75, 3.14)], "no", (6.95, 3.68), ha="left")
steps = [
    (9.85, 4.75, "index the BAM\n(if no .bai)"),
    (9.85, 3.65, "write barcode → patient table\n(protein-filtered cells, '-1' suffix)"),
    (9.85, 2.55, "sinto filterbarcodes\n(RG tag, CPU count − 2 processes)"),
    (9.85, 1.45, "move each patient BAM to its\nown folder and index it"),
]
for x, y, t in steps:
    box(ax, x, y, 3.6, 0.82, t, fc=BLUE_T, ec=BLUE, size=8.8)
arrow(ax, [(7.72, yt), (7.88, yt), (7.88, 4.75), (8.03, 4.75)], "yes", (7.8, 3.95))
for (x1, y1, _), (x2, y2, _) in zip(steps[:-1], steps[1:]):
    arrow(ax, [(x1, y1 - 0.42), (x2, y2 + 0.42)])
ax.text(0.15, 0.1,
        "Outputs: results/demultiplexing/<EXPERIMENT>/bam_demultiplexing_barcodes.tsv\n"
        "and raw_data/bam/<EXPERIMENT>/demultiplexed/<patient>/<patient>.bam (+ .bai).\n"
        "When the step is skipped, the notebook finishes normally.",
        fontsize=8.6, color=INK2, linespacing=1.45)
save(fig, "bam_demultiplexing.png")
print("done")
