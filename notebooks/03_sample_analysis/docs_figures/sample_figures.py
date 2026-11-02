"""Draws the schematics used in README.md (03_sample_analysis).

Usage: python sample_figures.py [output_dir]      (default: this folder)
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


def box(ax, x, y, w, h, text="", fc=SURFACE, ec=MUTED, weight="normal", size=9.5, ls="-", color=INK):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc=fc, ec=ec, lw=1.4, ls=ls))
    if text:
        ax.text(x, y, text, ha="center", va="center", fontsize=size, weight=weight, color=color, linespacing=1.4)


def titled_box(ax, x, y, w, h, title, sub, fc=SURFACE, ec=MUTED, ls="-", dy=0.22, size_sub=8.4):
    box(ax, x, y, w, h, fc=fc, ec=ec, ls=ls)
    ax.text(x, y + dy, title, ha="center", va="center", fontsize=9.4, weight="bold")
    ax.text(x, y - dy, sub, ha="center", va="center", fontsize=size_sub, color=INK2, linespacing=1.35)


def arrow(ax, pts, label=None, label_xy=None, ha="center"):
    for a, b in zip(pts[:-2], pts[1:-1]):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=INK2, lw=1.3, solid_capstyle="butt")
    ax.add_patch(FancyArrowPatch(pts[-2], pts[-1], arrowstyle="-|>", mutation_scale=12, lw=1.3, color=INK2,
                                 shrinkA=0, shrinkB=0))
    if label:
        ax.text(*label_xy, label, ha=ha, va="center", fontsize=8.6, color=INK2, style="italic")


def legend_chip(ax, x, y, fc, ec, text, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y - 0.13), 0.42, 0.26, boxstyle="round,pad=0.01,rounding_size=0.06",
                                fc=fc, ec=ec, lw=1.2, ls=ls))
    ax.text(x + 0.55, y, text, va="center", fontsize=8.6, color=INK2)


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
fig, ax = canvas(10.5, 7.0, 12.4, 8.3, "Sample analysis at a glance (one notebook per patient)")
W, H = 2.75, 1.85
xs = [1.55, 4.65, 7.75, 10.85]
ys = [7.05, 4.6, 2.15]
sections = [
    ("1  Configuration", "experiment_config.yml,\nH5 from 02, variant filter", "EXPERIMENT, PATIENT_ID,\nwhitelist"),
    ("2  Protein", "normalise, UMAP, clusters,\nEspressoPro cell types", "cell type → broad label map"),
    ("3  Variant annotation", "Mission Bio annotation\nof all loaded variants", None),
    ("4  Germline variants", "HET markers in lymphoid\ncells, CNV/BAF audit", "AF cluster names,\nLOH cluster IDs"),
    ("5  Somatic variants", "candidates in non-lymphoid\ncells vs WT clone", "WT_LABEL, final\ndifferential_variants"),
    ("6  Clonal annotation", "clones on somatic variants,\nAF outlier filter", "clone names"),
    ("7  LOH pipeline", "LOH sub-clones from\nextreme germline AF blocks", "label_map (optional)"),
    ("8  CNV analysis", "double deletions, ploidy,\ninformative variants, BAF", "clone names, plot order,\nchr_arms_to_plot, loh_rules"),
    ("9  VAFs, clone sizes", "pseudobulk VAF; counts\nwith / without lymphoid", None),
    ("10  Final plots", "protein UMAP, variant\nand clone UMAPs", None),
    ("11  Export", "protein and DNA layers,\nAF/NGT, aberrancy tables", None),
    ("12  Multi-omic view", "clone × cell type\nheatmap", None),
]
centres = [(x, y) for y in ys for x in xs]
for (x, y), (t, s, man) in zip(centres, sections):
    box(ax, x, y, W, H)
    if man:
        ax.text(x, y + 0.6, t, ha="center", va="center", fontsize=9.6, weight="bold")
        ax.text(x, y + 0.13, s, ha="center", va="center", fontsize=8.3, color=INK2, linespacing=1.3)
        ax.add_patch(FancyBboxPatch((x - W / 2 + 0.12, y - H / 2 + 0.1), W - 0.24, 0.62,
                                    boxstyle="round,pad=0.01,rounding_size=0.08", fc=YELLOW_T, ec=YELLOW, lw=1.2))
        ax.text(x, y - H / 2 + 0.41, man, ha="center", va="center", fontsize=8.0, linespacing=1.25)
    else:
        ax.text(x, y + 0.3, t, ha="center", va="center", fontsize=9.6, weight="bold")
        ax.text(x, y - 0.22, s, ha="center", va="center", fontsize=8.3, color=INK2, linespacing=1.3)
for y in ys:
    for a, b in zip(xs[:-1], xs[1:]):
        arrow(ax, [(a + W / 2 + 0.02, y), (b - W / 2 - 0.02, y)])
for y1, y2 in zip(ys[:-1], ys[1:]):
    ym = (y1 - H / 2 + y2 + H / 2) / 2
    arrow(ax, [(xs[-1], y1 - H / 2), (xs[-1], ym), (xs[0], ym), (xs[0], y2 + H / 2)])
legend_chip(ax, 0.2, 0.55, YELLOW_T, YELLOW, "per-patient input: cells tagged \"Specific to this notebook\" (most also marked ⚠ INPUT REQUIRED)")
save(fig, "overview_sample_analysis.png")

# ── 2. Protein → cell types ──────────────────────────────────────────────────
fig, ax = canvas(10.5, 5.9, 12.4, 7.0, "Protein analysis: from antibody counts to cell-type labels")
W, H = 2.75, 1.15
r1, r2, r3 = 5.95, 3.85, 1.5
row1 = [
    ("Drop markers", "IgG1, IgG2a, IgG2b\n(+ hashtags if used)"),
    ("Remove empty cells", "cells with 0 protein\nreads"),
    ("Normalise + scale", "EspressoPro (Seurat CLR);\nmanual CLR if missing"),
    ("PCA → UMAP", "8 PCs; n_neighbors 50,\nmin_dist 0.1, spread 8"),
]
for x, (t, s) in zip(xs, row1):
    titled_box(ax, x, r1, W, H, t, s)
for a, b in zip(xs[:-1], xs[1:]):
    arrow(ax, [(a + W / 2 + 0.02, r1), (b - W / 2 - 0.02, r1)])
row2 = [
    ("Clusters", "graph-community on\nUMAP, k = 5", SURFACE, MUTED),
    ("EspressoPro", "Averaged.Detailed.\nCelltype per cell", SURFACE, MUTED),
    ("Broad labels", "cell_type_to_clusters\nmap (edit per patient)", YELLOW_T, YELLOW),
    ("Mixed clusters", "mark_mixed_clusters,\ntop label < 35 %", SURFACE, MUTED),
]
for x, (t, s, fc, ec) in zip(xs, row2):
    titled_box(ax, x, r2, W, H, t, s, fc=fc, ec=ec)
arrow(ax, [(xs[-1], r1 - H / 2), (xs[-1], (r1 + r2) / 2), (xs[0], (r1 + r2) / 2), (xs[0], r2 + H / 2)])
for a, b in zip(xs[:-1], xs[1:]):
    arrow(ax, [(a + W / 2 + 0.02, r2), (b - W / 2 - 0.02, r2)])
ax.text(xs[0], r2 - H / 2 - 0.22, "kept as 'Clusters'", ha="center", fontsize=8.3, color=INK2, style="italic")
# Final annotation and the two uses
titled_box(ax, 6.2, r3, 3.0, 1.0, "Final_annotation", "copied to label; UMAP\ncopied to the DNA assay", fc=BLUE_T, ec=BLUE)
arrow(ax, [(xs[-1], r2 - H / 2), (xs[-1], 2.75), (6.2, 2.75), (6.2, r3 + 0.52)])
titled_box(ax, 1.95, r3, 3.3, 1.0, "Lymphoid", "CD4_T, CD8_T, Other_T, B, Plasma\n→ germline reference cells", fc=GRAY_T)
titled_box(ax, 10.45, r3, 3.3, 1.0, "Non-lymphoid", "all other labels (NK included)\n→ somatic variant discovery", fc=GRAY_T)
arrow(ax, [(6.2 - 1.52, r3), (1.95 + 1.67, r3)])
arrow(ax, [(6.2 + 1.52, r3), (10.45 - 1.67, r3)])
ax.text(0.15, 0.3, "Also from this section: per-label homogeneity / silhouette plot (PRO_Utils) and the annotated "
        "Normalized_reads table pro_df_anno.", fontsize=8.4, color=INK2)
save(fig, "protein_annotation.png")

# ── 3. Variants → clones ─────────────────────────────────────────────────────
fig, ax = canvas(10.5, 9.4, 12.4, 11.0, "From variants to clones: germline markers, somatic variants and clone labels")
xl, xr, W, H = 3.0, 9.4, 4.6, 0.95
ax.text(xl, 10.55, "Germline (lymphoid cells)", ha="center", fontsize=10, weight="bold", color=INK2)
ax.text(xr, 10.55, "Somatic (non-lymphoid cells)", ha="center", fontsize=10, weight="bold", color=INK2)
gl = [
    ("Step 1  HET-like markers", "find_germline_variants QC + AF checks,\ncohort-calibrated rules", SURFACE, MUTED),
    ("Step 2  WT concordance", "genotype-defined WT cells from\nwhitelist clones; re-apply rules", SURFACE, MUTED),
    ("Step 3  CNV/BAF audit", "all-variant WT vs clone test; adds\nCNV-informative markers", SURFACE, MUTED),
    ("AF clustering", "KMeans on per-variant AF features,\nk chosen by silhouette", SURFACE, MUTED),
    ("Name the clusters", "AF_CLUSTER_RENAME_MAP,\nLOH_CLUSTER_IDS", YELLOW_T, YELLOW),
]
ygl = [9.75, 8.55, 7.35, 6.15, 4.95]
for y, (t, s, fc, ec) in zip(ygl, gl):
    titled_box(ax, xl, y, W, H, t, s, fc=fc, ec=ec, dy=0.2, size_sub=8.2)
for a, b in zip(ygl[:-1], ygl[1:]):
    arrow(ax, [(xl, a - H / 2), (xl, b + H / 2)])
sm = [
    ("Preliminary clones", "Mosaic VariantSubcloneTable on the\nwhitelist; drop missing/small/ADO/FP", SURFACE, MUTED),
    ("Set the reference", "WT_LABEL (\"WT\", another clone\nlabel, or None)", YELLOW_T, YELLOW),
    ("find_somatic_variants", "correlation with whitelist, WT vs\nnon-WT, pathogenic, minus germline", SURFACE, MUTED),
    ("Confirm the list", "differential_variants\n(heatmap + variant UMAPs)", YELLOW_T, YELLOW),
]
ysm = [9.75, 8.55, 7.35, 6.15]
for y, (t, s, fc, ec) in zip(ysm, sm):
    titled_box(ax, xr, y, W, H, t, s, fc=fc, ec=ec, dy=0.2, size_sub=8.2)
for a, b in zip(ysm[:-1], ysm[1:]):
    arrow(ax, [(xr, a - H / 2), (xr, b + H / 2)])
arrow(ax, [(xl + W / 2 + 0.02, 4.95), (6.2, 4.95), (6.2, 7.35), (xr - W / 2 - 0.02, 7.35)],
      "germline\nexcluded", (6.32, 6.15), ha="left")
# merge
ym = 3.55
titled_box(ax, 6.2, ym, 7.2, 0.95, "Informative variants = germline ∪ somatic",
           "Sample.dna restricted to these variants", fc=BLUE_T, ec=BLUE, dy=0.2, size_sub=8.2)
arrow(ax, [(xl, 4.95 - H / 2), (xl, ym + 0.49)])
arrow(ax, [(xr, 6.15 - H / 2), (xr, ym + 0.49)])
# clone row
yc, Wc = 1.65, 2.75
cl = [
    ("Clones", "VariantSubcloneTable on\ndifferential_variants", SURFACE, MUTED),
    ("Drop labels", "missing, small,\nADO, FP", ORANGE_T, ORANGE),
    ("AF outlier filter", "carrier clones: DP > 3,\nAF in mean ± 1.25 SD", ORANGE_T, ORANGE),
    ("Clone names", "rename_labels, e.g.\nTP53_R248L_Het", YELLOW_T, YELLOW),
]
for x, (t, s, fc, ec) in zip(xs, cl):
    titled_box(ax, x, yc, Wc, 1.1, t, s, fc=fc, ec=ec, dy=0.21, size_sub=8.2)
for a, b in zip(xs[:-1], xs[1:]):
    arrow(ax, [(a + Wc / 2 + 0.02, yc), (b - Wc / 2 - 0.02, yc)])
arrow(ax, [(6.2, ym - 0.48), (6.2, 2.62), (xs[0], 2.62), (xs[0], yc + 0.57)])
ax.text(0.15, 0.35, "Orange: cells removed. The outlier filter removes a cell if it fails DP or the SD band at any somatic "
        "variant its clone carries;\nWT and non-carrier clones are not filtered.", fontsize=8.4, color=INK2,
        linespacing=1.4)
save(fig, "variants_to_clones.png")

# ── 4. Outlier filter on simulated AF ────────────────────────────────────────
rng = np.random.default_rng(11)
clones = [("WT", 0.0, 2.0, 160, False), ("TP53_Het", 48.0, 7.0, 120, True), ("TP53_Hom", 94.0, 3.5, 70, True)]
fig, ax = plt.subplots(figsize=(10.5, 4.1))
for i, (name, mu, sd, n, carrier) in enumerate(clones):
    af = np.clip(rng.normal(mu, sd, n), 0, 100)
    if carrier:
        af[:6] = np.clip(rng.uniform(5, 30, 6) if mu < 60 else rng.uniform(40, 75, 6), 0, 100)
    x = i + rng.uniform(-0.28, 0.28, n)
    if not carrier:
        ax.scatter(x, af, s=10, color=MUTED, alpha=0.6, lw=0)
        ax.text(i, 12, "not a carrier clone:\nnot filtered", ha="center", va="bottom", fontsize=8.6, color=INK2)
        continue
    m, s = af.mean(), af.std(ddof=1)
    lo, hi = m - 1.25 * s, m + 1.25 * s
    keep = (af >= lo) & (af <= hi)
    ax.add_patch(plt.Rectangle((i - 0.38, lo), 0.76, hi - lo, color=BLUE_T, lw=0, zorder=0))
    ax.plot([i - 0.38, i + 0.38], [m, m], color=BLUE, lw=1.4)
    ax.scatter(x[keep], af[keep], s=10, color=BLUE, alpha=0.75, lw=0)
    ax.scatter(x[~keep], af[~keep], s=16, color=ORANGE, lw=0)
    ax.text(i + 0.42, hi, "mean + 1.25 SD", va="center", fontsize=8.2, color=INK2)
    ax.text(i + 0.42, lo, "mean − 1.25 SD", va="center", fontsize=8.2, color=INK2)
    ax.text(i, -14, f"{int((~keep).sum())} of {n} removed", ha="center", fontsize=8.6, color=ORANGE)
ax.set_xticks(range(len(clones)))
ax.set_xticklabels([c[0] for c in clones])
ax.set_xlim(-0.6, len(clones) - 0.1)
ax.set_ylim(-20, 112)
ax.set_yticks([0, 25, 50, 75, 100])
ax.set_ylabel("AF at one somatic variant (%)")
for s in ["right", "top"]:
    ax.spines[s].set_visible(False)
ax.scatter([], [], s=16, color=BLUE, label="kept")
ax.scatter([], [], s=16, color=ORANGE, label="removed (outside the band)")
ax.legend(frameon=False, loc="upper left", fontsize=8.8, bbox_to_anchor=(0.0, 1.0))
ax.set_title("Somatic AF outlier filter, per variant and carrier clone (simulated data; DP > 3 is checked first)",
             loc="left", fontsize=11, weight="bold", pad=10)
save(fig, "somatic_outlier_filter.png")

# ── 5. CNV / LOH ─────────────────────────────────────────────────────────────
fig, ax = canvas(10.5, 7.6, 12.4, 9.0, "CNV analysis: LOH sub-clones, double deletions, informative variants and ploidy")
W, H = 2.75, 1.15
ya, yb, yc2 = 8.1, 5.05, 2.75
rowa = [
    ("Group by arm", "germline markers →\narms (cytobands)", SURFACE, MUTED),
    ("LOH_Pipeline.run", "phases 0–5, sub-clone scan\n(phase 6), summary (7)", SURFACE, MUTED),
    ("Keep plotted cells", "drop missing, ADO,\nFP, small", ORANGE_T, ORANGE),
    ("label_map", "optional renaming of\nLOH labels", YELLOW_T, YELLOW),
]
for x, (t, s, fc, ec) in zip(xs, rowa):
    titled_box(ax, x, ya, W, H, t, s, fc=fc, ec=ec, size_sub=8.2)
for a, b in zip(xs[:-1], xs[1:]):
    arrow(ax, [(a + W / 2 + 0.02, ya), (b - W / 2 - 0.02, ya)])
# label vocabulary
yv = 6.6
ax.text(0.2, yv + 0.62, "Clone labels after LOH:", fontsize=8.8, weight="bold", va="center")
vocab = [
    (xs[0], "TP53_R248W_Hem", "base clone", BLUE_T, BLUE),
    (xs[1], "… LOH-C0 chr17p", "LOH sub-clone", BLUE_T, BLUE),
    (xs[2], "FP: … C0 chr17p", "failed LOH QC", ORANGE_T, ORANGE),
    (xs[3], "ADO, small", "dropped", ORANGE_T, ORANGE),
]
for x, lab, desc, fc, ec in vocab:
    box(ax, x, yv + 0.05, 2.25, 0.42, lab, fc=fc, ec=ec, size=8.2)
    ax.text(x, yv - 0.37, desc, ha="center", fontsize=8.0, color=INK2)
rowb = [
    ("Double deletions", "ploidy per amplicon of genes\nwith somatic variants", SURFACE, MUTED),
    ("Final clone labels", "post-LOH label_map;\ndesired_plotting_order", YELLOW_T, YELLOW),
    ("Informative variants", "clone vs WT per variant:\nBAF shift, depth ratio", SURFACE, MUTED),
    ("Expand panel", "relaxed H5 reload adds\nLOH-informative variants", BLUE_T, BLUE),
]
for x, (t, s, fc, ec) in zip(xs, rowb):
    titled_box(ax, x, yb, W, H, t, s, fc=fc, ec=ec, size_sub=8.2)
for a, b in zip(xs[:-1], xs[1:]):
    arrow(ax, [(a + W / 2 + 0.02, yb), (b - W / 2 - 0.02, yb)])
arrow(ax, [(xs[-1], ya - H / 2), (xs[-1], 7.4), (12.3, 7.4), (12.3, 5.95), (xs[0], 5.95), (xs[0], yb + H / 2)])
rowc = [
    ("Ploidy", "extract_cnv_data: WT cells =\ndiploid; variant amplicons", SURFACE, MUTED),
    ("Ploidy, all amplicons", "reload H5, same cells;\nfiles tagged _all_vars", SURFACE, MUTED),
    ("Integrated plots", "BAF + ploidy per clone:\ngenome, chr17p, chosen arms", YELLOW_T, YELLOW),
    ("CNA table", "loh_rules → Copy_Number_\nAlterations.csv", YELLOW_T, YELLOW),
]
for x, (t, s, fc, ec) in zip(xs, rowc):
    titled_box(ax, x, yc2, W, H, t, s, fc=fc, ec=ec, size_sub=8.2)
for a, b in zip(xs[:-1], xs[1:]):
    arrow(ax, [(a + W / 2 + 0.02, yc2), (b - W / 2 - 0.02, yc2)])
arrow(ax, [(xs[-1], yb - H / 2), (xs[-1], (yb + yc2) / 2), (xs[0], (yb + yc2) / 2), (xs[0], yc2 + H / 2)])
ax.text(0.15, 1.15,
        "The LOH step relabels cells: a clone whose germline HET variants become extreme (AF ≤ 10 or ≥ 90 %) over a\n"
        "contiguous block on an arm gets an ' LOH-C<n> <arm>' suffix. Ploidy is computed relative to the cells labelled\n"
        "\"WT\" (one notebook uses its TP53 Het clone instead). Yellow: values edited per patient.",
        fontsize=8.4, color=INK2, linespacing=1.45, va="center")
save(fig, "cnv_loh.png")
print("done")
