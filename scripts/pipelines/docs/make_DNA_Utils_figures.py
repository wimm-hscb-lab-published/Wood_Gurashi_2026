"""Draw the README schematics for DNA_Utils into ``DNA_Utils/docs/``.

Run from the pipelines folder with the `mosaic` conda environment::

    ~/miniconda3/envs/mosaic/bin/python docs/make_DNA_Utils_figures.py

Nothing here touches the Varsome cache, a Mosaic H5 file or the network: the flowcharts are
drawn by hand and the one data panel runs ``piecewise_elbow_threshold`` on simulated values.

Figures
-------
doublet_scoring.png            What ``flag_doublet_change_cells`` computes when called by
                               ``define_doublet_cutoff_per_donor`` on Mosaic AF (percent).
doublet_cutoff_reassign.png    Elbow cutoff + bootstrap CI (real function, simulated metric) and the
                               borderline reassignment of ``apply_doublet_calls_with_corr_reassign``.
low_quality_gates.png          The OR-ed gates of ``apply_low_quality_qc`` (incl. gate 1 =
                               ``flag_amplicon_loss_cells``).
find_germline_variants.png     Filters of the legacy AF-window germline finder.
af_cluster_rename.png          ``run_af_clustering_only`` + ``rename_af_clusters_and_update_filtered_germline``.
find_somatic_variants.png      Candidate sources and filters of ``find_somatic_variants``.
somatic_outlier_filter.png     Per-variant funnel of ``filter_somatic_variant_outliers``.
varsome_repair.png             Chunked annotation, repair-and-retry and ``repair_varsome_db``.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from _style import (AQUA, BLUE, BLUE_T, GRAY_T, INK, INK2, MUTED, ORANGE, ORANGE_T, PIPELINES,  # noqa: E402
                    YELLOW, YELLOW_T, arrow, box, canvas, node, plt, save)

sys.path.insert(0, PIPELINES)

import numpy as np  # noqa: E402

from DNA_Utils.qc_filters import piecewise_elbow_threshold  # noqa: E402

PKG = "DNA_Utils"
WHITE = "#ffffff"


def note(ax, x, y, text, size=8.4, va="bottom", ha="left"):
    ax.text(x, y, text, fontsize=size, color=INK2, va=va, ha=ha, linespacing=1.4, style="italic")


def head(ax, x, y, text, size=9.6):
    ax.text(x, y, text, fontsize=size, color=INK, weight="bold", va="center", ha="left")


def kept(ax, x, y, w, h, text, **kw):
    box(ax, x, y, w, h, text, fc=BLUE_T, ec=BLUE, **kw)


def flagged(ax, x, y, w, h, text, **kw):
    box(ax, x, y, w, h, text, fc=ORANGE_T, ec=ORANGE, **kw)


def neutral(ax, x, y, w, h, text, **kw):
    box(ax, x, y, w, h, text, fc=GRAY_T, **kw)


def decision(ax, x, y, w, h, text, **kw):
    box(ax, x, y, w, h, text, fc=WHITE, ec=INK2, weight="bold", **kw)


def manual(ax, x, y, w, h, text, **kw):
    box(ax, x, y, w, h, text, fc=YELLOW_T, ec=YELLOW, **kw)


# ── 1. Doublet scoring ───────────────────────────────────────────────────────────────────────────
def doublet_scoring():
    fig, ax = canvas(10.5, 7.6, 21, 15.2,
                     title="flag_doublet_change_cells, as called per donor by define_doublet_cutoff_per_donor")

    neutral(ax, 5.0, 13.9, 9.6, 1.1, "AF of one donor's cells x all variants\n(Mosaic AF layer: percent, 0-100)")
    neutral(ax, 5.0, 12.1, 9.6, 0.9, "consensus = per-variant median AF of the donor")
    arrow(ax, [(5.0, 13.35), (5.0, 12.55)])

    decision(ax, 5.0, 10.2, 9.6, 1.5,
             "informative = median in [0.10, 0.90]\n>= 30 informative variants?", size=9.2)
    arrow(ax, [(5.0, 11.65), (5.0, 10.95)])
    neutral(ax, 16.1, 10.2, 8.8, 1.5,
            "no -> all variants are informative\n(typical on percent AF: few loci have a\nmedian between 0.1 % and 0.9 %)",
            size=8.8)
    arrow(ax, [(9.8, 10.2), (11.7, 10.2)], label="no", label_xy=(10.75, 10.5))

    decision(ax, 5.0, 7.9, 9.6, 1.7,
             "discriminating mode?\n>= 2 donors with >= 10 cells AND\n>= 5 variants with |median - other donor| >= 0.20\n(also >= 5 after intersecting with informative)",
             size=8.6)
    arrow(ax, [(5.0, 9.45), (5.0, 8.75)], label="yes", label_xy=(5.25, 9.1), ha="left")
    arrow(ax, [(16.1, 9.45), (16.1, 8.9), (9.8, 8.9)])
    neutral(ax, 16.1, 7.3, 8.8, 1.2,
            "active = discriminating AND informative\nelse active = informative", size=8.8)
    arrow(ax, [(9.8, 7.5), (11.7, 7.5)])

    box(ax, 5.0, 5.55, 9.6, 1.4,
        "d = AF - consensus on active variants\ngain: d >= +0.30     loss: d <= -0.30\n(0.30 = 0.3 percentage points on percent AF)",
        fc=WHITE, ec=MUTED, size=8.8)
    arrow(ax, [(16.1, 6.7), (16.1, 5.55), (9.8, 5.55)])

    neutral(ax, 5.0, 3.75, 9.6, 1.1, "per cell: frac_gain, frac_loss, frac_changed\n(counts / number of active variants)")
    arrow(ax, [(5.0, 4.85), (5.0, 4.3)])

    kept(ax, 5.0, 1.55, 9.6, 1.9,
         "frac_changed_bidir = 2 x min(frac_gain, frac_loss)\n= the metric thresholded per donor\n(use_bidirectional=True, default)",
         weight="bold", size=9)
    arrow(ax, [(5.0, 3.2), (5.0, 2.5)])

    neutral(ax, 16.1, 3.75, 8.8, 2.4,
            "also returned (not used for the cutoff):\nfail_bidir = bidir >= its median AND\n   gain >= 0.01 AND loss >= 0.01\nlegacy fail = robust_z(frac_changed) +\n   robust_z(af_dilution) > median + 3.5 x 1.4826 x MAD",
            size=8.3, ha="left")
    arrow(ax, [(9.8, 3.75), (11.7, 3.75)])
    note(ax, 11.75, 0.65,
         "af_dilution = mean drop below the consensus at variants\nwith median >= 0.30 (fixed). All cut-offs above are 0-1\nvalues applied unchanged to percent AF (no rescaling).",
         size=8.3)
    return save(fig, PKG, "doublet_scoring.png")


# ── 2. Elbow cutoff + reassignment ───────────────────────────────────────────────────────────────
def _simulated_metric(seed=1):
    rng = np.random.default_rng(seed)
    singlets = rng.beta(1.6, 30.0, 900) * 0.6
    doublets = rng.uniform(0.12, 0.45, 70)
    return np.concatenate([singlets, doublets])


def doublet_cutoff_reassign():
    fc = _simulated_metric()
    thr, k, info = piecewise_elbow_threshold(fc, min_frac=0.01, max_frac=0.55, slope_ratio_min=3.0)
    rng = np.random.default_rng(42)
    boot = []
    for _ in range(200):
        t, _, _ = piecewise_elbow_threshold(fc[rng.integers(0, len(fc), len(fc))],
                                            min_frac=0.01, max_frac=0.55, slope_ratio_min=3.0)
        if np.isfinite(t):
            boot.append(t)
    lo, hi = np.quantile(boot, [0.025, 0.975])
    band = 0.02
    n_hi = int((fc >= thr + band).sum())
    n_mid = int(((fc >= thr - band) & (fc < thr + band)).sum())
    print(f"elbow thr={thr:.3f} k={k} ratio={info.get('ratio'):.1f} CI=({lo:.3f},{hi:.3f}) "
          f"high={n_hi} mid={n_mid} flagged={(fc >= thr).sum()}")

    fig = plt.figure(figsize=(10.5, 4.9))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.12], wspace=0.08)
    ax = fig.add_subplot(gs[0])
    y = np.sort(fc)[::-1]
    x = np.arange(len(y))
    ax.axhspan(thr - band, thr + band, color="#e4e2dc", lw=0, label=f"reassign band thr +/- {band}")
    ax.axhspan(lo, hi, color=ORANGE, alpha=0.30, lw=0, label="95 % bootstrap CI (200x, seed 42)")
    ax.plot(x, y, color=INK, lw=1.4, label="metric, sorted high -> low")
    ax.axhline(thr, color=ORANGE, lw=1.6, ls="--", label=f"elbow threshold = {thr:.3f}")
    ax.axvline(k, color=INK2, lw=0.9, ls=":", label=f"split k = {k}")
    ax.set_xlabel("cells (rank)")
    ax.set_ylabel("frac_changed_bidir")
    ax.set_title(f"piecewise_elbow_threshold on simulated values\n{len(fc)} cells; split range 1-55 % of cells, "
                 f"slope ratio >= 3", loc="left", fontsize=9.3)
    ax.legend(fontsize=7.8, frameon=False, loc="upper right")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    ax2 = fig.add_subplot(gs[1])
    ax2.set_xlim(0, 11.2)
    ax2.set_ylim(-0.9, 9.8)
    ax2.axis("off")
    ax2.set_title("apply_doublet_calls_with_corr_reassign (per donor)", loc="left", fontsize=9.6, weight="bold")
    decision(ax2, 5.6, 8.9, 10.6, 0.95, "metric vs threshold thr (band = 0.02)", size=9)
    flagged(ax2, 1.85, 6.9, 3.5, 1.5, "metric >= thr + band\nHighChange\n-> doublet", size=8.2)
    neutral(ax2, 5.6, 6.9, 3.5, 1.5, "in between\n(borderline)", size=8.6)
    kept(ax2, 9.35, 6.9, 3.5, 1.5, "metric < thr - band\nLowChange\n-> keeps label", size=8.2)
    for xx in (1.85, 5.6, 9.35):
        arrow(ax2, [(xx, 8.42), (xx, 7.65)])
    box(ax2, 5.6, 4.55, 10.6, 1.75,
        "Pearson r of the cell's AF with the donor median (corr_donor)\nand with the median of all 'Mixed' cells (corr_mixed),\n"
        "each over >= 40 shared finite variants",
        fc=WHITE, ec=MUTED, size=8.5)
    arrow(ax2, [(5.6, 6.15), (5.6, 5.43)])
    flagged(ax2, 2.85, 1.95, 5.3, 1.9,
            "Borderline->Mixed (doublet)\ncorr_mixed - corr_donor >= 0\nor only corr_mixed is finite", size=8.4)
    kept(ax2, 8.45, 1.95, 5.3, 1.9,
         "Borderline->Donor\notherwise (always when\ncorr_mixed is NaN / no Mixed cells)", size=8.4)
    arrow(ax2, [(3.6, 3.68), (3.6, 2.9)])
    arrow(ax2, [(7.6, 3.68), (7.6, 2.9)])
    ax2.text(0.3, -0.75, "Doublets get label 'Mixed' in the per-donor sub-assays only (write_back=True);\n"
             "the caller copies labels back to the Sample by barcode.",
             fontsize=8, color=INK2, style="italic", va="bottom")
    fig.suptitle("Doublet cutoff (define_doublet_cutoff_per_donor) and final calls - simulated metric",
                 x=0.01, ha="left", fontsize=11, weight="bold")
    fig.subplots_adjust(top=0.82, left=0.07, right=0.99, bottom=0.12)
    return save(fig, PKG, "doublet_cutoff_reassign.png")


# ── 3. Low-quality gates ─────────────────────────────────────────────────────────────────────────
def low_quality_gates():
    fig, ax = canvas(10.5, 8.6, 21, 17.2,
                     title="apply_low_quality_qc: a cell becomes Low_quality if any active gate fires (per donor)")

    neutral(ax, 7.6, 15.95, 14.8, 1.5,
            "Sample.dna AF + labels. Percent AF detected (99th percentile > 1.5) -> cut-offs below in %.\n"
            "dropout = fraction of ALL loci missing or AF <= 1 %   |   low-AF = fraction of donor-informative\n"
            "loci (donor median >= 10 %) missing or < 5 %",
            size=8.3, ha="left")

    gx, gw = 7.6, 14.8
    box(ax, gx, 13.25, gw, 2.75,
        "Gate 1 - flag_amplicon_loss_cells (always on; uses ITS defaults k_mad=5, q=0.995)\n"
        "donors >= 20 cells (Mixed / Low_quality skipped); AF divided by 100\n"
        "stable loci = median 0.10-0.90 AND per-variant MAD <= 35th pct of those MADs\n"
        "   (< 25 loci -> MAD <= 50th pct of all variants; < 3 loci -> donor skipped)\n"
        "score = z(median |AF - consensus|) + z(1 - r with consensus) + 0.7 z(missing fraction)\n"
        "flag score >= max(median + 5 x 1.4826 x MAD, 99.5th pct)  -> at most ~0.5 % per donor",
        fc=WHITE, ec=MUTED, size=8.2, ha="left")
    box(ax, gx, 9.95, gw, 2.3,
        "Gate 2 - global dropout (use_global_dropout_gate=True)\n"
        "thr = min(median + 3.75 x 1.4826 x MAD, 97.5th pct) per donor (pooled thr if donor < 20 cells)\n"
        "flag dropout > thr AND (dropout - thr) / SE >= 2,   SE = sqrt(thr (1 - thr) / n_loci)\n"
        "OR dropout >= max(0.50, donor 98th pct)   (depth_adaptive=True, abs_floor_quantile=0.98)",
        fc=WHITE, ec=MUTED, size=8.2, ha="left")
    box(ax, gx, 7.35, gw, 1.55,
        "Gate 3 - low AF at informative loci (OFF: use_low_af_gate=False)\n"
        "same rule as gate 2 on the low-AF fraction (donor >= 20 cells, >= 10 informative loci)",
        fc=GRAY_T, ec=MUTED, ls="--", size=8.2, ha="left")
    box(ax, gx, 5.15, gw, 1.55,
        "Gate 4 - combined moderate risk (use_combined_gate=True)\n"
        "dropout >= donor 95th pct AND low-AF >= donor 95th pct AND both effect sizes >= 2",
        fc=WHITE, ec=MUTED, size=8.2, ha="left")

    nx, ny = 16.3, 9.3
    node(ax, nx, ny, "any", r=0.42)
    for yy in (13.25, 9.95, 7.35, 5.15):
        arrow(ax, [(gx + gw / 2, yy), (15.5, yy), (nx - 0.42, ny)] if yy != 9.95 else
              [(gx + gw / 2, yy), (nx - 0.42, ny + 0.0)])
    box(ax, 19.0, ny, 3.6, 1.9, "localized-LOH\nrescue (OFF:\nprotect_localized\n_loh=False)",
        fc=GRAY_T, ec=MUTED, ls="--", size=8.0)
    arrow(ax, [(nx + 0.42, ny), (17.2, ny)])
    flagged(ax, 19.0, 5.6, 3.6, 2.4, "label ->\nLow_quality\n(Mixed never\nrelabelled)", weight="bold",
            size=8.6)
    arrow(ax, [(19.0, ny - 0.95), (19.0, 6.8)])
    arrow(ax, [(gx, 15.2), (gx, 14.62)])

    note(ax, 0.2, 1.75,
         "Written to sample_qc.dna.row_attrs: label, lowq_germline_fail, lowq_germline_score, lowq_stable_loci_n,\n"
         "loh_protected, chr7_loh_protected (same mask). Then a summary table and fast_af_heatmap (global; per subset if multiplexed).\n"
         "Preprocessing notebooks call it with k_mad=3.5 and abs_floor_quantile=0.95 (floor = max(0.50, donor 95th pct)).",
         size=8.2)
    return save(fig, PKG, "low_quality_gates.png")


# ── 4. find_germline_variants ────────────────────────────────────────────────────────────────────
def find_germline_variants_fig():
    fig, ax = canvas(10.5, 8.6, 21, 17.2,
                     title="find_germline_variants: legacy AF-window filter for germline HET variants")
    lx, lw = 6.6, 12.8
    neutral(ax, lx, 16.0, lw, 1.0, "Sample -> cells whose protein label is in keep_labels\n(ValueError if none) = Sample_T_subset",
            size=8.8)
    neutral(ax, lx, 14.2, lw, 1.4,
            "per variant, valid cells = NGT in {0, 1, 2} AND DP >= min_dp (5)\n"
            "-> N_WT, N_HET, N_HOM, fractions, call rate, mean / median AF of valid cells\n"
            "(DP filter off if min_dp <= 0 or no DP layer)", size=8.6)
    arrow(ax, [(lx, 15.5), (lx, 14.9)])

    gates = [
        (11.8, "(a) coverage", "valid cells >= 20, call rate >= 0.20,\nNGT-missing fraction <= 0.80, HOM fraction <= 0.25"),
        (9.6, "(b) AF window on mean valid AF (%)", "common / known-HET variant: 25-75\nother (novel) variant: 40-60"),
        (7.4, "(c) HET dominance", "N_HET >= 3, N_HET / N_HOM >= 3 (no HOM -> passes),\nN_HET / (N_HET + N_HOM) >= 0.75"),
        (5.2, "(d) not excluded", "not ClinVar pathogenic, not in a blacklist interval"),
    ]
    arrow(ax, [(lx, 13.5), (lx, 12.75)])
    for i, (yy, t, body) in enumerate(gates):
        box(ax, lx, yy, lw, 1.7, "", fc=WHITE, ec=MUTED)
        head(ax, lx - lw / 2 + 0.25, yy + 0.45, t, size=9.2)
        ax.text(lx - lw / 2 + 0.25, yy - 0.3, body, fontsize=8.5, va="center", ha="left", color=INK, linespacing=1.4)
        if i < len(gates) - 1:
            arrow(ax, [(lx, yy - 0.85), (lx, yy - 1.35)], label="AND", label_xy=(lx + 0.25, yy - 1.1), ha="left")
    kept(ax, lx, 2.3, lw, 2.0,
         "pass_final_germline_filter -> filtered_germline_variants (list)\n"
         "Sample_T_subset.dna_subset = passing variants (None if none)\n"
         "Sample_T_subset.germline_qc = per-variant QC table (sorted)\n"
         "return_qc=True also returns the QC table", size=8.6)
    arrow(ax, [(lx, 4.35), (lx, 3.3)])

    rx, rw = 17.25, 7.3
    manual(ax, rx, 15.0, rw, 2.2,
           "dbSNP / gnomAD common ID files\n(union; 'both' = intersection)\n+ known_het_variants (whitelist)\n"
           "+ known_het_variant_paths", size=8.4)
    neutral(ax, rx, 11.8, rw, 2.1,
            "is_common_resource\nknown-HET IDs are added to the\ncommon set (treat_known_het_as_common);\n"
            "no known-HET input -> known = common", size=8.2)
    arrow(ax, [(rx, 13.9), (rx, 12.85)])
    arrow(ax, [(rx, 10.75), (rx, 9.6), (lx + lw / 2, 9.6)])
    manual(ax, rx, 7.0, rw, 1.9,
           "annotation_df['ClinVar']: 'pathogenic'\nand not 'benign'\nblacklist BED / CSV intervals", size=8.4)
    arrow(ax, [(rx, 6.05), (rx, 5.2), (lx + lw / 2, 5.2)])
    note(ax, 13.6, 1.0, "Default resource paths are hard-coded cluster paths\n(/project/meadlab/...); scalar row_attrs of\nSample.dna / Sample.protein are dropped in place.",
         size=8.2)
    return save(fig, PKG, "find_germline_variants.png")


# ── 5. AF clustering + rename ────────────────────────────────────────────────────────────────────
def af_cluster_rename():
    fig, ax = canvas(10.5, 10.2, 21, 20.4,
                     title="run_af_clustering_only -> rename_af_clusters_and_update_filtered_germline")
    head(ax, 0.2, 19.4, "1. run_af_clustering_only (unsupervised)", size=9.6)
    neutral(ax, 3.3, 17.7, 6.2, 2.3,
            "df_AF_all (cells x variants\n+ label)\nvariants = germline list +\nother list (original list ->\nother_vars -> all remaining)",
            size=8.2)
    neutral(ax, 10.45, 17.7, 7.3, 2.3,
            "features per variant: valid fraction,\nmedian / mean / IQR / SD AF, fractions\n25-75, 35-65, <= 5, >= 95 %, fixed;\n"
            "median per label, label min / max / IQR,\nfraction labels 25-75; |WT - non-WT|", size=8.2)
    kept(ax, 17.6, 17.7, 6.0, 2.3,
         "median-impute, StandardScaler\nKMeans k = 2..min(20, n-1),\nn_init 50, seed 7\nbest silhouette -> af_cluster",
         size=8.2)
    arrow(ax, [(6.4, 17.7), (6.8, 17.7)])
    arrow(ax, [(14.1, 17.7), (14.6, 17.7)])

    head(ax, 0.2, 15.5, "2. rename_af_clusters_and_update_filtered_germline (per variant)", size=9.6)
    manual(ax, 5.2, 14.0, 10.0, 1.6,
           "user: af_cluster_rename_map (default 0, 1, 2 -> germline_het;\n3 -> germline_wt; 4 -> germline_hom; other -> cluster_<id>)\n"
           "and loh_cluster_ids (raw clusters to treat as LOH)", size=8.2)
    arrow(ax, [(17.6, 16.55), (17.6, 15.3), (10.2, 15.3), (10.2, 14.4)])

    steps = [
        (11.9, "hard exclusion: blacklist / pathogenic flags found in variant_af_cluster_df, df,\n"
               "all_variant_germline_df or df_review (by column name or text)"),
        (10.0, "independent baseline evidence: calibrated / clean / rescued / concordant seed,\n"
               "donor_genotype het, or validated broad class (input list alone does NOT count)"),
        (8.1, "LOH marker: raw cluster in loh_cluster_ids AND simple SNV AND not excluded AND\n"
              "(independent evidence OR in the input germline list)"),
        (6.2, "preserve: (baseline evidence AND simple SNV) OR LOH marker -> germline_het,\n"
              "whatever the raw cluster (preserve_baseline_germline_evidence=True)"),
    ]
    sx, sw = 7.0, 13.6
    arrow(ax, [(5.2, 13.2), (5.2, 12.65)])
    for i, (yy, t) in enumerate(steps):
        box(ax, sx, yy, sw, 1.45, t, fc=WHITE, ec=MUTED, size=8.3, ha="left")
        if i < len(steps) - 1:
            arrow(ax, [(sx, yy - 0.73), (sx, yy - 1.17)])
    flagged(ax, 17.6, 11.9, 5.4, 1.45, "excluded_blacklisted_\npathogenic (always wins)", size=8.3)
    arrow(ax, [(sx + sw / 2, 11.9), (14.9, 11.9)])

    box(ax, sx, 3.45, sw, 3.3,
        "poor-QC discard if any (candidate = germline_het, input-list or preliminary-LOH variant):\n"
        " - candidate without baseline evidence and not an LOH marker\n"
        " - candidate that is not a simple A/C/G/T SNV\n"
        " - systematic skew: all label medians <= 30 or all >= 70, label IQR <= 10, range <= 15,\n"
        "   >= 2 labels, < 2/3 labels at 25-75 % (raw LOH clusters protected)\n"
        " - same-amplicon block of >= 2 rescued-not-clean SNVs, same raw cluster, all label\n"
        "   medians <= 35 or >= 65, each label range <= 8, not LOH",
        fc=WHITE, ec=MUTED, size=8.0, ha="left")
    arrow(ax, [(sx, 5.47), (sx, 5.1)])
    flagged(ax, 17.6, 3.45, 5.4, 1.45, "discarded_poor_qc", size=8.6)
    arrow(ax, [(sx + sw / 2, 3.45), (14.9, 3.45)])
    kept(ax, sx, 0.75, sw, 1.2,
         "filtered_germline_variants = germline_het AND (baseline evidence OR LOH marker)\n"
         "AND simple SNV AND not excluded; LOH markers also returned as loh_variants",
         size=8.4)
    arrow(ax, [(sx, 1.8), (sx, 1.35)])
    return save(fig, PKG, "af_cluster_rename.png")


# ── 6. find_somatic_variants ─────────────────────────────────────────────────────────────────────
def find_somatic_variants_fig():
    fig, ax = canvas(10.5, 8.8, 21, 17.6,
                     title="find_somatic_variants: candidate sources (union) and filters - carrier = AF > 10 %")
    neutral(ax, 10.5, 16.5, 20.4, 1.0,
            "df: first column = cell label, other columns = variant AF (%); WT cells = label == wt_label ('WT')",
            size=8.8)
    srcs = [
        (2.15, "(1) Spearman", "each detected\nwhitelist target vs\nevery other variant\n(>= 3 shared cells)\nFDR-BH adj p < 0.05"),
        (6.3, "(2) Mann-Whitney", "WT vs non-WT\n(>= 3 cells each)\nadj p < 0.05 AND\n|carrier diff| >= 0.10\nAND non-WT carrier\nfrac >= 0.15"),
        (10.45, "(2b) within-WT", ">= 10 WT cells; WT carrier\nfrac >= 0.01 or >= 5 cells,\n<= 0.50, >= 3 carriers;\nnon-WT frac <= 0.05;\nFisher (WT >) adj p < 0.05\nor zero-in-mutant rescue"),
        (14.6, "detected whitelist", "whitelist IDs matched\nto df columns on the\nchr:pos:ref/alt part"),
        (18.75, "pathogenic", "annotation value has\n'pathogenic' or 'likely\npathogenic' (not 'not /\nno / non pathogenic')\nalways_keep_pathogenic"),
    ]
    for xx, t, body in srcs:
        box(ax, xx, 12.9, 3.95, 3.9, "", fc=WHITE, ec=MUTED)
        ax.text(xx, 14.45, t, ha="center", va="center", fontsize=8.9, weight="bold")
        ax.text(xx, 12.55, body, ha="center", va="center", fontsize=7.8, linespacing=1.35)
        arrow(ax, [(xx, 15.95), (xx, 14.9)])
        arrow(ax, [(xx, 10.95), (xx, 10.2), (10.5, 10.2), (10.5, 9.75)] if xx != 10.45 else [(xx, 10.95), (xx, 9.75)])
    node(ax, 10.5, 9.4, "U", r=0.35)
    steps = [
        (8.0, "sort by chromosome number (positions not sorted numerically)", GRAY_T, MUTED),
        (6.6, "get_real_somatic_variants: drop germline-list IDs, IDs not in Sample.dna, intronic\n"
              "(Consequence ~ 'intron', else Protein blank / 'p.?')", WHITE, MUTED),
        (5.0, "re-add protected variants removed above: detected whitelist (whitelist_bypass_germline_filter=True);\n"
              "pathogenic only if pathogenic_bypass_germline_filter=True", GRAY_T, MUTED),
        (3.4, "annotation_df given: drop Function ~ 'intron' or Coding impact ~ 'synonymous' (protected kept)", WHITE, MUTED),
    ]
    arrow(ax, [(10.5, 9.05), (10.5, 8.45)])
    for i, (yy, t, fc, ec) in enumerate(steps):
        box(ax, 10.5, yy, 18.0, 1.05 if "\n" not in t else 1.3, t, fc=fc, ec=ec, size=8.5)
        if i < len(steps) - 1:
            nxt = steps[i + 1][0]
            h1 = (1.05 if "\n" not in t else 1.3) / 2
            h2 = (1.05 if "\n" not in steps[i + 1][1] else 1.3) / 2
            arrow(ax, [(10.5, yy - h1), (10.5, nxt + h2)])
    kept(ax, 10.5, 1.3, 18.0, 1.3,
         "returns (differential_variants = exact df column names, significant Spearman table,\n"
         "significant Mann-Whitney table); the within-WT table is not returned", weight="bold", size=8.6)
    arrow(ax, [(10.5, 2.87), (10.5, 1.95)])
    return save(fig, PKG, "find_somatic_variants.png")


# ── 7. filter_somatic_variant_outliers ───────────────────────────────────────────────────────────
def somatic_outlier_filter():
    fig, ax = canvas(10.5, 6.6, 21, 13.2,
                     title="filter_somatic_variant_outliers: per variant, carrier clones only")
    manual(ax, 3.4, 11.6, 6.4, 1.6,
           "clone_variant_calls\n(clone x variant genotype\ntable, e.g. clones_to_dataframe)", size=8.3)
    neutral(ax, 10.5, 11.6, 6.8, 1.6, "df_AF (cells x variants + label)\nSample.dna DP layer\nrows aligned to DNA barcodes",
            size=8.3)
    neutral(ax, 17.6, 11.6, 6.4, 1.6, "differential_variants\n(processed if in df_AF\nand in the DP layer)", size=8.3)

    lx, lw = 7.0, 13.4
    box(ax, lx, 9.0, lw, 1.9,
        "eligible clones: match the variant on its chr:pos:ref/alt key; clone columns that are\n"
        "active labels in Sample.dna; call starts with Het / Hom / Hemi (or 0/1, 1/0, 1/1)\n"
        "-> eligible cells = union of those clones' cells (WT calls never eligible)",
        fc=WHITE, ec=MUTED, size=8.3, ha="left")
    arrow(ax, [(3.4, 10.8), (3.4, 9.95)])
    arrow(ax, [(10.5, 10.8), (10.5, 9.95)])
    arrow(ax, [(17.6, 10.8), (17.6, 9.0), (lx + lw / 2, 9.0)])
    box(ax, lx, 6.6, lw, 1.1, "DP > dp_threshold (default 5; notebooks use 3); NaN DP fails", fc=WHITE, ec=MUTED,
        size=8.6)
    arrow(ax, [(lx, 8.05), (lx, 7.15)], label="eligible cells", label_xy=(lx + 0.25, 7.6), ha="left")
    box(ax, lx, 4.5, lw, 1.5,
        "per clone: keep mean - 1.25 SD <= AF <= mean + 1.25 SD (SD ddof=1)\n"
        "clones with < 3 cells or SD = 0 are kept whole", fc=WHITE, ec=MUTED, size=8.6)
    arrow(ax, [(lx, 6.05), (lx, 5.25)])
    kept(ax, lx, 2.2, lw, 1.6,
         "kept_barcodes[variant] + removed_barcodes_by_variant[variant]\n(removed = eligible - kept, incl. DP failures)",
         size=8.6)
    arrow(ax, [(lx, 3.75), (lx, 3.0)])
    flagged(ax, 17.6, 6.6, 6.4, 1.1, "0 eligible cells -> skipped", size=8.6)
    arrow(ax, [(lx + lw / 2, 7.9), (17.6, 7.9), (17.6, 7.15)])
    neutral(ax, 17.6, 3.4, 6.4, 3.6,
            "files in outdir: jitter plots\nbefore / after SD filter\n(PNG + PDF), kept-barcode\nTXT, audit CSV per variant,\n"
            "combined audit + summary\nCSV; plot jitter seeded\nwith random_seed (0)", size=8.2)
    arrow(ax, [(lx + lw / 2, 4.5), (14.4, 4.5)])
    return save(fig, PKG, "somatic_outlier_filter.png")


# ── 8. Varsome cache repair ──────────────────────────────────────────────────────────────────────
def varsome_repair():
    fig, ax = canvas(10.5, 8.2, 21, 16.4,
                     title="Varsome cache: chunked annotation with repair-and-retry, and repair_varsome_db")
    head(ax, 0.2, 15.6, "annotate_assay_in_chunks", size=9.6)
    neutral(ax, 3.4, 13.9, 6.4, 1.6, "variant ids of the assay\nsplit into chunks of 1000", size=8.6)
    box(ax, 10.6, 13.9, 6.6, 2.4,
        "per chunk: up to 3 tries\n(sleep 5 s x attempt between)\n-> annotate_with_repair", fc=WHITE, ec=MUTED, size=8.6)
    arrow(ax, [(6.6, 13.9), (7.3, 13.9)])
    flagged(ax, 17.6, 15.0, 6.2, 1.2, "3 tries failed -> failed list\n(start, stop, error)", size=8.3)
    kept(ax, 17.6, 12.6, 6.2, 1.9, "concat, drop duplicate ids,\nreindex to assay order (NaN\ngaps), add col_attrs", size=8.3)
    arrow(ax, [(13.9, 14.6), (14.5, 15.0)])
    arrow(ax, [(13.9, 13.3), (14.5, 12.8)])

    head(ax, 0.2, 11.0, "annotate_with_repair (max_attempts=2)", size=9.6)
    box(ax, 4.6, 9.4, 8.8, 1.6, "with varsome_db_lock (flock on <cache>.lock,\nwait up to 600 s): assay.get_annotations()",
        fc=WHITE, ec=MUTED, size=8.6)
    decision(ax, 14.4, 9.4, 8.4, 1.6, "ParserError, or ValueError\n\"invalid literal for int()\" with '.'?", size=8.6)
    arrow(ax, [(9.0, 9.4), (10.2, 9.4)], label="error", label_xy=(9.6, 9.7))
    note(ax, 9.6, 7.25, "no / last attempt -> raise", size=8.4)
    ax.text(14.65, 7.4, "yes: repair_varsome_db()\n(unlocked), then retry", fontsize=8.6, color=INK2, style="italic",
            va="center")

    head(ax, 0.2, 6.5, "repair_varsome_db (default cache = varsome_db_path())", size=9.6)
    rs = [
        (2.4, "csv.reader:\nclassify each row"),
        (6.6, "normalise Position\n'123.0' -> '123'"),
        (10.8, "dedupe (Genome,\nVariant ID): keep the\nmore populated row"),
        (15.0, "backup .<stamp>.bak,\nwrite .tmp, os.replace"),
        (19.0, "re-parse: 16 data\ncols, row count,\nPosition not object"),
    ]
    for i, (xx, t) in enumerate(rs):
        box(ax, xx, 4.6, 3.75 if i < 4 else 3.6, 1.9, t, fc=WHITE if i < 4 else BLUE_T, ec=MUTED if i < 4 else BLUE,
            size=8.2)
        if i < len(rs) - 1:
            arrow(ax, [(xx + 1.9, 4.6), (rs[i + 1][0] - (1.9 if i + 1 < 4 else 1.82), 4.6)])
    arrow(ax, [(14.4, 8.6), (14.4, 6.2), (2.4, 6.2), (2.4, 5.55)])
    flagged(ax, 2.4, 1.6, 4.2, 2.2, "dropped: field_count(n),\nrepeat_header,\nbad_genome,\nbad_variant_id", size=8.2)
    arrow(ax, [(2.4, 3.65), (2.4, 2.7)])
    neutral(ax, 12.0, 1.6, 7.4, 1.6, "dry_run -> report only; nothing changed and\ncanonical header -> not rewritten", size=8.2)
    arrow(ax, [(10.8, 3.65), (10.8, 2.4)])
    flagged(ax, 19.0, 1.6, 3.6, 1.6, "AssertionError:\nrestore backup", size=8.2)
    arrow(ax, [(19.0, 3.65), (19.0, 2.4)])
    return save(fig, PKG, "varsome_repair.png")


FIGURES = [doublet_scoring, doublet_cutoff_reassign, low_quality_gates, find_germline_variants_fig,
           af_cluster_rename, find_somatic_variants_fig, somatic_outlier_filter, varsome_repair]

if __name__ == "__main__":
    only = set(sys.argv[1:])
    for f in FIGURES:
        if not only or f.__name__ in only:
            print(f())
