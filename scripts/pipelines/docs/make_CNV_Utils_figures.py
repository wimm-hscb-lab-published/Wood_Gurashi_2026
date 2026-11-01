"""Draw the README schematics for CNV_Utils into ``CNV_Utils/docs/``.

Run from the pipelines folder with the `mosaic` conda environment::

    ~/miniconda3/envs/mosaic/bin/python docs/make_CNV_Utils_figures.py

Figures
-------
informativeness_row_paths.png     Per (variant, clone) decision paths of ``test_variant_informativeness``.
informativeness_arm_passes.png    Per-arm corroboration / rescue passes of ``test_variant_informativeness``.
extract_cnv_flow.png              Data flow of ``extract_cnv_data`` and ``get_cnv_info``.
ploidy_region_pipeline.png        The 12 stages of ``prepare_ploidy_region_plot_inputs``.
integrated_plot_layout.png        Panel layout of ``plot_integrated_cnv_baf_with_significance`` (simulated values).

All thresholds shown are the function defaults.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from _style import (BLUE, BLUE_T, GRAY_T, GRID, INK, INK2, MUTED, ORANGE, ORANGE_T, SURFACE,  # noqa: E402
                    YELLOW, YELLOW_T, arrow, box, canvas, note, plt, save)

import numpy as np  # noqa: E402

PKG = "CNV_Utils"
WHITE = "#ffffff"


# ── 1. test_variant_informativeness: per-row decision paths ─────────────────────────────────────
def informativeness_row_paths():
    fig, ax = canvas(10.5, 9.4, 21, 18.8,
                     title="test_variant_informativeness, stage 1: one row per (variant, non-WT clone)")

    box(ax, 10.5, 17.6, 13.0, 1.1,
        "Clone cells vs WT/reference cells (exact wt_labels, default ['WT'])\n"
        "AF in percent (0-100); fractions below are AF/100 or read fractions", fc=GRAY_T, size=9)

    box(ax, 5.0, 15.3, 9.0, 1.7,
        "Plot-aligned aggregates (ALL clone cells)\nmean ALT, mean REF, mean depth, total reads,\n"
        "pooled minor fraction, # cells with depth >= 5", fc=SURFACE, size=8.7)
    box(ax, 15.6, 15.3, 9.6, 1.7,
        "Statistical aggregates (cells with ALT+REF >= 5)\nread-weighted AF = 100*sum(ALT)/sum(ALT+REF),\n"
        "AF IQR, Wilcoxon rank-sum p, depth ratio clone/WT", fc=SURFACE, size=8.7)
    arrow(ax, [(7.0, 17.05), (5.0, 16.15)])
    arrow(ax, [(14.0, 17.05), (15.6, 16.15)])

    # Column A: direct near-zero
    box(ax, 5.0, 12.7, 9.0, 1.5,
        "Near-zero pattern?\nmin(mean ALT, mean REF) <= 2.0 reads\nAND pooled minor fraction <= 0.10",
        fc=WHITE, weight="bold", size=8.7)
    arrow(ax, [(5.0, 14.45), (5.0, 13.45)])
    box(ax, 5.0, 10.3, 9.0, 1.7,
        "Reference OK?\ngermline_variants given: variant in the whitelist;\n"
        "else WT clean het: WT minor >= 0.25, WT AF 30-70", fc=WHITE, size=8.5)
    arrow(ax, [(5.0, 11.95), (5.0, 11.15)], label="yes", label_xy=(5.3, 11.55), ha="left")
    box(ax, 5.0, 7.8, 9.0, 1.5,
        "High-confidence depth?\nmean depth >= 15, total reads >= 100,\n>= 4 cells with depth >= 5",
        fc=WHITE, weight="bold", size=8.7)
    arrow(ax, [(5.0, 9.45), (5.0, 8.55)], label="yes", label_xy=(5.3, 9.0), ha="left")

    box(ax, 2.4, 4.9, 4.4, 1.7, "CALL\ndirect near-zero\n(tier direct_plot_aligned_\nnear_zero_ALT_or_REF)",
        fc=BLUE_T, ec=BLUE, size=8.2)
    arrow(ax, [(2.4, 7.05), (2.4, 5.75)], label="yes", label_xy=(2.65, 6.4), ha="left")

    box(ax, 7.75, 4.9, 4.9, 2.5,
        "Low-depth candidate\ndepth >= 5, reads >= 20,\n>= 3 cells; binomial p <= 1e-3\n"
        "vs WT allele fraction;\n95% upper bound <= 0.15", fc=GRAY_T, size=7.9)
    arrow(ax, [(7.6, 7.05), (7.6, 6.15)], label="no", label_xy=(7.85, 6.6), ha="left")
    box(ax, 7.75, 2.35, 4.9, 1.4, "not called yet: needs\nsame-arm anchors (stage 2a)", fc=GRAY_T, size=8.0, ls="--")
    arrow(ax, [(7.75, 3.65), (7.75, 3.05)])

    # Column B: statistical fallback
    box(ax, 14.0, 9.4, 6.8, 5.2,
        "Strict statistical fallback\n(only if not a direct call)\n\n"
        ">= 5 clone and >= 5 WT cells\n(min_cells, depth >= 5)\n"
        "Wilcoxon p < 0.01\n|clone AF - WT AF| >= 10 points\n|clone AF - 50| > 20 points\n"
        "clone AF IQR <= 0.30\npooled minor fraction <= 0.25\n>= 50% of reads in cells\nwith minor fraction < 0.10",
        fc=WHITE, size=8.5)
    arrow(ax, [(14.0, 14.45), (14.0, 12.05)])
    box(ax, 14.0, 4.9, 6.8, 1.4, "CALL\nstatistical_fallback", fc=BLUE_T, ec=BLUE, size=8.7)
    arrow(ax, [(14.0, 6.8), (14.0, 5.6)], label="all pass", label_xy=(14.25, 6.2), ha="left")
    note(ax, 10.75, 2.9, "no reference (WT clean-het)\ncheck on this path", size=8.0)

    # Column C: optional
    box(ax, 19.25, 9.4, 3.1, 5.2,
        "Optional,\noff by default\n\npooled-read\nrescue\n(enable_\ndropout_rescue)\n\narm_rescue\ncandidate\n(enable_arm_\nrescue)",
        fc=GRAY_T, size=7.9, ls="--")
    arrow(ax, [(18.6, 14.45), (18.9, 12.05)])

    box(ax, 17.5, 1.3, 6.6, 1.9,
        "Called rows: event_type from depth ratio\n< 0.7 DELETION, > 1.3 AMPLIFICATION,\n"
        "else cnLOH (NaN ratio: UNKNOWN)", fc=GRAY_T, size=8.3)
    box(ax, 3.2, 1.0, 6.0, 1.3, "No path passes: not_informative;\nclassification_reason = failing gate",
        fc=ORANGE_T, ec=ORANGE, size=8.2)
    return save(fig, PKG, "informativeness_row_paths.png")


# ── 2. test_variant_informativeness: per-arm passes ─────────────────────────────────────────────
def informativeness_arm_passes():
    fig, ax = canvas(10.5, 9.6, 21, 19.2,
                     title="test_variant_informativeness, stage 2: sequential passes over each arm's table")
    rows = [
        (17.6, 1.3, "Stage-1 table\n(one arm, all clones)", GRAY_T, None,
         "Base calls = direct near-zero OR statistical fallback OR pooled-read rescue"),
        (14.9, 2.6, "(a) Low-depth\ncorroboration", WHITE, BLUE_T,
         "Per clone: anchors = base calls; need >= 2 anchors from >= 2 amplicons other than the\n"
         "candidate's. Low-depth candidate promoted when |its pooled minor - anchor median| <= 0.15\n"
         "and |its |ALT frac - 0.5| - anchor median| <= 0.20.  Tier low_depth_near_zero_corroborated"),
        (11.5, 2.9, "(b) LOH-arm\npattern rescue", WHITE, BLUE_T,
         "Anchors = base calls with WT clean het, shift >= 0.08, mean depth >= 15, depth ratio <= 1.3;\n"
         ">= 2 anchors from >= 2 other amplicons. Uncalled row promoted if eligible (WT clean het,\n"
         "shift >= 0.08, IQR <= 0.45, depth >= 15, reads >= 30, minor <= 0.30, ratio <= 1.3) and similar\n"
         "to anchor medians (minor <= 0.18, imbalance <= 0.15, ratio <= 0.35).  Tier LOH_arm_pattern_rescue"),
        (8.3, 2.1, "(c) Generic arm\nrescue (off)", GRAY_T, None,
         "enable_arm_rescue=True: candidates (p < 0.10, shift >= 0.15, IQR <= 0.50) not yet called\n"
         "are all promoted when they span >= 2 distinct amplicons.  Tier arm_rescue_partial_shift"),
        (5.2, 2.6, "(d) Cross-Hem\nanti-phase rescue", WHITE, BLUE_T,
         "Clones whose label contains 'Hem' (case-insensitive), per variant. Uncalled Hem clone with\n"
         "reference OK, p < 0.01, AF IQR <= 0.30, shift >= 0.10, allele dropout, and another Hem clone\n"
         "already called with opposite-signed delta AF (shift >= 0.10).  Tier Hem_antiphase_rescue"),
        (1.6, 2.2, "Output per arm", BLUE_T, None,
         "informative = OR of the 7 path flags; event_type filled for calls (depth ratio rule).\n"
         "Written to output_dir/wilcoxon_results_<arm>_enhanced.csv; returned as {arm: DataFrame}"),
    ]
    for y, h, label, fc, _, detail in rows:
        ec = BLUE if fc == BLUE_T else MUTED
        ls = "--" if "off" in label else "-"
        box(ax, 2.6, y, 4.6, h, label, fc=fc, ec=ec, weight="bold", size=8.8, ls=ls)
        box(ax, 12.85, y, 15.5, h, detail, fc=SURFACE, size=8.1, ha="left", ls=ls)
        ax.plot([4.9, 5.1], [y, y], color=MUTED, lw=1.0)
    for (y1, h1, *_), (y2, h2, *_) in zip(rows[:-1], rows[1:]):
        arrow(ax, [(2.6, y1 - h1 / 2), (2.6, y2 + h2 / 2)])
    note(ax, 0.3, 19.0 - 0.15, "(a) and (b) are skipped when no variant_to_amplicon key matches df_AF (default settings).",
         size=8.2)
    return save(fig, PKG, "informativeness_arm_passes.png")


# ── 3. extract_cnv_data / get_cnv_info ──────────────────────────────────────────────────────────
def extract_cnv_flow():
    fig, ax = canvas(10.5, 9.6, 21, 19.2, title="extract_cnv_data (mutates Sample) and get_cnv_info")

    box(ax, 10.5, 18.0, 17.0, 1.1,
        "OUT_DIR given? else {wd}/results/per_sample/{EXPERIMENT}/{PATIENT_ID}/dna_cnv/clonotypes/ploidy\n"
        "(ValueError if neither); directory always created", fc=GRAY_T, size=8.5)
    box(ax, 10.5, 16.2, 13.5, 1.0,
        "optional: restrict Sample.dna to barcodes of sample_filter_source", fc=YELLOW_T, ec=YELLOW, size=8.7)
    arrow(ax, [(10.5, 17.45), (10.5, 16.7)])

    # Lane A
    box(ax, 5.0, 14.2, 9.0, 0.9, "ploidy_before_sync=True (default)", fc=WHITE, weight="bold", size=8.8)
    box(ax, 16.0, 14.2, 9.0, 0.9, "ploidy_before_sync=False (all-variant)", fc=WHITE, weight="bold", size=8.8)
    arrow(ax, [(8.0, 15.7), (5.0, 14.65)])
    arrow(ax, [(13.0, 15.7), (16.0, 14.65)])

    lane_a = ["1  diploid cells: DNA label_col == diploid_label\n   (default 'WT'; ValueError if none)",
              "2  cnv.compute_ploidy(diploid_cells,\n   normalized_counts='Normalized_reads')",
              "3  sync DNA/CNV to shared barcodes (DNA order);\n   CNV amplicons = DNA variant amplicons\n   (subset_cnv_to_variant_amplicons=True)",
              "4  transfer labels by barcode\n   (label_source_sample -> DNA, CNV; else DNA -> CNV)"]
    lane_b = ["1  sync DNA/CNV to shared barcodes\n   (same optional amplicon subset)",
              "2  transfer labels by barcode",
              "3  diploid cells + cnv.compute_ploidy",
              "4  re-sync, then transfer labels DNA -> CNV"]
    ys = [12.7, 11.05, 9.4, 7.75]
    for y, t in zip(ys, lane_a):
        box(ax, 5.0, y, 9.0, 1.4, t, fc=SURFACE, size=8.0, ha="left")
    for y, t in zip(ys, lane_b):
        box(ax, 16.0, y, 9.0, 1.4, t, fc=SURFACE, size=8.0, ha="left")
    for lane_x in (5.0, 16.0):
        arrow(ax, [(lane_x, 13.75), (lane_x, 13.4)])
        for y1, y2 in zip(ys[:-1], ys[1:]):
            arrow(ax, [(lane_x, y1 - 0.7), (lane_x, y2 + 0.7)])

    box(ax, 10.5, 5.7, 19.0, 1.3,
        "df_ploidy = CNV 'ploidy' layer + label;  df_ploidy_median = median per label;\n"
        "variant -> amplicon map from Sample.dna col_attrs (id, amplicon)  ->  Amplicons_to_variant{tag}.csv",
        fc=GRAY_T, size=8.5)
    arrow(ax, [(5.0, 7.05), (5.0, 6.35)])
    arrow(ax, [(16.0, 7.05), (16.0, 6.35)])

    box(ax, 10.5, 3.2, 19.0, 2.6,
        "get_cnv_info(Sample, df_mut_to_amplicon, desired_plotting_order)\n"
        "df_cnv: cells x amplicons ploidy, 'label' first, duplicate columns/rows dropped, amplicons in genomic order\n"
        "   (chr 1-22, X, Y, M/MT, then unparseable; then numeric start_pos)\n"
        "df_cnv_grouped: amplicons x clones, np.nanmedian per clone, duplicate rows dropped, reindexed to genomic order\n"
        "df_cnv_vars: cells x variants, each variant copies its amplicon's column (NaN if amplicon missing)\n"
        "cnv_info: sorted CNV col_attrs (KeyError if CHROM / start_pos / id missing)",
        fc=BLUE_T, ec=BLUE, size=8.2, ha="left")
    arrow(ax, [(10.5, 5.05), (10.5, 4.5)])
    box(ax, 10.5, 0.8, 19.0, 1.0,
        "save_outputs: Cell_x_ploidy_data / Clone_x_ploidy_data / Variants_x_ploidy_data / Ploidy_metadata{tag}.csv;"
        "  return dict", fc=GRAY_T, size=8.3)
    arrow(ax, [(10.5, 1.9), (10.5, 1.3)])
    return save(fig, PKG, "extract_cnv_flow.png")


# ── 4. prepare_ploidy_region_plot_inputs ────────────────────────────────────────────────────────
def ploidy_region_pipeline():
    fig, ax = canvas(10.5, 8.2, 21, 16.4, title="prepare_ploidy_region_plot_inputs: 12 stages")
    stages = [
        ("1  Annotation", "annotation_exon as is, else\nannotation_df + GTF: exon\ncontaining each variant\n(exon_match_status)"),
        ("2  Standardise", "variant_id, gene_name_plot,\nCHROM_norm (no chr, M->MT),\nvariant_pos, mutation_label\n+ ' / Exon N'"),
        ("3  Lookup", "stack ID-like columns\n(variant_id, index, locus,\nname, ...) into one key\ntable; first key wins"),
        ("4  Diff table", "left-join differential IDs;\nfill gene from 'GENE_...',\nchrom/pos from 'chrN:pos'"),
        ("5  Transcript", "per gene: most frequent\namong matched variants ->\nin annotation -> GTF tx with\nmost exon numbers -> ''"),
        ("6  Amplicons", "CNV col_attrs: id, CHROM,\nstart, end (end = start if\nabsent; swap if reversed)"),
        ("7  Filter", "amplicons of genes of\ninterest that are columns\nof df_ploidy_median\n(ValueError if none)"),
        ("8  Regions", "Exon N from GTF rows of\nchosen transcripts; Intron N\n= gap Exon N .. N+1\nrank N / N + 0.5"),
        ("9  Label amplicons", "overlap with gene regions;\nexon hit -> 'Exon a,b',\nelse 'Intron a,b';\nnone -> rank 10**9"),
        ("10 Label variants", "variant without exon gets\nthe region containing it:\n'label / Intron N'"),
        ("11 Order", "sort by gene, region rank,\nstart, end, id; subset\nplot_df; rows in\ndesired_plotting_order"),
        ("12 Mutation labels", "append label ('; ') to first\ncovering amplicon of gene,\nelse nearest midpoint,\nelse gene's first amplicon"),
    ]
    xs = [2.7, 7.9, 13.1, 18.3]
    ys = [13.0, 8.0, 3.0]
    w, h = 4.7, 3.7
    pos = []
    for i, (title, body) in enumerate(stages):
        r, c = divmod(i, 4)
        if r == 1:
            c = 3 - c  # snake layout
        x, y = xs[c], ys[r]
        pos.append((x, y))
        fc = YELLOW_T if i == 0 else (BLUE_T if i == 11 else SURFACE)
        ec = YELLOW if i == 0 else (BLUE if i == 11 else MUTED)
        box(ax, x, y, w, h, "", fc=fc, ec=ec)
        ax.text(x - w / 2 + 0.2, y + h / 2 - 0.45, title, ha="left", va="center", fontsize=9, weight="bold")
        ax.text(x - w / 2 + 0.2, y - 0.35, body, ha="left", va="center", fontsize=8.1, linespacing=1.35)
    for i in range(len(pos) - 1):
        (x1, y1), (x2, y2) = pos[i], pos[i + 1]
        if y1 == y2:
            d = 1 if x2 > x1 else -1
            arrow(ax, [(x1 + d * w / 2, y1), (x2 - d * w / 2, y2)])
        else:
            arrow(ax, [(x1, y1 - h / 2), (x2, y2 + h / 2)])
    note(ax, 0.4, 0.2, "Returns plot_df (clones x amplicons), amp_sel, diff, gene_to_transcript, transcript_regions, "
         "genes_of_interest, annotation_exon. No files or plots.", size=8.2)
    return save(fig, PKG, "ploidy_region_pipeline.png")


# ── 5. plot_integrated_cnv_baf_with_significance layout ─────────────────────────────────────────
def integrated_plot_layout():
    rng = np.random.default_rng(3)
    n = 24
    arms = ["chr5p"] * 8 + ["chr5q"] * 10 + ["chr7q"] * 6
    bands = ["p15"] * 4 + ["p13"] * 4 + ["q13"] * 5 + ["q31"] * 5 + ["q22"] * 6
    loh = np.array([a == "chr5q" for a in arms])
    mean_baf = np.where(loh, np.where(np.arange(n) % 2, 0.05, 0.95), 0.5 + rng.normal(0, 0.04, n))
    inf = loh.copy()
    inf[9] = False
    diffv = np.zeros(n, bool)
    diffv[20] = True
    depth = rng.uniform(8, 40, n)
    depth[3] = 3.0
    ploidy = np.where(loh, 1.05, 2.0) + rng.normal(0, 0.06, n)

    fig = plt.figure(figsize=(10.5, 7.4))
    gs = fig.add_gridspec(4, 2, width_ratios=[3.0, 1.35], height_ratios=[2.5, 2.5, 1.2, 1.2], hspace=0.35,
                          wspace=0.04)
    axs = [fig.add_subplot(gs[i, 0]) for i in range(4)]
    for a in axs[1:]:
        a.sharex(axs[0])
    x = np.arange(n)

    def separators(a):
        for i in range(1, n):
            if arms[i] != arms[i - 1]:
                a.axvline(i - 0.5, color=INK, lw=1.8)
            elif bands[i] != bands[i - 1]:
                a.axvline(i - 0.5, color=MUTED, lw=0.9, ls="--")
        a.set_xlim(-0.5, n - 0.5)
        a.tick_params(axis="x", bottom=False, labelbottom=False)

    def arm_bg(a):
        start = 0
        k = 0
        for i in range(1, n + 1):
            if i == n or arms[i] != arms[start]:
                if k % 2 == 0:
                    a.axvspan(start - 0.5, i - 0.5, color=GRAY_T, zorder=0)
                k += 1
                start = i

    # BAF
    a = axs[0]
    arm_bg(a)
    for i in range(n):
        cells = np.clip(mean_baf[i] + rng.normal(0, 0.12, 30), 0, 1)
        a.scatter(i + rng.uniform(-0.3, 0.3, 30), cells, s=5, color="#5e548e", alpha=0.12, lw=0)
    for i in range(n):
        alpha = 0.4 if depth[i] < 5 else 1.0
        if diffv[i]:
            a.scatter(i, mean_baf[i], s=60, marker="D", color="#ffc733", ec="black", lw=0.5, alpha=alpha, zorder=4)
        elif inf[i]:
            col = "#48cae4" if mean_baf[i] < 0.5 else "#e63946"
            a.scatter(i, mean_baf[i], s=48, color=col, ec="black", lw=0.4, alpha=alpha, zorder=3)
        else:
            a.scatter(i, mean_baf[i], s=24, color="#b0b0b0", ec="black", lw=0.3, alpha=alpha, zorder=2)
    for yv in (0.25, 0.75):
        a.axhline(yv, color="grey", ls="--", lw=0.9)
    a.set_ylim(0, 1)
    a.set_ylabel("BAF")
    separators(a)
    top = a.secondary_xaxis("top")
    starts = {}
    for i, arm in enumerate(arms):
        starts.setdefault(arm, []).append(i)
    top.set_xticks([np.mean(v) for v in starts.values()])
    top.set_xticklabels(list(starts.keys()), fontsize=8.5)

    # Ploidy
    a = axs[1]
    arm_bg(a)
    segs = [(0, 8, 2.0, "neutral"), (8, 18, 1.05, "extreme"), (18, 24, 2.0, "neutral")]
    for s, e, lvl, _ in segs:
        a.axhspan(lvl - 0.15, lvl + 0.15, xmin=(s) / n, xmax=(e) / n, color="grey", alpha=0.18, lw=0)
        a.hlines(lvl, s - 0.5, e - 0.5, color="black", lw=2.2)
    for yv in (1, 3):
        a.axhline(yv, color="grey", ls="--", lw=0.9)
    a.set_ylim(0, 4)
    a.set_ylabel("Ploidy")
    separators(a)

    # ALT / REF
    alt = np.log10(np.where(loh & (mean_baf < 0.5), 0.3, depth * mean_baf) + 1)
    ref = np.log10(np.where(loh & (mean_baf > 0.5), 0.3, depth * (1 - mean_baf)) + 1)
    for a, vals, fc, col, lab in ((axs[2], alt, "#f8eedb", "#8b4513", "ALT\n(log10)"),
                                  (axs[3], ref, "#dbf8ee", "#459d5b", "REF\n(log10)")):
        a.set_facecolor(fc)
        alphas = np.where(depth < 5, 0.4, 1.0)
        for i in range(n):
            a.scatter(i, vals[i], s=20, color=col, ec="black", lw=0.3, alpha=alphas[i])
        a.set_ylim(0, np.ceil(vals.max() * 10) / 10)
        a.set_ylabel(lab, fontsize=9)
        separators(a)
    for i, bnd in enumerate(bands):
        if i == 0 or bnd != bands[i - 1]:
            axs[3].text(i, -0.25, bnd, transform=axs[3].get_xaxis_transform(), rotation=90, ha="left",
                        va="top", fontsize=7.5, color=INK2)

    # Explanations
    texts = [
        "BAF (0-1): faint single-cell AF/100 jitter\n(<= 100 random cells / variant).\n"
        "Clone mean: big blue (< 0.5) / red (> 0.5)\n= informative for this clone; small grey\n"
        "= not; yellow diamond = differential.\nFaded if mean ALT+REF < 5. Guides 0.25/0.75.",
        "Ploidy (clipped at 4): single-cell jitter\n+ per-variant medians, or with\nplot_ploidy_median=True one black line\n"
        "per (merged) cytoband block = mean of\namplicon means, grey 25-75th pct band.\nGuides at 1 and 3.",
        "ALT: mean log10(count+1) per variant",
        "REF: mean log10(count+1) per variant",
    ]
    for i, t in enumerate(texts):
        ta = fig.add_subplot(gs[i, 1])
        ta.axis("off")
        ta.text(0.04, 0.5, t, va="center", ha="left", fontsize=8.1, color=INK, linespacing=1.35,
                transform=ta.transAxes)
    fig.suptitle("plot_integrated_cnv_baf_with_significance: one figure per clone (schematic, simulated values)",
                 x=0.02, ha="left", fontsize=11, weight="bold", y=0.98)
    fig.text(0.02, 0.015, "Solid black line = chromosome-arm change; dashed grey = cytoband change. Arm names on the "
             "top axis, cytoband names under the last panel.", fontsize=8.2, color=INK2)
    return save(fig, PKG, "integrated_plot_layout.png")


if __name__ == "__main__":
    for f in (informativeness_row_paths, informativeness_arm_passes, extract_cnv_flow, ploidy_region_pipeline,
              integrated_plot_layout):
        print(f())
