"""Draw the README schematics for LOH_Pipeline into ``LOH_Pipeline/docs/``.

Run from the pipelines folder with the `mosaic` conda environment::

    ~/miniconda3/envs/mosaic/bin/python docs/make_LOH_Pipeline_figures.py

Figures
-------
pipeline_overview.png      ``run(...)`` phase order and the clone-label vocabulary.
edge_tolerant_runs.png     ``edge_tolerant_extreme_runs`` on a simulated extreme mask (real function).
detect_loh_steps.png       Steps of ``detect_loh_and_update_clones`` for one arm.
phase1_flow.png            ``run_phase1``: two passes, pool-and-reinstate, amplicon validation, cytobands.
phase1_clone_gates.png     ``_show_clones_for_arm``: the per-clone Phase-1 gates.
phase3_flow.png            ``run_phase3``: QC steps and the per-clone homogeneity decision.
phase4_flow.png            ``run_phase4``: steps 4a-4f.
phase5_flow.png            ``run_phase5``: cleanup, final LOH-block QC and the ADO plausibility model.
phase6_flow.png            ``run_post_phase5_subclone_scan``: residual-parent passes.

All thresholds are the ``build_config`` defaults (which equal the ``LOHConfig`` defaults except where noted).
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from _style import (BLUE, BLUE_T, GRAY_T, INK, INK2, MUTED, ORANGE, ORANGE_T, PIPELINES,  # noqa: E402
                    SURFACE, YELLOW, YELLOW_T, arrow, box, canvas, note, plt, save)

sys.path.insert(0, PIPELINES)

import numpy as np  # noqa: E402

from LOH_Pipeline.loh_config import edge_tolerant_extreme_runs  # noqa: E402

PKG = "LOH_Pipeline"
WHITE = "#ffffff"


def kept(ax, x, y, w, h, text, **kw):
    box(ax, x, y, w, h, text, fc=BLUE_T, ec=BLUE, **kw)


def flagged(ax, x, y, w, h, text, **kw):
    box(ax, x, y, w, h, text, fc=ORANGE_T, ec=ORANGE, **kw)


def neutral(ax, x, y, w, h, text, **kw):
    box(ax, x, y, w, h, text, fc=GRAY_T, **kw)


def decision(ax, x, y, w, h, text, **kw):
    box(ax, x, y, w, h, text, fc=WHITE, ec=INK2, **kw)


def down(ax, x, y_from, y_to, label=None, dx=0.15):
    arrow(ax, [(x, y_from), (x, y_to)], label=label,
          label_xy=(x + dx, (y_from + y_to) / 2) if label else None, ha="left")


def right(ax, x_from, x_to, y, label=None, dy=0.22):
    arrow(ax, [(x_from, y), (x_to, y)], label=label,
          label_xy=((x_from + x_to) / 2, y + dy) if label else None)


# ── 1. Pipeline overview ─────────────────────────────────────────────────────────────────────────
def pipeline_overview():
    fig, ax = canvas(10.5, 7.2, 21, 15.8, title="LOH_Pipeline.run(...): phase order and clone-label vocabulary")
    ax.set_ylim(1.4, 15.8)

    x, w, h, step, y0 = 5.3, 10.4, 1.08, 1.5, 14.3
    steps = [
        ("Phase 0  run_phase0\nkeep arms with >= 4 variants in df; build PipelineState", "n"),
        ("Phase 1  run_phase1\nper-arm detection adds '<base> LOH-C<n> <arm>'; strict QC", "k"),
        ("Phase 2  p/q arm merge: SKIPPED\n(run records {'status': 'SKIPPED'} only)", "skip"),
        ("Phase 3  run_phase3\noutliers, homogeneity: LOH / new child / 'SUB: ...' / 'FP: ...'", "k"),
        ("Phase 4  run_phase4\nmerge/rescue non-LOH cells, 'ADO' label, reassignment, residual", "k"),
        ("Phase 5  run_phase5\nlabels < 5 cells -> 'small'; LOH-block QC, ADO plausibility", "k"),
        ("Phase 6 + 7  (only with post_phase5_subclone_scan_labels)\nnested child scan of final parents, then Phase-5 summary again", "opt"),
        ("Final assignment accounting\nevery Phase-0 cell -> LOH / SUB / FP / non-LOH / ADO / small", "n"),
        ("plot_loh_clones  (if run_plotting)\nper-clone AF plots via CNV_Utils.visualise_variants_by_clone", "n"),
    ]
    for i, (text, kind) in enumerate(steps):
        y = y0 - i * step
        if kind == "skip":
            box(ax, x, y, w, h, text, fc=SURFACE, ec=MUTED, ls="--", size=8.6)
        elif kind == "opt":
            box(ax, x, y, w, h, text, fc=GRAY_T, ls="--", size=8.6)
        elif kind == "k":
            kept(ax, x, y, w, h, text, size=8.6)
        else:
            neutral(ax, x, y, w, h, text, size=8.6)
        if i:
            down(ax, x, y + step - h / 2, y + h / 2)

    # Label vocabulary
    xr, wr = 16.15, 9.3
    ax.text(xr - wr / 2, 15.15, "Clone labels (Sample.dna.row_attrs['Clone'])", fontsize=9.6, weight="bold", va="center")
    entries = [
        (kept, "Active LOH  (is_loh_label)\nTP53_R248W_Hem LOH-C0 chr17p\nmulti-arm: <base> LOH-C0 chr5q LOH-C1 chr7q\nPhase 1 may add a band range: chr5q:q14.3-q33.1"),
        (neutral, "SUB: <base> C0 chr17p   (is_sub_label)\ncoherent LOH cells with no validated directional\nchild, or a multi-arm clone failing some arms"),
        (flagged, "FP: <base> C0 chr17p   (is_fp_label)\nfailed LOH QC; kept for review ('possible LOH')"),
        (flagged, "small   (is_small_label)\nany terminal label with < 5 cells after Phase 5"),
        (flagged, "ADO\nPhase 4b: Het/WT cell matching a Hem/Hom profile;\nremoved from the AF table"),
        (neutral, "Base clone (non-LOH), e.g. TP53_R248W_Hem, WT\nget_base_clone() strips every ' LOH-C<n> <arm>'"),
    ]
    heights = [1.75, 1.45, 1.1, 1.1, 1.45, 1.1]
    y = 14.55
    for (fn, text), hh in zip(entries, heights):
        fn(ax, xr, y - hh / 2, wr, hh, text, size=8.4, ha="left")
        y -= hh + 0.3
    note(ax, xr - wr / 2, y - 1.15,
         "FP:/SUB: rewrite LOH-C<n> to C<n>, so the text 'LOH' no longer\n"
         "matches. is_uncertain_loh_label = FP or SUB. clone_category()\n"
         "-> small / subclonal_LOH / possible_LOH / LOH / non_LOH.", size=8.3)
    return save(fig, PKG, "pipeline_overview.png")


# ── 2. Edge-tolerant runs (real function, simulated mask) ────────────────────────────────────────
def edge_tolerant_runs():
    mask = np.array([0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0])
    runs = edge_tolerant_extreme_runs(mask, max_edge_tolerance=1)
    fig, ax = plt.subplots(figsize=(10.5, 3.3))
    ax.set_xlim(-0.8, len(mask) + 0.2)
    ax.set_ylim(-2.9, 1.6)
    ax.axis("off")
    ax.set_title("edge_tolerant_extreme_runs(mask, max_edge_tolerance=1) on a simulated clone-mean extreme mask",
                 loc="left", fontsize=11, weight="bold", pad=8)
    for i, m in enumerate(mask):
        ax.add_patch(plt.Rectangle((i, 0), 0.9, 0.9, fc=BLUE_T if m else SURFACE, ec=BLUE if m else MUTED, lw=1.2))
        ax.text(i + 0.45, 0.45, str(m), ha="center", va="center", fontsize=10)
        ax.text(i + 0.45, 1.15, str(i), ha="center", va="center", fontsize=8.5, color=INK2)
    ax.text(-0.7, 0.45, "extreme", ha="left", va="center", fontsize=8.5, color=INK2, rotation=90)
    for k, r in enumerate(runs):
        core, span = r["core_indices"], r["span_indices"]
        y_core, y_span = -0.45, -1.05
        ax.plot([core[0] + 0.05, core[-1] + 0.85], [y_core, y_core], color=BLUE, lw=3, solid_capstyle="butt")
        ax.plot([span[0] + 0.05, span[-1] + 0.85], [y_span, y_span], color=INK2, lw=2, ls="--")
        ok = len(span) >= 4
        ax.text((span[0] + span[-1] + 0.9) / 2, -1.5,
                f"core {core[0]}-{core[-1]}, span {span[0]}-{span[-1]} ({len(span)})\n"
                + ("span >= 4: block" if ok else "span < 4: no block"),
                ha="center", va="top", fontsize=8.4, color=INK if ok else ORANGE)
    ax.add_patch(plt.Rectangle((4, 0), 0.9, 0.9, fc="none", ec=ORANGE, lw=2))
    ax.text(4.45, -0.55, "contested\n(neither)", ha="center", va="top", fontsize=7.8, color=ORANGE)
    ax.plot([], [], color=BLUE, lw=3, label="core (strict support: purity, direction, amplicons)")
    ax.plot([], [], color=INK2, lw=2, ls="--", label="span (core + terminal edge padding: block length only)")
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, -0.08), ncol=2, frameon=False, fontsize=8.6)
    return save(fig, PKG, "edge_tolerant_runs.png")


# ── 3. detect_loh_and_update_clones ──────────────────────────────────────────────────────────────
def detect_loh_steps():
    fig, ax = canvas(10.5, 9.7, 21, 21.2, title="detect_loh_and_update_clones: one chromosome arm")
    ax.set_ylim(1.8, 21.2)
    x, w = 8.0, 15.6
    xe, we = 18.75, 4.0
    rows = [
        ("0  Reference labels: clone_reference_labels given -> align by barcode / position;\n"
         "   else Clone_backup (made from Clone on first call; restore_from_backup=True copies it back)", None),
        ("1  Arm variants present in df, sorted by position; LOW = AF <= 10, HIGH = AF >= 90", "no variants"),
        ("2  Per cell: connected runs of extreme variants, length >= min_segment_length (4)\n"
         "   and extreme fraction >= min_extreme_fraction (function 0.80; Phase 1 / 6 pass 0.90)", "no segments"),
        ("3  Greedy clustering in df row order: a seed absorbs every later unassigned cell with\n"
         "   |LOW ∩ LOW| + |HIGH ∩ HIGH| >= 4 shared with the seed; keep groups >= 5 cells", "no clusters"),
        ("3b Arm-wide coherence: drop a cluster unless >= 40 % of cell pairs have r >= 0.30\n"
         "   (constant pair: r = 1 if mean |dAF| < 15 else 0); keep_incoherent_clusters skips this", "none left"),
        ("4  Map df rows -> barcode -> base clone (reference label)", None),
        ("5  Clone-mean profiles (>= 10 shared variants): merge same-base LOH clusters with\n"
         "   r >= correlation_threshold (0.60; pipeline 0.80); absorb a same-base non-LOH clone\n"
         "   with r >= threshold if it is no larger than the LOH clone", None),
        ("6  Size gate: '<base> LOH-C<k> <arm>' >= min_final_clone_size (5)\n"
         "   (skipped with skip_size_filtering: Phase 1 skips, Phase 6 applies)", "none pass"),
        ("7  Output frames: df_loh_segments, df_loh_only, df_correlations", None),
        ("8  Append ' LOH-C<k> <arm>' to the current Clone label (never added twice)", None),
        ("9  Homogeneity (unless skip_homogeneity_check; Phase 1 and 6 skip it): keep a clone if\n"
         "   >= 70 % of cell pairs have r >= 0.60; else strip only this arm's token", None),
    ]
    heights = [1.25, 0.8, 1.25, 1.25, 1.25, 0.8, 1.6, 1.25, 0.8, 0.8, 1.25]
    y = 20.2
    centers = []
    for (text, exit_), hh in zip(rows, heights):
        yc = y - hh / 2
        centers.append((yc, hh))
        fc = BLUE_T if text.startswith("8") else GRAY_T
        ec = BLUE if text.startswith("8") else MUTED
        box(ax, x, yc, w, hh, text, fc=fc, ec=ec, size=8.5, ha="left")
        if exit_:
            flagged(ax, xe, yc, we, 0.85, f"{exit_}: return\n4 empty DataFrames", size=7.8)
            right(ax, x + w / 2, xe - we / 2, yc)
        y -= hh + 0.42
    for (y1, h1), (y2, h2) in zip(centers[:-1], centers[1:]):
        down(ax, x, y1 - h1 / 2, y2 + h2 / 2)
    yl, hl = centers[-1]
    kept(ax, x, yl - hl / 2 - 0.95, w, 0.75,
         "Return (df_loh_segments, df_loh_only, df_correlations, df_final_homogeneity)", size=8.6)
    down(ax, x, yl - hl / 2, yl - hl / 2 - 0.57)
    return save(fig, PKG, "detect_loh_steps.png")


# ── 4. Phase 1 ───────────────────────────────────────────────────────────────────────────────────
def phase1_flow():
    fig, ax = canvas(10.5, 6.6, 21, 13.2, title="run_phase1: two passes over the arms, then arm-level clean-up")
    x, w = 10.5, 20.4
    rows = [
        (neutral, "Pass 1, each arm: detect_loh_fn (size filter and homogeneity deferred to later phases); "
                  "an exception marks the arm ERROR", 0.8),
        (neutral, "For every LOH clone >= 5 cells: median SD of folded extreme AFs on its clone-mean block "
                  "-> sd_med.\nglobal dynamic SD = max(sd_med) x phase1_sd_dynamic_factor (1.2); None when no "
                  "clone qualified", 1.15),
        (kept, "Pass 2, each arm: _show_clones_for_arm (per-clone gates, next figure). Failing clones lose "
               "' LOH-C<n> <arm>';\nif no clone on the arm passes, the arm is DROPPED", 1.15),
        (neutral, "Pool-and-reinstate DROPPED arms: cells >= 50 % extreme on the arm, pooled by base clone; "
                  "a pool >= 5 cells with\npooled extreme >= 0.90 (or a qualifying clone-mean block) is "
                  "relabelled with a new LOH-C<k> (no other gates re-run)", 1.15),
        (kept, "Amplicon validation (validate_segment_length): a clone-mean run with span >= 4 must cover "
               ">= 3 informative\namplicons, else the arm token is stripped; an arm with no LOH cells left is "
               "dropped", 1.15),
        (neutral, "Cytoband annotation: clones that passed by a localised block become "
                  "'<base> LOH-C<n> <arm>:<band>-<band>'", 0.8),
        (neutral, "Printed summary per arm: LOH / DROPPED / - / ERROR; results stored in state.results", 0.8),
    ]
    y = 12.5
    cl = []
    for fn, t, hh in rows:
        fn(ax, x, y - hh / 2, w, hh, t, size=8.3, ha="left")
        cl.append((y - hh / 2, hh))
        y -= hh + 0.5
    for (y1, h1), (y2, h2) in zip(cl[:-1], cl[1:]):
        down(ax, x, y1 - h1 / 2, y2 + h2 / 2)
    ax.set_ylim(y + 0.2, 13.2)
    return save(fig, PKG, "phase1_flow.png")


def phase1_clone_gates():
    fig, ax = canvas(10.5, 9.0, 21, 18.0, title="_show_clones_for_arm: Phase-1 gates for each LOH clone on an arm")
    xc = 10.5
    neutral(ax, xc, 16.9, 13.0, 0.85,
            "LOH block = longest edge-tolerant extreme run of the clone-mean profile\n"
            "(span >= 4, somatic-flanking rule passes); ties -> higher block extreme fraction", size=8.4)
    decision(ax, xc, 15.2, 9.0, 0.8, "n >= phase1_min_clone_cells_all (5)?", size=8.5)
    down(ax, xc, 16.475, 15.6)
    flagged(ax, 18.6, 15.2, 3.6, 0.7, "fail:n<5", size=8.2)
    right(ax, xc + 4.5, 16.8, 15.2, "no")
    decision(ax, xc, 13.3, 9.0, 1.0, "arm-wide extreme fraction (all arm AFs)\n>= min_extreme_fraction (0.90)?", size=8.5)
    down(ax, xc, 14.8, 13.8, "yes")
    # yes branch: SD QC
    neutral(ax, 4.6, 10.4, 8.4, 1.25, "SD QC on the block (n >= 5):\nsd_median <= 10 and sd_p90 <= 15", size=8.4)
    arrow(ax, [(xc - 4.5, 13.3), (4.6, 13.3), (4.6, 11.025)], label="yes", label_xy=(4.75, 12.2), ha="left")
    flagged(ax, 2.6, 8.0, 4.4, 0.85, "fail:extreme-\nAF-variance", size=8.0)
    arrow(ax, [(2.6, 9.775), (2.6, 8.425)], label="no", label_xy=(2.75, 9.1), ha="left")
    # no branch: localised
    neutral(ax, 15.9, 9.85, 9.6, 2.35,
            "Localised check (cytobands given)\n"
            "- best run of block bands, each >= 0.90 extreme,\n  >= 4 variants; somatic flanking OK\n"
            "- cell-to-block-mean r: mean >= 0.85, >= 85 % of\n  cells with r >= 0.85; sd_median <= dynamic SD",
            size=8.2, ha="left")
    arrow(ax, [(xc + 4.5, 13.3), (15.9, 13.3), (15.9, 11.025)], label="no", label_xy=(16.05, 12.2), ha="left")
    flagged(ax, 18.3, 7.35, 4.8, 0.85, "arm-wide_fail (no cytobands)\n/ localized_fail:...", size=7.8)
    arrow(ax, [(18.3, 8.675), (18.3, 7.775)], label="no", label_xy=(18.45, 8.2), ha="left")
    # join at directional
    decision(ax, xc, 5.6, 9.0, 1.25,
             "Directional consistency on the block: variants\nwith >= 5 extreme cells need >= 95 % one\ndirection; all of them must pass",
             size=8.2)
    arrow(ax, [(6.8, 9.775), (6.8, 6.225)], label="pass", label_xy=(6.95, 8.0), ha="left")
    arrow(ax, [(13.4, 8.675), (13.4, 6.225)], label="pass", label_xy=(13.55, 7.5), ha="left")
    flagged(ax, 18.6, 5.6, 3.6, 0.85, "fail:directional", size=8.0)
    right(ax, xc + 4.5, 16.8, 5.6, "no")
    decision(ax, xc, 3.6, 9.0, 1.0, "Strip cells with any NGT / no-call in the block\n(max fraction 0.0); still >= 5 cells?",
             size=8.2)
    down(ax, xc, 4.975, 4.1, "pass")
    flagged(ax, 18.6, 3.6, 3.6, 0.85, "fail:NGT_\nexcluded_n<5", size=8.0)
    right(ax, xc + 4.5, 16.8, 3.6, "no")
    kept(ax, xc, 1.7, 9.0, 0.95, "PASS: block variants -> state.clone_block_variants[label]", size=8.4)
    down(ax, xc, 3.1, 2.175, "yes")
    note(ax, 0.3, 0.3, "Dynamic SD = global value from Pass 1, or the static 10 when it is None. "
                       "Failing clones are stripped by run_phase1.", size=8.2)
    return save(fig, PKG, "phase1_clone_gates.png")


# ── 5. Phase 3 ───────────────────────────────────────────────────────────────────────────────────
def phase3_flow():
    fig, ax = canvas(10.5, 9.0, 23, 21.0, title="run_phase3: LOH clone QC (labels rewritten in place)")
    ax.set_ylim(1.2, 21.0)
    xl, wl = 4.9, 9.4
    ax.text(0.2, 20.3, "run_phase3 steps, in order", fontsize=9.4, weight="bold", va="center")
    left = [
        (neutral, "0  Seed numpy/random with 123; snapshot every\n   cell's pre-QC LOH label"),
        (neutral, "1  Outliers (clones >= 3 cells): cell r vs clone mean\n   on the LOH block. > 10 cells: drop r < mean - 2 SD;\n   <= 10 cells: drop r < 0.80  ->  base label"),
        (kept, "2  Rescue merge: LOH clone < 5 cells -> same-base\n   LOH clone with a strict superset of arms and\n   r >= 0.80 on every target arm"),
        (kept, "3  Homogeneity (right): stays LOH / new child\n   LOH-C<new> / 'SUB: ...' / 'FP: ...'"),
        (kept, "3b Nested rescue: 'FP:' cells whose same-base\n   active child has a superset of arms -> child /\n   parent / stay FP (one unambiguous child only)"),
        (flagged, "4  Active LOH clones < 5 cells -> base label"),
        (flagged, "4b Cells LOH before Phase 3, now plain, still >= 0.90\n   extreme on an LOH arm -> 'FP: <pre-QC label>'"),
        (neutral, "5  Reconcile state.results with current labels\n   (the only step when skip_homogeneity_check=True)"),
    ]
    hs = [1.25, 1.65, 1.65, 1.25, 1.65, 0.8, 1.25, 1.25]
    y = 19.7
    cl = []
    for (fn, t), hh in zip(left, hs):
        fn(ax, xl, y - hh / 2, wl, hh, t, size=8.3, ha="left")
        cl.append((y - hh / 2, hh))
        y -= hh + 0.5
    for (y1, h1), (y2, h2) in zip(cl[:-1], cl[1:]):
        down(ax, xl, y1 - h1 / 2, y2 + h2 / 2)

    # Right: homogeneity decision
    xr, wr = 15.0, 8.4
    xf = 21.6
    ax.text(10.8, 20.3, "_clone_homogeneity_check: each LOH clone", fontsize=9.4, weight="bold", va="center")
    decision(ax, xr, 19.2, wr, 0.8, ">= 2 cells?", size=8.5)
    flagged(ax, xf, 19.2, 2.2, 0.7, "FP:", size=8.5)
    right(ax, xr + wr / 2, xf - 1.1, 19.2, "no")
    decision(ax, xr, 17.45, wr, 1.0, "multi-arm (>= 2 LOH tokens)?", size=8.5)
    down(ax, xr, 18.8, 17.95, "yes")
    neutral(ax, xr, 15.35, wr, 1.65,
            "multi-arm: each arm on its Phase-1 block\n(>= 4 variants): mean cell-to-mean r >= 0.80\nall pass: keep | some: SUB: | none: FP:",
            size=8.3)
    down(ax, xr, 16.95, 16.175, "yes")
    decision(ax, xr, 12.9, wr, 1.25, "single arm: mean cell-to-mean r\n>= phase3_min_mean_corr_to_mean (0.80)?", size=8.3)
    arrow(ax, [(xr - wr / 2, 17.45), (xr - wr / 2 - 0.35, 17.45), (xr - wr / 2 - 0.35, 12.9), (xr - wr / 2, 12.9)],
          label="no", label_xy=(xr - wr / 2 - 0.2, 16.6), ha="right")
    flagged(ax, xf, 12.9, 2.2, 0.7, "FP:", size=8.5)
    right(ax, xr + wr / 2, xf - 1.1, 12.9, "no")
    decision(ax, xr, 10.75, wr, 1.25, "directional consistency (>= 95 % per\nvariant, all variants with >= 5 extreme cells)?",
             size=8.3)
    down(ax, xr, 12.275, 11.375, "yes")
    kept(ax, xf, 10.75, 2.2, 0.7, "keep", size=8.5)
    right(ax, xr + wr / 2, xf - 1.1, 10.75, "yes")
    decision(ax, xr, 8.45, wr, 1.65,
             "ADO tolerance: >= 6 variants evaluated,\n>= 5 pass, failing ones in <= 1 amplicon?\n(build_config; LOHConfig: 5 / 4)",
             size=8.3)
    down(ax, xr, 10.125, 9.275, "no")
    kept(ax, xf, 8.45, 2.2, 0.7, "keep", size=8.5)
    right(ax, xr + wr / 2, xf - 1.1, 8.45, "yes")
    neutral(ax, xr, 4.75, wr, 3.8,
            "Directional split\n"
            "- anchor = discordant variant with the most\n  balanced LOW/HIGH split\n"
            "- LOW and HIGH sign signatures over the\n  discordant variants; each cell -> closer one\n"
            "- group >= 5 cells passing homogeneity\n  -> '<base> LOH-C<new> <arm>'\n"
            "- unassigned cells -> SUB:;  no child -> FP:",
            size=8.2, ha="left")
    down(ax, xr, 7.625, 6.65, "no")
    note(ax, 10.8, 1.5,
         "Recovered children get their own block in\nclone_block_variants; failed parents lose theirs.", size=8.2)
    return save(fig, PKG, "phase3_flow.png")


# ── 6. Phase 4 ───────────────────────────────────────────────────────────────────────────────────
def phase4_flow():
    fig, ax = canvas(10.5, 8.6, 21, 20.8, title="run_phase4: steps 4a-4f (4e legacy discovery is not run)")
    x, w = 10.5, 20.4
    rows = [
        (neutral, "Classify: active LOH clones (any size) and active non-LOH clones (>= 5 cells)", 0.8),
        (kept, "4a  Clone merge. Each non-LOH clone (largest first) vs each same-base LOH clone, per arm with >= 2\n"
               "      variants: clone-mean Pearson r. Mean r >= correlation_threshold (0.80) and mean extreme\n"
               "      fraction >= merge_min_extreme_frac (0.85)  ->  whole clone relabelled to the LOH clone", 1.6),
        (kept, "4a  Per-cell rescue (if r passed, mean r >= 0.70 or any arm r >= 0.85). Per amplicon: all its\n"
               "      variants extreme and in the profile direction; all amplicons must pass (one may fail once the\n"
               "      block has >= 5; floor 3, capped at n_amp) and cell r >= 0.80  ->  cell relabelled", 1.6),
        (flagged, "4b  ADO check ('PHASE 4.1'). Het / WT cells vs Hem / Hom target means, somatic amplicon excluded:\n"
                  "      >= 20 variants, r >= 0.80, median |dAF| <= 15  ->  'ADO', cell removed from df / df_numeric", 1.25),
        (neutral, "4c  Cross-LOH similarity ('PHASE 4.4'), report only: LOH clone pairs sharing an arm; suggestion if\n"
                  "      r >= 0.95, median |dAF| <= 10, extreme >= 0.85, block overlap >= 50 %, compatible direction", 1.25),
        (kept, "4d  Cell reassignment ('PHASE 4.5'), up to 10 passes until no cell moves or a label state repeats.\n"
               "      Each cell of a smaller LOH clone (arms a strict subset of the target's) moves to the largest\n"
               "      passing target: per arm >= 4 shared variants, r >= 0.95, median |dAF| <= 10, extreme >= 0.85", 1.6),
        (kept, "4f  Residual base-clone pass ('PHASE 4.7'), once. Per arm in each non-LOH clone >= 5 cells:\n"
               "      cells with >= 4 contiguous extremes, linked by >= 4 shared same-sign extremes -> components\n"
               "      >= 5 cells; homogeneity (r >= 0.80, median |d| <= 15), direction, >= 3 amplicons and\n"
               "      score_ado_combination = ADO_UNLIKELY (rate 0.01 per amplicon, p < 0.01)  ->  new / existing LOH", 1.95),
        (neutral, "Summary print; results keys phase4_* written to state.results", 0.8),
    ]
    y = 19.9
    cl = []
    for i, (fn, t, hh) in enumerate(rows):
        xi, wi = (7.0, 13.4) if i == 0 else (x, w)
        fn(ax, xi, y - hh / 2, wi, hh, t, size=8.3, ha="left")
        cl.append((y - hh / 2, hh))
        y -= hh + 0.55
    for (y1, h1), (y2, h2) in zip(cl[:-1], cl[1:]):
        down(ax, 7.0 if y1 == cl[0][0] else x, y1 - h1 / 2, y2 + h2 / 2)
    y0, h0 = cl[0]
    flagged(ax, 17.6, y0, 5.6, 0.95, "no LOH clone or no non-LOH clone:\nreturn (4a-4f skipped)", size=7.8)
    right(ax, 13.7, 14.8, y0)
    ax.set_ylim(y - 1.6, 20.8)
    note(ax, 0.3, y - 1.4,
         "4d targets: active LOH >= 5 cells, compatible base name, larger than the source; Phase-3b nested\n"
         "parent/child pairs are protected. Ties go to higher r; r within 0.02 of each other is skipped as ambiguous.",
         size=8.2)
    return save(fig, PKG, "phase4_flow.png")


# ── 7. Phase 5 ───────────────────────────────────────────────────────────────────────────────────
def phase5_flow():
    fig, ax = canvas(10.5, 8.6, 23, 18.6, title="run_phase5: final cleanup, LOH-block QC and ADO plausibility")
    xl, wl = 5.3, 10.2
    left = [
        (neutral, "Merge near-identical LOH siblings (same base and arm\nset, r >= 0.80): OFF by default", 1.25, "--"),
        (flagged, "Prune: every label with < 5 cells (or loh_min_cells for\nLOH labels) and every 'small: ...' -> 'small'", 1.25, "-"),
        (neutral, "Per clone (largest first): flags LOH / SUB / small /\nHem / TP53 / chr17p and its LOH arms", 1.25, "-"),
        (kept, "Active LOH, each arm: best block of the clone mean\n(mean AF <= 10 or >= 90, >= 4 variants, somatic-\nflanking OK, >= 3 amplicons); most amplicons wins", 1.65, "-"),
        (neutral, "Confidence from the weakest block:\n<= 3 amplicons low, 4 mid, >= 5 high", 1.25, "-"),
        (kept, "Table -> state.results['phase5_summary']; printed\nreport: QC alerts (blocks with exactly 3 amplicons),\nADO readouts, Hem-TP53 without chr17p warning", 1.65, "-"),
    ]
    y = 17.9
    cl = []
    for fn, t, hh, ls in left:
        fn(ax, xl, y - hh / 2, wl, hh, t, size=8.3, ha="left", ls=ls)
        cl.append((y - hh / 2, hh))
        y -= hh + 0.55
    for (y1, h1), (y2, h2) in zip(cl[:-1], cl[1:]):
        down(ax, xl, y1 - h1 / 2, y2 + h2 / 2)

    xr, wr = 17.2, 10.6
    ax.text(11.9, 18.0, "ADO plausibility, each block", fontsize=9.4, weight="bold", va="center")
    decision(ax, xr, 16.7, wr, 1.25,
             "phase5_qc_alert_ado_rate_source in {empirical,\nempirical_background, background}? (default yes)", size=8.3)
    neutral(ax, xr, 13.95, wr, 2.1,
            "Background = same-base cells not LOH / FP / SUB /\nsmall. Per amplicon: event = all covered variants\n"
            "extreme in the block direction (DP filter: see\nquirks). rate = (events + 1) / (valid + 10)",
            size=8.2, ha="left")
    down(ax, xr, 16.075, 15.0, "yes")
    decision(ax, xr, 11.45, wr, 1.0, "every amplicon >= 50 valid background cells?", size=8.3)
    down(ax, xr, 12.9, 11.95)
    kept(ax, xr - 2.75, 9.55, 5.0, 1.0, "p_cell = product of\nper-amplicon rates", size=8.3)
    neutral(ax, xr + 2.75, 9.55, 5.0, 1.0, "p_cell = 0.10 ** n_amp\n('fixed_fallback')", size=8.3)
    arrow(ax, [(xr - 2.75, 10.95), (xr - 2.75, 10.05)], label="yes", label_xy=(xr - 2.6, 10.5), ha="left")
    arrow(ax, [(xr + 2.75, 10.95), (xr + 2.75, 10.05)], label="no", label_xy=(xr + 2.9, 10.5), ha="left")
    neutral(ax, xr, 7.25, wr, 1.25,
            "N = base-lineage cells; expected = N x p_cell\np = P[Binomial(N, p_cell) >= observed clone size]", size=8.3)
    arrow(ax, [(xr - 2.75, 9.05), (xr - 2.75, 7.875)])
    arrow(ax, [(xr + 2.75, 9.05), (xr + 2.75, 7.875)])
    decision(ax, xr, 5.3, wr, 0.8, "p < 0.01?", size=8.5)
    down(ax, xr, 6.625, 5.7)
    kept(ax, xr - 2.75, 3.6, 5.0, 1.0, "'unlikely from\nADO alone'", size=8.3)
    flagged(ax, xr + 2.75, 3.6, 5.0, 1.0, "'statistically plausible\nunder ADO'", size=8.3)
    arrow(ax, [(xr - 2.75, 4.9), (xr - 2.75, 4.1)], label="yes", label_xy=(xr - 2.6, 4.5), ha="left")
    arrow(ax, [(xr + 2.75, 4.9), (xr + 2.75, 4.1)], label="no", label_xy=(xr + 2.9, 4.5), ha="left")
    note(ax, 11.9, 0.9,
         "Source not empirical: p_cell = 0.10 ** n_amp ('fixed').\n0.10 is the build_config default (LOHConfig: 0.20).\n"
         "Per clone the block with the largest p is reported.", size=8.2)
    return save(fig, PKG, "phase5_flow.png")


# ── 8. Phase 6 ───────────────────────────────────────────────────────────────────────────────────
def phase6_flow():
    fig, ax = canvas(10.5, 9.2, 21, 21.2, title="run_post_phase5_subclone_scan (Phase 6): nested LOH children in final parents")
    x, w = 9.8, 18.8
    rows = [
        (box, YELLOW_T, YELLOW,
         "Parents: explicit labels, 'auto' (parents with >= 5 cells carrying their own >= 4-variant extreme run)\n"
         "or 'all'. WT excluded unless auto_include_wt. Size floor min_parent_cells = max(2 x 5, 5) = 10", 1.25),
        (box, GRAY_T, MUTED,
         "Eligibility: parent is not LOH / SUB / FP / small / ADO and has >= min_parent_cells cells", 0.8),
        (box, GRAY_T, MUTED,
         "Residual pass k (cells still carrying the exact parent label); per arm with >= 4 variants:", 0.8),
        (box, BLUE_T, BLUE,
         "  1  detect_loh_and_update_clones on the residual cells against a frozen reference\n"
         "     (size gate on, homogeneity off; min_extreme_fraction 0.90, correlation 0.80, min child 5)", 1.25),
        (box, BLUE_T, BLUE,
         "  2  Strict validation of each new child on this arm: >= 5 cells; block = edge-tolerant run of\n"
         "     variants where >= 90 % of cells are extreme (>= 4); >= 80 % of cells with r >= 0.80 to the\n"
         "     block mean; directional consistency; >= 3 amplicons. Failing children lose this arm token", 1.6),
        (box, GRAY_T, MUTED,
         "  3  Nothing changed: signed-block partition (seed signatures, match >= 0.8 n / 0.9 n / n positions,\n"
         "     strict validation, greedy non-overlapping groups)", 1.25),
        (box, BLUE_T, BLUE,
         "  4  New primary child: expand with similar residual cells (r >= 0.85, median |d| <= 15, block\n"
         "     extreme >= 0.75, signed agreement >= 0.75), then partial-block members (run >= 4, r >= 0.80,\n"
         "     >= 3 amplicons, signs all matching)", 1.6),
        (box, GRAY_T, MUTED,
         "  5  Second arm in < 90 % of the primary family: keep the primary label; record the arm in\n"
         "     row_attrs['phase6_secondary_loh_arms']", 1.25),
        (box, GRAY_T, MUTED,
         "Next pass while cells moved, residual >= min_parent_cells and the residual state is new", 0.8),
        (box, BLUE_T, BLUE,
         "Merge-back (merge_back_after_scan): Phase-4d cell merge once, equal arm sets allowed", 0.8),
        (box, BLUE_T, BLUE,
         "Phase 7: run_phase5 again (heading 'PHASE 7'); audit table in results['audit_df']", 0.8),
    ]
    y = 20.3
    cl = []
    for _, fc, ec, t, hh in rows:
        box(ax, x, y - hh / 2, w, hh, t, fc=fc, ec=ec, size=8.2, ha="left")
        cl.append((y - hh / 2, hh))
        y -= hh + 0.5
    for (y1, h1), (y2, h2) in zip(cl[:-1], cl[1:]):
        down(ax, x, y1 - h1 / 2, y2 + h2 / 2)
    # loop arrow from "next pass" back to "residual pass"
    ax.set_ylim(y - 0.2, 21.2)
    (yn, hn), (yp, hp) = cl[8], cl[2]
    arrow(ax, [(x + w / 2, yn), (x + w / 2 + 0.7, yn), (x + w / 2 + 0.7, yp), (x + w / 2, yp)],
          label="loop", label_xy=(x + w / 2 + 0.85, (yn + yp) / 2), ha="left")
    return save(fig, PKG, "phase6_flow.png")


if __name__ == "__main__":
    for fn in (pipeline_overview, edge_tolerant_runs, detect_loh_steps, phase1_flow, phase1_clone_gates, phase3_flow, phase4_flow,
               phase5_flow, phase6_flow):
        print(fn())
