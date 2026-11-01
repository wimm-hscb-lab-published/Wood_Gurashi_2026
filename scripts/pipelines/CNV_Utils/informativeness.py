"""Germline-marker informativeness testing for clone-level allelic (LOH) events.

This module decides which germline heterozygous variants are *informative*
markers for a clone, i.e. show a clone-specific allelic imbalance relative to
the WT/reference population that is consistent with loss of heterozygosity
(deletion, copy-neutral LOH or amplification).

Main entry points
-----------------
test_variant_informativeness
    Per chromosome arm, compare every non-WT clone with the WT/reference cells
    for each variant and classify it through a hierarchy of decision paths
    (direct near-zero allele call, low-depth corroborated call, strict
    statistical fallback, LOH-arm pattern rescue, cross-Hem anti-phase rescue).
    Writes one ``wilcoxon_results_<arm>_enhanced.csv`` per arm.
expand_informative_variants
    Reload the raw H5 with relaxed variant filtering, rerun
    ``test_variant_informativeness`` on the same cells, quality-gate newly
    found informative variants and return an expanded ``Sample`` object.

Notes
-----
``test_variant_informativeness`` is a library function despite its ``test_``
prefix; import it via the module/package object in test code so that pytest
does not try to collect it.

WT/reference labels are always matched *exactly* (after ``str.strip``), never
by prefix, so a label such as ``'WT LOH-C2 chr7q:q22.1'`` is not treated as WT
unless it is listed explicitly.
"""

import os
import sys
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from scipy.stats import beta, binomtest, ranksums


def _normalize_wt_labels(wt_labels: Optional[Union[str, List[str]]]) -> List[str]:
    """Return WT/reference labels as a list (``None`` -> ``['WT']``, str -> ``[str]``)."""
    if wt_labels is None:
        return ["WT"]
    if isinstance(wt_labels, str):
        return [wt_labels]
    return list(wt_labels)


def _build_wt_mask(labels: pd.Series, wt_labels: Optional[Union[str, List[str]]]) -> pd.Series:
    """Boolean mask of cells whose stripped label exactly matches a WT/reference label."""
    wt_labels = _normalize_wt_labels(wt_labels)
    return labels.astype(str).str.strip().isin(wt_labels)


def _is_wt_label(label: str, wt_labels: Optional[Union[str, List[str]]]) -> bool:
    """Return True if ``label`` (stripped) exactly matches a WT/reference label."""
    wt_labels = _normalize_wt_labels(wt_labels)
    return str(label).strip() in wt_labels


def test_variant_informativeness(
    df_AF: pd.DataFrame,
    df_ALT: pd.DataFrame,
    df_REF: pd.DataFrame,
    grouped_by_chr_arm: dict,
    output_dir: str,
    p_threshold: float = 0.01,
    baf_deviation_threshold: float = 0.2,
    min_baf_shift_vs_wt: float = 0.10,
    require_shift_vs_wt: bool = True,
    require_clone_imbalance: bool = True,
    require_allele_dropout: bool = True,
    max_cell_minor_fraction: float = 0.10,
    min_dropout_cell_fraction: float = 0.60,
    min_cell_depth_for_dropout: int = 5,
    max_minor_allele_fraction: float = 0.10,
    # Primary direct path: the same clone-level ALT/REF means displayed in the
    # plot. A marker with either allele near zero is called before any
    # Wilcoxon/AF-IQR/dropout criterion is considered.
    enable_direct_near_zero_allele_call: bool = True,
    direct_near_zero_max_mean_allele_reads: float = 2.0,
    direct_near_zero_max_minor_allele_fraction: float = 0.10,
    # Near-zero evidence must have substantial aggregate support before a
    # monoallelic-looking allele is interpreted as LOH. These stricter defaults
    # retain high-depth four-cell chr5q LOH calls, while rejecting low-DP
    # pseudo-monoallelic calls such as 62 total reads across six cells.
    # High-confidence direct near-zero calls: sufficient evidence to stand alone.
    direct_near_zero_min_mean_total_depth: float = 15.0,
    direct_near_zero_min_total_clone_reads: int = 100,
    direct_near_zero_min_depth_qualified_cells: int = 4,
    # Low-depth near-zero candidates: retain plausible calls such as ~10 reads
    # per cell across five cells, but promote them only after independent
    # same-arm LOH corroboration.
    enable_low_depth_near_zero_rescue: bool = True,
    low_depth_near_zero_min_mean_total_depth: float = 5.0,
    low_depth_near_zero_min_total_clone_reads: int = 20,
    low_depth_near_zero_min_depth_qualified_cells: int = 3,
    low_depth_near_zero_max_binom_p: float = 1e-3,
    low_depth_near_zero_confidence_level: float = 0.95,
    low_depth_near_zero_max_depleted_allele_upper_bound: float = 0.15,
    low_depth_near_zero_min_anchor_variants: int = 2,
    low_depth_near_zero_min_anchor_amplicons: int = 2,
    # A low-depth candidate must also match the phase-invariant pooled-allele
    # pattern of the established LOH arm, not merely be near-zero in isolation.
    low_depth_near_zero_max_minor_fraction_distance: float = 0.15,
    low_depth_near_zero_max_phase_invariant_imbalance_distance: float = 0.20,
    low_depth_near_zero_require_independent_amplicons: bool = True,
    direct_near_zero_require_wt_clean_het: bool = True,
    # Second decision path: strict statistical fallback for variants that
    # are not directly monoallelic by pooled clone ALT/REF reads.
    enable_statistical_fallback: bool = True,
    # Read-weighted depletion settings. These use the same pooled read signal
    # displayed in the ALT/REF panels, avoiding disagreements between the plot
    # and the informativeness call.
    max_pooled_minor_allele_fraction: float = 0.25,
    min_weighted_dropout_read_fraction: float = 0.50,
    min_clone_total_reads: int = 30,
    # Direct read-evidence rescue. It is intentionally independent of the
    # rank-sum p-value and clone size, but requires strong pooled depletion.
    enable_dropout_rescue: bool = False,
    dropout_rescue_max_clone_cells: Optional[int] = None,
    dropout_rescue_min_baf_shift_vs_wt: float = 0.10,
    dropout_rescue_max_af_iqr: float = 0.30,
    dropout_rescue_require_af_iqr: bool = False,
    dropout_rescue_min_wt_minor_allele_fraction: float = 0.25,
    dropout_rescue_max_clone_minor_allele_fraction: float = 0.20,
    dropout_rescue_max_cell_minor_fraction: float = 0.20,
    dropout_rescue_min_cell_fraction: float = 0.50,
    depth_ratio_deletion: float = 0.7,
    depth_ratio_amplification: float = 1.3,
    max_af_iqr: float = 0.3,
    min_cells: int = 5,
    wt_labels: Optional[Union[str, List[str]]] = None,
    variant_to_amplicon=None,
    require_amplicon_mapping_for_corroboration: bool = True,
    germline_variants=None,
    enable_arm_rescue: bool = False,
    arm_rescue_p_threshold: float = 0.10,
    arm_rescue_min_baf_shift_vs_wt: float = 0.15,
    arm_rescue_max_af_iqr: float = 0.50,
    arm_rescue_min_unique_amplicons: int = 2,
    # Final conservative rescue: an uncalled marker may be promoted only when
    # its clone-level allelic depletion closely matches validated LOH anchors
    # on the same chromosome arm, from independent amplicons.
    enable_loh_arm_pattern_rescue: bool = True,
    loh_arm_pattern_min_anchor_variants: int = 2,
    loh_arm_pattern_min_anchor_amplicons: int = 2,
    loh_arm_pattern_min_baf_shift_vs_wt: float = 0.08,
    loh_arm_pattern_max_af_iqr: float = 0.45,
    loh_arm_pattern_max_minor_allele_fraction: float = 0.30,
    loh_arm_pattern_max_minor_fraction_distance: float = 0.18,
    loh_arm_pattern_max_phase_invariant_imbalance_distance: float = 0.15,
    loh_arm_pattern_max_depth_ratio_distance: float = 0.35,
    loh_arm_pattern_require_independent_amplicons: bool = True,
    # Cross-Hem rescue: promote an otherwise uncalled Hem clone only when the
    # same germline marker is informative in another Hem clone and the two
    # clone-vs-WT AF shifts have opposite sign. This is intentionally stricter
    # than a simple opposite-AF rule: the candidate still needs a significant,
    # consistent, allele-depleted signal.
    enable_hem_antiphase_rescue: bool = True,
    hem_label_token: str = "Hem",
    hem_antiphase_p_threshold: Optional[float] = None,
    hem_antiphase_min_baf_shift_vs_wt: float = 0.10,
    hem_antiphase_require_allele_dropout: bool = True,
    hem_antiphase_require_statistical_cell_eligibility: bool = False,
    verbose: bool = True,
) -> Dict[str, pd.DataFrame]:
    """Identify germline markers informative for clone-level allelic events.

    For every chromosome arm in ``grouped_by_chr_arm`` and every non-WT clone,
    each variant is compared with the WT/reference cells and classified by a
    hierarchy of decision paths (see Notes). One row per (variant, clone) is
    produced, with all intermediate statistics and boolean gates as columns.

    Parameters
    ----------
    df_AF : pd.DataFrame
        Cells x variants allele frequencies in percent (0-100) plus a ``label``
        column with clone labels. Index = cell barcodes.
    df_ALT, df_REF : pd.DataFrame
        Cells x variants ALT and REF read counts. Re-indexed to ``df_AF``'s
        cells; a ``label`` column, if present, is dropped. Variants missing
        from either table are handled with AF-only statistics.
    grouped_by_chr_arm : dict
        ``{arm_name: [variant_id, ...]}`` (e.g. from
        ``group_informative_variants_by_chr_arm``). Arms are processed in dict
        order; variants absent from ``df_AF`` are ignored.
    output_dir : str
        Directory for the per-arm CSV files (created only if any result exists).
    p_threshold : float, default 0.01
        Wilcoxon rank-sum p-value threshold of the statistical fallback.
    baf_deviation_threshold : float, default 0.2
        ``clone_imbalanced`` requires ``|mean_af_clone/100 - 0.5|`` above this.
    min_baf_shift_vs_wt : float, default 0.10
        ``has_shift_vs_wt`` requires ``|mean_af_clone - mean_af_wt|/100 >=`` this.
    require_shift_vs_wt, require_clone_imbalance : bool, default True
        Only shape the diagnostic ``has_imbalance`` column. The statistical
        fallback always requires both ``has_shift_vs_wt`` and
        ``clone_imbalanced`` regardless of these flags.
    require_allele_dropout : bool, default True
        Statistical fallback additionally requires ``allele_dropout``.
    max_cell_minor_fraction : float, default 0.10
        A cell counts as a strict dropout cell when its minor-allele read
        fraction is below this value.
    min_dropout_cell_fraction : float, default 0.60
        Accepted but not used.
    min_cell_depth_for_dropout : int, default 5
        Per-cell depth (ALT+REF) required for a cell to enter the statistical
        paths, and for counting "depth-qualified" cells in the direct paths.
    max_minor_allele_fraction : float, default 0.10
        Accepted but not used.
    enable_direct_near_zero_allele_call : bool, default True
        Enable the high-confidence direct near-zero path (and, with it, the
        low-depth candidate path).
    direct_near_zero_max_mean_allele_reads : float, default 2.0
        Max clone-wide mean of the lower of ALT/REF reads per cell.
    direct_near_zero_max_minor_allele_fraction : float, default 0.10
        Max pooled clone minor-allele read fraction for the near-zero pattern.
    direct_near_zero_min_mean_total_depth : float, default 15.0
        High-confidence: min clone mean depth. Also used as the anchor and
        candidate depth floor of the LOH-arm pattern rescue.
    direct_near_zero_min_total_clone_reads : int, default 100
        High-confidence: min pooled clone reads.
    direct_near_zero_min_depth_qualified_cells : int, default 4
        High-confidence: min clone cells with depth >= ``min_cell_depth_for_dropout``.
    enable_low_depth_near_zero_rescue : bool, default True
        Enable low-depth near-zero candidates and their same-arm corroboration.
    low_depth_near_zero_min_mean_total_depth : float, default 5.0
    low_depth_near_zero_min_total_clone_reads : int, default 20
    low_depth_near_zero_min_depth_qualified_cells : int, default 3
        Basic depth support a low-depth candidate must reach.
    low_depth_near_zero_max_binom_p : float, default 1e-3
        Max one-sided exact binomial p-value of the depleted-allele read count
        under the WT allele fraction.
    low_depth_near_zero_confidence_level : float, default 0.95
        Confidence level of the one-sided upper bound on the depleted-allele
        fraction (``beta.ppf(level, x + 1, n - x)``).
    low_depth_near_zero_max_depleted_allele_upper_bound : float, default 0.15
        Max allowed value of that upper bound.
    low_depth_near_zero_min_anchor_variants : int, default 2
    low_depth_near_zero_min_anchor_amplicons : int, default 2
        Minimum high-confidence anchors (variants / distinct amplicons) on the
        same arm and clone.
    low_depth_near_zero_max_minor_fraction_distance : float, default 0.15
    low_depth_near_zero_max_phase_invariant_imbalance_distance : float, default 0.20
        Max distance of the candidate from the anchors' median pooled minor
        fraction / median ``|ALT fraction - 0.5|``.
    low_depth_near_zero_require_independent_amplicons : bool, default True
        Anchors must come from amplicons other than the candidate's.
    direct_near_zero_require_wt_clean_het : bool, default True
        Without a ``germline_variants`` panel, direct paths require the WT to be
        a clean heterozygote (see Notes).
    enable_statistical_fallback : bool, default True
        Enable the strict Wilcoxon-based fallback path.
    max_pooled_minor_allele_fraction : float, default 0.25
        ``pooled_depletion``: clone pooled minor-allele fraction <= this.
    min_weighted_dropout_read_fraction : float, default 0.50
        ``weighted_dropout``: share of clone reads in strict dropout cells >= this.
    min_clone_total_reads : int, default 30
        Read floor for the pooled-read rescue and the LOH-arm pattern rescue.
    enable_dropout_rescue : bool, default False
        Enable the optional pooled-read depletion rescue.
    dropout_rescue_max_clone_cells : int, optional
        If set, the pooled-read rescue applies only to clones with at most this
        many statistically usable cells. Must be >= 1.
    dropout_rescue_min_baf_shift_vs_wt : float, default 0.10
    dropout_rescue_max_af_iqr : float, default 0.30
    dropout_rescue_require_af_iqr : bool, default False
    dropout_rescue_max_clone_minor_allele_fraction : float, default 0.20
    dropout_rescue_max_cell_minor_fraction : float, default 0.20
    dropout_rescue_min_cell_fraction : float, default 0.50
        Thresholds of the pooled-read rescue (shift, optional IQR gate, strong
        pooled depletion or broad depth-weighted near-dropout).
    dropout_rescue_min_wt_minor_allele_fraction : float, default 0.25
        WT pooled minor fraction required for the WT to count as a clean
        heterozygote (used by several paths, not only the rescue).
    depth_ratio_deletion, depth_ratio_amplification : float, default 0.7, 1.3
        Clone/WT mean-depth ratio cut-offs for ``event_type``.
    max_af_iqr : float, default 0.3
        ``is_consistent``: clone AF IQR (as fraction) <= this.
    min_cells : int, default 5
        Minimum depth-qualified clone *and* WT cells for statistical paths.
    wt_labels : str or list of str, optional
        Exact WT/reference label(s); ``None`` means ``['WT']``.
    variant_to_amplicon : dict or pd.DataFrame, optional
        Variant -> amplicon map (DataFrame with variant/id/mutation and
        amplicon/amplicon_id/amp columns). Variant IDs are canonicalised.
    require_amplicon_mapping_for_corroboration : bool, default True
        Skip the low-depth and LOH-arm corroboration when no mapping key
        overlaps ``df_AF`` (and independent amplicons are required).
    germline_variants : iterable of str, optional
        Germline whitelist used instead of the WT clean-het check as the
        eligibility reference for the direct and Hem paths.
    enable_arm_rescue : bool, default False
    arm_rescue_p_threshold : float, default 0.10
    arm_rescue_min_baf_shift_vs_wt : float, default 0.15
    arm_rescue_max_af_iqr : float, default 0.50
    arm_rescue_min_unique_amplicons : int, default 2
        Optional generic arm rescue of partial shifts supported by several
        distinct amplicons in the same clone and arm.
    enable_loh_arm_pattern_rescue : bool, default True
    loh_arm_pattern_min_anchor_variants : int, default 2
    loh_arm_pattern_min_anchor_amplicons : int, default 2
    loh_arm_pattern_min_baf_shift_vs_wt : float, default 0.08
    loh_arm_pattern_max_af_iqr : float, default 0.45
    loh_arm_pattern_max_minor_allele_fraction : float, default 0.30
    loh_arm_pattern_max_minor_fraction_distance : float, default 0.18
    loh_arm_pattern_max_phase_invariant_imbalance_distance : float, default 0.15
    loh_arm_pattern_max_depth_ratio_distance : float, default 0.35
    loh_arm_pattern_require_independent_amplicons : bool, default True
        LOH-arm pattern rescue settings (see Notes).
    enable_hem_antiphase_rescue : bool, default True
    hem_label_token : str, default "Hem"
        Case-insensitive substring identifying Hem clones.
    hem_antiphase_p_threshold : float, optional
        Extra p-value bound for Hem candidates (default ``p_threshold``).
    hem_antiphase_min_baf_shift_vs_wt : float, default 0.10
    hem_antiphase_require_allele_dropout : bool, default True
    hem_antiphase_require_statistical_cell_eligibility : bool, default False
        Cross-Hem anti-phase rescue settings (see Notes).
    verbose : bool, default True
        Print parameters, per-arm counts and a summary.

    Returns
    -------
    dict of str -> pd.DataFrame
        ``{arm_name: results}`` for arms with at least one comparison; ``{}``
        if none. Key columns: ``variant``, ``label``, ``amplicon``,
        ``mean_af_clone``, ``mean_af_wt``, ``delta_af``, ``wilcoxon_p``,
        ``depth_ratio``, one ``informative_*`` flag per path, ``informative``
        (OR of all paths), ``informativeness_tier``, ``classification_reason``
        and ``event_type`` (``DELETION``/``AMPLIFICATION``/``cnLOH``/
        ``UNKNOWN`` for informative rows, ``""`` otherwise). Each table is also
        written to ``output_dir/wilcoxon_results_<arm>_enhanced.csv``.

    Raises
    ------
    ValueError
        If ``dropout_rescue_max_clone_cells`` < 1, or no cell carries a WT label.
    KeyError
        If ``df_AF`` has no ``label`` column.

    Notes
    -----
    Per (variant, clone), two sets of aggregates are computed: *plot-aligned*
    ones over all clone cells (as shown in ALT/REF plots) and *statistical*
    ones over cells with depth >= ``min_cell_depth_for_dropout``. Clone and WT
    mean AF are read-weighted (``100 * sum(ALT) / sum(ALT+REF)``). The WT is a
    *clean heterozygote* when its pooled minor fraction is >=
    ``dropout_rescue_min_wt_minor_allele_fraction`` and its mean AF is in
    [30, 70]. Decision hierarchy:

    1. High-confidence direct near-zero: plot-aligned min(mean ALT, mean REF)
       <= 2.0 reads and pooled minor fraction <= 0.10, with mean depth >= 15,
       total reads >= 100 and >= 4 depth-qualified cells, and the reference
       check passes (whitelist membership or WT clean het). Stand-alone call.
    2. Low-depth near-zero candidate: same pattern with only basic depth
       support (5 / 20 / 3), exact binomial p <= 1e-3 and depleted-allele
       upper bound <= 0.15. Promoted only if, in the same clone and arm, at
       least 2 anchors (paths 1, 3 or the pooled-read rescue) from at least 2
       other amplicons exist and the candidate matches their median minor
       fraction (<= 0.15) and phase-invariant imbalance (<= 0.20).
    3. Strict statistical fallback (not path 1): both groups have >=
       ``min_cells`` cells, Wilcoxon p < ``p_threshold``, shift vs WT >= 0.10,
       clone imbalanced (> 0.2 from 0.5), AF IQR <= 0.3, and allele dropout
       (pooled minor <= 0.25 and >= 50% of reads in cells with minor < 0.10).
    4. LOH-arm pattern rescue: an uncalled row (w.r.t. paths 1, 3 and the
       pooled-read rescue) is promoted when >= 2 anchors from >= 2 other
       amplicons exist in the clone/arm and it passes eligibility (shift >=
       0.08, IQR <= 0.45, mean depth >= 15, reads >= 30, minor <= 0.30, depth
       ratio <= 1.3) and similarity (minor-fraction distance <= 0.18, imbalance
       distance <= 0.15, depth-ratio distance <= 0.35) to the anchor medians.
    5. Cross-Hem anti-phase rescue: for a variant, an uncalled Hem clone with
       significant, consistent, allele-depleted evidence and shift >= 0.10 is
       promoted when another Hem clone is already informative with an
       opposite-signed ``delta_af`` (anchor = largest ``|delta_af|``).

    The pooled-read rescue and the generic arm rescue are disabled by default.
    """

    wt_labels = _normalize_wt_labels(wt_labels)

    def vprint(msg: str) -> None:
        if verbose:
            print(msg)
            sys.stdout.flush()

    def _canonical_variant_id(value):
        """Normalize benign formatting differences in variant identifiers."""
        if pd.isna(value):
            return ""
        token = str(value).strip().replace(" ", "")
        parts = token.split(":")
        if not parts:
            return token
        chrom = parts[0]
        if chrom and not chrom.lower().startswith("chr"):
            chrom = f"chr{chrom}"
        chrom = "chr" + chrom[3:] if chrom.lower().startswith("chr") else chrom
        if len(parts) >= 3:
            parts = [chrom, parts[1], *[part.upper() for part in parts[2:]]]
        else:
            parts = [chrom, *parts[1:]]
        return ":".join(parts)

    def _coerce_map(x):
        if x is None:
            return {}
        if isinstance(x, dict):
            raw_items = x.items()
        elif isinstance(x, pd.DataFrame):
            cols = {str(c).lower(): c for c in x.columns}
            vcol = next((cols[k] for k in ("variant", "id", "mutation") if k in cols), None)
            acol = next((cols[k] for k in ("amplicon", "amplicon_id", "amp") if k in cols), None)
            if vcol is None or acol is None:
                return {}
            raw_items = zip(x[vcol], x[acol])
        else:
            return {}

        result = {}
        for raw_variant, raw_amplicon in raw_items:
            if pd.isna(raw_variant) or pd.isna(raw_amplicon):
                continue
            variant_id = _canonical_variant_id(raw_variant)
            amplicon_id = str(raw_amplicon).strip()
            if variant_id and amplicon_id and amplicon_id.lower() != "nan":
                result[variant_id] = amplicon_id
        return result

    def _event_type(depth_ratio):
        if pd.isna(depth_ratio):
            return "UNKNOWN"
        if depth_ratio < depth_ratio_deletion:
            return "DELETION"
        if depth_ratio > depth_ratio_amplification:
            return "AMPLIFICATION"
        return "cnLOH"

    variant_to_amplicon = _coerce_map(variant_to_amplicon)

    def _amplicon_for_variant(variant):
        return variant_to_amplicon.get(_canonical_variant_id(variant), pd.NA)

    germline_variant_set = {
        str(v) for v in ([] if germline_variants is None else germline_variants)
        if pd.notna(v)
    }
    if dropout_rescue_max_clone_cells is not None and int(dropout_rescue_max_clone_cells) < 1:
        raise ValueError("dropout_rescue_max_clone_cells must be None or >= 1.")

    vprint("\n" + "=" * 60)
    vprint("VARIANT INFORMATIVENESS TESTING")
    vprint("High-confidence direct evidence, low-depth corroboration, then strict statistical fallback")
    vprint("=" * 60)
    vprint(f"  Minimum qualified cell depth:          {min_cell_depth_for_dropout} (statistical paths only)")
    vprint(f"  Min clone total reads:                 {min_clone_total_reads}")
    vprint(f"  Strict p-value threshold:              {p_threshold}")
    vprint(f"  Min AF shift versus WT:                {min_baf_shift_vs_wt}")
    vprint(f"  Direct near-zero ALT/REF call:         {enable_direct_near_zero_allele_call}")
    vprint(f"  Direct near-zero allele mean max:      {direct_near_zero_max_mean_allele_reads}")
    vprint(f"  Direct near-zero minor fraction max:   {direct_near_zero_max_minor_allele_fraction}")
    vprint(f"  Direct high-confidence mean depth min: {direct_near_zero_min_mean_total_depth}")
    vprint(f"  Direct high-confidence total reads min:{direct_near_zero_min_total_clone_reads}")
    vprint(f"  Direct high-confidence qualified cells:{direct_near_zero_min_depth_qualified_cells}")
    vprint(f"  Low-depth candidate rescue:            {enable_low_depth_near_zero_rescue}")
    vprint(f"  Low-depth basic mean depth min:        {low_depth_near_zero_min_mean_total_depth}")
    vprint(f"  Low-depth basic total reads min:       {low_depth_near_zero_min_total_clone_reads}")
    vprint(f"  Low-depth basic qualified cells min:   {low_depth_near_zero_min_depth_qualified_cells}")
    vprint(f"  Low-depth exact-binomial p max:        {low_depth_near_zero_max_binom_p}")
    vprint(f"  Low-depth depleted allele upper bound: {low_depth_near_zero_max_depleted_allele_upper_bound} at {low_depth_near_zero_confidence_level:.0%} confidence")
    vprint(f"  Low-depth anchor requirement:          {low_depth_near_zero_min_anchor_variants} variants / {low_depth_near_zero_min_anchor_amplicons} amplicons")
    vprint(f"  Low-depth max minor-fraction distance:  {low_depth_near_zero_max_minor_fraction_distance}")
    vprint(f"  Low-depth max imbalance distance:       {low_depth_near_zero_max_phase_invariant_imbalance_distance}")
    vprint(f"  Germline whitelist size:               {len(germline_variant_set)}")
    vprint(f"  Statistical fallback enabled:          {enable_statistical_fallback}")
    vprint(f"  Fallback requires clone imbalance:     {require_clone_imbalance}")
    vprint(f"  Max pooled minor allele fraction:      {max_pooled_minor_allele_fraction}")
    vprint(f"  Min depth-weighted dropout read share: {min_weighted_dropout_read_fraction}")
    vprint(f"  Optional pooled-read rescue:           {enable_dropout_rescue} (disabled by default)")
    vprint(f"  LOH-arm pattern rescue:                 {enable_loh_arm_pattern_rescue}")
    vprint(f"  LOH-arm anchors required:               {loh_arm_pattern_min_anchor_variants} variants / {loh_arm_pattern_min_anchor_amplicons} amplicons")
    vprint(f"  LOH-arm independent amplicons required: {loh_arm_pattern_require_independent_amplicons}")
    vprint(f"  Cross-Hem anti-phase rescue:            {enable_hem_antiphase_rescue}")
    vprint(f"  Cross-Hem token:                         {hem_label_token!r}")
    vprint(f"  Cross-Hem p-value threshold:             {hem_antiphase_p_threshold if hem_antiphase_p_threshold is not None else p_threshold}")

    if "label" not in df_AF.columns:
        raise KeyError("df_AF must contain a 'label' column.")

    df_AF = df_AF.copy()
    df_AF.index = df_AF.index.astype(str)
    df_AF.columns = df_AF.columns.astype(str)
    df_AF["label"] = df_AF["label"].astype(str).str.strip()

    df_ALT = df_ALT.copy()
    df_REF = df_REF.copy()
    df_ALT.index = df_ALT.index.astype(str)
    df_REF.index = df_REF.index.astype(str)
    df_ALT.columns = df_ALT.columns.astype(str)
    df_REF.columns = df_REF.columns.astype(str)
    df_ALT = df_ALT.reindex(df_AF.index).drop(columns=["label"], errors="ignore")
    df_REF = df_REF.reindex(df_AF.index).drop(columns=["label"], errors="ignore")

    wt_mask = _build_wt_mask(df_AF["label"], wt_labels)
    if int(wt_mask.sum()) == 0:
        raise ValueError(
            f"No WT/reference cells found with exact label(s): {wt_labels}. "
            f"Available: {sorted(df_AF['label'].unique().tolist())}"
        )

    clone_labels = sorted(
        lab for lab in df_AF["label"].unique() if not _is_wt_label(lab, wt_labels)
    )
    all_variant_cols = [c for c in df_AF.columns if c != "label"]
    canonical_variant_universe = {_canonical_variant_id(v) for v in all_variant_cols}
    amplicon_mapping_overlap = int(len(canonical_variant_universe.intersection(variant_to_amplicon)))
    amplicon_mapping_available = bool(amplicon_mapping_overlap > 0)
    vprint(f"  Variant-to-amplicon mappings supplied: {len(variant_to_amplicon)}")
    vprint(f"  Variant-to-amplicon overlaps in df_AF: {amplicon_mapping_overlap}")
    if variant_to_amplicon and not amplicon_mapping_available:
        vprint("  WARNING: no amplicon-map keys overlap df_AF variants; low-depth corroboration will be disabled.")
    out: Dict[str, pd.DataFrame] = {}

    for chr_idx, (chr_name, variants) in enumerate(grouped_by_chr_arm.items(), 1):
        variants_in_df = [str(v) for v in variants if str(v) in all_variant_cols]
        if not variants_in_df:
            continue
        vprint(f"  [{chr_idx}/{len(grouped_by_chr_arm)}] {chr_name}: {len(variants_in_df)} variants")
        rows = []

        for variant in variants_in_df:
            has_counts = variant in df_ALT.columns and variant in df_REF.columns
            af_series = pd.to_numeric(df_AF[variant], errors="coerce")

            if has_counts:
                alt_series = pd.to_numeric(df_ALT[variant], errors="coerce").fillna(0.0)
                ref_series = pd.to_numeric(df_REF[variant], errors="coerce").fillna(0.0)
                depth_series = alt_series + ref_series
                depth_ok = depth_series.ge(float(min_cell_depth_for_dropout))
            else:
                alt_series = ref_series = depth_series = None
                depth_ok = af_series.notna()

            # Raw WT aggregates are intentionally calculated without the
            # statistical per-cell depth filter. They serve as the expected
            # allele fraction for the low-depth pooled evidence test.
            if has_counts:
                plot_wt_alt_all = alt_series.loc[wt_mask]
                plot_wt_ref_all = ref_series.loc[wt_mask]
                plot_wt_total_reads_all = float((plot_wt_alt_all + plot_wt_ref_all).sum())
                plot_wt_alt_fraction_all = float(
                    plot_wt_alt_all.sum() / max(plot_wt_total_reads_all, np.finfo(float).eps)
                )
                plot_wt_ref_fraction_all = float(
                    plot_wt_ref_all.sum() / max(plot_wt_total_reads_all, np.finfo(float).eps)
                )
            else:
                plot_wt_total_reads_all = np.nan
                plot_wt_alt_fraction_all = np.nan
                plot_wt_ref_fraction_all = np.nan

            for lab in clone_labels:
                # Match the ALT/REF panels: these are clone-wide means over all
                # cells of the displayed clone, rather than only depth-qualified
                # AF observations used by the statistical test below.
                plot_clone_mask = df_AF["label"].eq(lab)
                if has_counts:
                    plot_clone_alt = alt_series.loc[plot_clone_mask]
                    plot_clone_ref = ref_series.loc[plot_clone_mask]
                    plot_clone_depth = plot_clone_alt + plot_clone_ref
                    plot_clone_total_reads = float(plot_clone_depth.sum())
                    plot_mean_alt_clone = float(plot_clone_alt.mean())
                    plot_mean_ref_clone = float(plot_clone_ref.mean())
                    plot_mean_depth_clone = float(plot_clone_depth.mean())
                    plot_minor_allele_fraction = float(
                        min(float(plot_clone_alt.sum()), float(plot_clone_ref.sum()))
                        / max(plot_clone_total_reads, np.finfo(float).eps)
                    )
                    plot_depth_qualified_cell_count = int(
                        plot_clone_depth.ge(float(min_cell_depth_for_dropout)).sum()
                    )
                else:
                    plot_clone_total_reads = np.nan
                    plot_mean_alt_clone = np.nan
                    plot_mean_ref_clone = np.nan
                    plot_mean_depth_clone = np.nan
                    plot_minor_allele_fraction = np.nan
                    plot_depth_qualified_cell_count = 0

                clone_mask = df_AF["label"].eq(lab) & depth_ok & af_series.notna()
                wt_valid_mask = wt_mask & depth_ok & af_series.notna()
                clone_af = af_series.loc[clone_mask].dropna()
                wt_af = af_series.loc[wt_valid_mask].dropna()
                n_cells = int(len(clone_af))
                n_wt_cells = int(len(wt_af))
                # The per-cell depth filter is required for the Wilcoxon/AF-IQR
                # branch, but must not suppress a direct clone-wide ALT/REF
                # depletion call.  The latter uses the plot-aligned aggregates
                # computed above and has independent aggregate and
                # depth-qualified-cell safeguards.
                statistical_test_cell_eligible = bool(
                    n_cells >= int(min_cells)
                    and n_wt_cells >= int(min_cells)
                )

                if has_counts:
                    clone_alt = alt_series.loc[clone_mask]
                    clone_ref = ref_series.loc[clone_mask]
                    wt_alt = alt_series.loc[wt_valid_mask]
                    wt_ref = ref_series.loc[wt_valid_mask]
                    clone_depth = clone_alt + clone_ref
                    wt_depth = wt_alt + wt_ref
                    clone_total_reads = float(clone_depth.sum())
                    wt_total_reads = float(wt_depth.sum())
                    clone_minor_reads = np.minimum(
                        clone_alt.to_numpy(dtype=float), clone_ref.to_numpy(dtype=float)
                    )
                    clone_depth_arr = clone_depth.to_numpy(dtype=float)
                    cell_minor_fraction = clone_minor_reads / np.maximum(clone_depth_arr, np.finfo(float).eps)

                    mean_af_clone = (
                        100.0 * float(clone_alt.sum()) / clone_total_reads
                        if clone_total_reads > 0 else float(clone_af.mean())
                    )
                    mean_af_wt = (
                        100.0 * float(wt_alt.sum()) / wt_total_reads
                        if wt_total_reads > 0 else float(wt_af.mean())
                    )
                    mean_alt_clone = float(clone_alt.mean())
                    mean_ref_clone = float(clone_ref.mean())
                    mean_depth_clone = float(clone_depth.mean())
                    mean_depth_wt = float(wt_depth.mean())
                    clone_minor_allele_fraction = float(
                        clone_minor_reads.sum() / max(clone_total_reads, np.finfo(float).eps)
                    )
                    wt_minor_allele_fraction = float(
                        min(float(wt_alt.sum()), float(wt_ref.sum()))
                        / max(wt_total_reads, np.finfo(float).eps)
                    )
                    # Raw cell fraction remains diagnostic only. The weighted
                    # form is the actual criterion, so shallow cells cannot
                    # outvote high-depth evidence.
                    clone_dropout_fraction = float(
                        (cell_minor_fraction < float(max_cell_minor_fraction)).mean()
                    )
                    clone_near_dropout_fraction = float(
                        (cell_minor_fraction <= float(dropout_rescue_max_cell_minor_fraction)).mean()
                    )
                    strict_low_minor = cell_minor_fraction < float(max_cell_minor_fraction)
                    rescue_low_minor = cell_minor_fraction <= float(dropout_rescue_max_cell_minor_fraction)
                    weighted_dropout_fraction = float(
                        clone_depth_arr[strict_low_minor].sum() / max(clone_total_reads, np.finfo(float).eps)
                    )
                    weighted_near_dropout_fraction = float(
                        clone_depth_arr[rescue_low_minor].sum() / max(clone_total_reads, np.finfo(float).eps)
                    )
                else:
                    clone_total_reads = wt_total_reads = np.nan
                    mean_af_clone = float(clone_af.mean())
                    mean_af_wt = float(wt_af.mean())
                    mean_alt_clone = mean_ref_clone = np.nan
                    mean_depth_clone = mean_depth_wt = np.nan
                    clone_minor_allele_fraction = min(mean_af_clone, 100.0 - mean_af_clone) / 100.0
                    wt_minor_allele_fraction = min(mean_af_wt, 100.0 - mean_af_wt) / 100.0
                    clone_dropout_fraction = clone_near_dropout_fraction = np.nan
                    weighted_dropout_fraction = weighted_near_dropout_fraction = np.nan

                delta_af = mean_af_clone - mean_af_wt
                baf_shift_vs_wt = abs(delta_af) / 100.0
                baf_deviation = abs(mean_af_clone / 100.0 - 0.5)
                clone_imbalanced = bool(baf_deviation > baf_deviation_threshold)
                has_shift_vs_wt = bool(baf_shift_vs_wt >= float(min_baf_shift_vs_wt))
                has_imbalance = has_shift_vs_wt if require_shift_vs_wt else clone_imbalanced
                if require_clone_imbalance:
                    has_imbalance = bool(has_imbalance and clone_imbalanced)

                af_iqr = float((clone_af / 100.0).quantile(0.75) - (clone_af / 100.0).quantile(0.25))
                wt_af_iqr = float((wt_af / 100.0).quantile(0.75) - (wt_af / 100.0).quantile(0.25))
                is_consistent = bool(af_iqr <= float(max_af_iqr))
                try:
                    stat, p = ranksums(clone_af, wt_af)
                    stat, p = float(stat), float(p)
                except Exception:
                    stat, p = np.nan, 1.0
                is_significant = bool(p < float(p_threshold))

                depth_ratio = (
                    mean_depth_clone / mean_depth_wt
                    if pd.notna(mean_depth_clone) and pd.notna(mean_depth_wt) and mean_depth_wt > 0
                    else np.nan
                )

                pooled_depletion = bool(
                    clone_minor_allele_fraction <= float(max_pooled_minor_allele_fraction)
                )
                weighted_dropout = bool(
                    pd.notna(weighted_dropout_fraction)
                    and weighted_dropout_fraction >= float(min_weighted_dropout_read_fraction)
                )
                # Strict path requires both a low pooled minor allele and
                # read-weighted per-cell support. This removes calls where a
                # majority of shallow cells are monoallelic but the aggregate
                # read counts still show both alleles.
                allele_dropout = bool(pooled_depletion and weighted_dropout)
                # Retained as a compatibility placeholder until the direct
                # near-zero decision is evaluated below.
                informative_primary = False

                wt_is_clean_het = bool(
                    wt_minor_allele_fraction >= float(dropout_rescue_min_wt_minor_allele_fraction)
                    and 30.0 <= mean_af_wt <= 70.0
                )

                # ── Direct near-zero paths: high confidence vs low depth ───
                # Both paths use exactly the clone-wide ALT/REF aggregates shown
                # in the plot.  High-confidence calls can stand alone.  Lower
                # coverage candidates are retained, but later require an
                # independent, same-arm corroboration layer.
                plot_clone_min_mean_allele_reads = (
                    min(plot_mean_alt_clone, plot_mean_ref_clone)
                    if pd.notna(plot_mean_alt_clone) and pd.notna(plot_mean_ref_clone)
                    else np.nan
                )
                clone_min_mean_allele_reads = plot_clone_min_mean_allele_reads

                if has_counts:
                    plot_clone_alt_sum = float(plot_clone_alt.sum())
                    plot_clone_ref_sum = float(plot_clone_ref.sum())
                    if plot_clone_alt_sum <= plot_clone_ref_sum:
                        plot_depleted_allele = "ALT"
                        plot_depleted_allele_reads = int(round(plot_clone_alt_sum))
                        plot_expected_depleted_allele_fraction_wt = plot_wt_alt_fraction_all
                    else:
                        plot_depleted_allele = "REF"
                        plot_depleted_allele_reads = int(round(plot_clone_ref_sum))
                        plot_expected_depleted_allele_fraction_wt = plot_wt_ref_fraction_all
                    plot_total_reads_int = int(round(plot_clone_total_reads))
                else:
                    plot_clone_alt_sum = np.nan
                    plot_clone_ref_sum = np.nan
                    plot_depleted_allele = pd.NA
                    plot_depleted_allele_reads = np.nan
                    plot_expected_depleted_allele_fraction_wt = np.nan
                    plot_total_reads_int = 0

                direct_near_zero_pattern = bool(
                    has_counts
                    and pd.notna(plot_clone_min_mean_allele_reads)
                    and plot_clone_min_mean_allele_reads <= float(direct_near_zero_max_mean_allele_reads)
                    and pd.notna(plot_minor_allele_fraction)
                    and plot_minor_allele_fraction <= float(direct_near_zero_max_minor_allele_fraction)
                )

                # Stand-alone, high-confidence depth support.
                direct_near_zero_mean_depth_ok = bool(
                    pd.notna(plot_mean_depth_clone)
                    and plot_mean_depth_clone >= float(direct_near_zero_min_mean_total_depth)
                )
                direct_near_zero_total_reads_ok = bool(
                    pd.notna(plot_clone_total_reads)
                    and plot_clone_total_reads >= float(direct_near_zero_min_total_clone_reads)
                )
                direct_near_zero_qualified_cells_ok = bool(
                    int(plot_depth_qualified_cell_count)
                    >= int(direct_near_zero_min_depth_qualified_cells)
                )
                direct_near_zero_has_adequate_depth = bool(
                    direct_near_zero_mean_depth_ok
                    and direct_near_zero_total_reads_ok
                    and direct_near_zero_qualified_cells_ok
                )

                # Basic support that permits a low-depth candidate to undergo
                # an additional evidence and regional corroboration test.
                low_depth_near_zero_mean_depth_ok = bool(
                    pd.notna(plot_mean_depth_clone)
                    and plot_mean_depth_clone >= float(low_depth_near_zero_min_mean_total_depth)
                )
                low_depth_near_zero_total_reads_ok = bool(
                    pd.notna(plot_clone_total_reads)
                    and plot_clone_total_reads >= float(low_depth_near_zero_min_total_clone_reads)
                )
                low_depth_near_zero_qualified_cells_ok = bool(
                    int(plot_depth_qualified_cell_count)
                    >= int(low_depth_near_zero_min_depth_qualified_cells)
                )
                low_depth_near_zero_basic_depth_ok = bool(
                    low_depth_near_zero_mean_depth_ok
                    and low_depth_near_zero_total_reads_ok
                    and low_depth_near_zero_qualified_cells_ok
                )

                is_germline_whitelist_variant = bool(
                    germline_variant_set and variant in germline_variant_set
                )
                if germline_variant_set:
                    direct_near_zero_wt_ok = is_germline_whitelist_variant
                    direct_near_zero_reference_source = "germline_whitelist"
                else:
                    direct_near_zero_wt_ok = bool(
                        (not direct_near_zero_require_wt_clean_het) or wt_is_clean_het
                    )
                    direct_near_zero_reference_source = "current_WT_clean_het"

                # Exact pooled evidence: under the WT-derived depleted-allele
                # fraction, how surprising is this few depleted-allele reads?
                # The upper bound prevents a nominally low observed fraction
                # from being accepted when its uncertainty still permits a
                # substantial residual allele fraction.
                low_depth_near_zero_binom_p = np.nan
                low_depth_near_zero_depleted_allele_upper_bound = np.nan
                low_depth_near_zero_exact_evidence_ok = False
                if (
                    has_counts
                    and plot_total_reads_int > 0
                    and pd.notna(plot_expected_depleted_allele_fraction_wt)
                    and 0.0 < float(plot_expected_depleted_allele_fraction_wt) < 1.0
                    and pd.notna(plot_depleted_allele_reads)
                ):
                    _x = int(max(0, min(plot_depleted_allele_reads, plot_total_reads_int)))
                    _n = int(plot_total_reads_int)
                    _p0 = float(plot_expected_depleted_allele_fraction_wt)
                    try:
                        low_depth_near_zero_binom_p = float(
                            binomtest(_x, n=_n, p=_p0, alternative="less").pvalue
                        )
                        _alpha = 1.0 - float(low_depth_near_zero_confidence_level)
                        if _x >= _n:
                            low_depth_near_zero_depleted_allele_upper_bound = 1.0
                        else:
                            low_depth_near_zero_depleted_allele_upper_bound = float(
                                beta.ppf(1.0 - _alpha, _x + 1, _n - _x)
                            )
                        low_depth_near_zero_exact_evidence_ok = bool(
                            low_depth_near_zero_binom_p <= float(low_depth_near_zero_max_binom_p)
                            and low_depth_near_zero_depleted_allele_upper_bound
                            <= float(low_depth_near_zero_max_depleted_allele_upper_bound)
                        )
                    except Exception:
                        pass

                # High-confidence direct calls can stand alone. Low-depth
                # candidates are deliberately not called here; a later pass
                # requires independent same-arm anchors.
                informative_direct_near_zero = bool(
                    enable_direct_near_zero_allele_call
                    and direct_near_zero_pattern
                    and direct_near_zero_has_adequate_depth
                    and direct_near_zero_wt_ok
                )
                low_depth_near_zero_candidate = bool(
                    enable_low_depth_near_zero_rescue
                    and enable_direct_near_zero_allele_call
                    and direct_near_zero_pattern
                    and low_depth_near_zero_basic_depth_ok
                    and not direct_near_zero_has_adequate_depth
                    and direct_near_zero_wt_ok
                    and low_depth_near_zero_exact_evidence_ok
                )
                # Backward-compatible diagnostic: this denotes high-confidence
                # direct depletion only; low-depth candidates receive their own
                # explicit columns and cannot be mistaken for direct calls.
                direct_near_zero_allele_depletion = bool(
                    direct_near_zero_pattern and direct_near_zero_has_adequate_depth
                )

                # Second path: only markers that are not directly monoallelic
                # can pass the conventional statistical rule.  This is the
                # intentionally strict fallback requested by the caller.
                informative_primary = bool(
                    enable_statistical_fallback
                    and statistical_test_cell_eligible
                    and not informative_direct_near_zero
                    and is_significant
                    and has_shift_vs_wt
                    and clone_imbalanced
                    and is_consistent
                    and (allele_dropout or not require_allele_dropout)
                )

                strong_pooled_allele_depletion = bool(
                    clone_minor_allele_fraction <= float(dropout_rescue_max_clone_minor_allele_fraction)
                )
                broad_weighted_near_dropout = bool(
                    pd.notna(weighted_near_dropout_fraction)
                    and weighted_near_dropout_fraction >= float(dropout_rescue_min_cell_fraction)
                    and clone_minor_allele_fraction <= float(max_pooled_minor_allele_fraction)
                )
                dropout_like_allelic_pattern = bool(
                    strong_pooled_allele_depletion or broad_weighted_near_dropout
                )
                rescue_iqr_ok = bool(
                    (not dropout_rescue_require_af_iqr)
                    or af_iqr <= float(dropout_rescue_max_af_iqr)
                )
                has_minimum_read_evidence = bool(
                    (not has_counts)
                    or clone_total_reads >= float(min_clone_total_reads)
                )
                informative_dropout_rescue = bool(
                    enable_dropout_rescue
                    and statistical_test_cell_eligible
                    and not informative_direct_near_zero
                    and not informative_primary
                    and (
                        dropout_rescue_max_clone_cells is None
                        or n_cells <= int(dropout_rescue_max_clone_cells)
                    )
                    and wt_is_clean_het
                    and baf_shift_vs_wt >= float(dropout_rescue_min_baf_shift_vs_wt)
                    and has_minimum_read_evidence
                    and rescue_iqr_ok
                    and dropout_like_allelic_pattern
                )

                arm_rescue_candidate = bool(
                    enable_arm_rescue
                    and statistical_test_cell_eligible
                    and bool(variant_to_amplicon)
                    and p < float(arm_rescue_p_threshold)
                    and baf_shift_vs_wt >= float(arm_rescue_min_baf_shift_vs_wt)
                    and af_iqr <= float(arm_rescue_max_af_iqr)
                )

                if informative_direct_near_zero:
                    classification_reason = "direct_plot_aligned_near_zero_ALT_or_REF"
                elif informative_primary:
                    classification_reason = "statistical_fallback_wilcoxon_AF_shift_imbalance_dropout"
                elif informative_dropout_rescue:
                    classification_reason = "direct_pooled_read_depletion"
                elif low_depth_near_zero_candidate:
                    classification_reason = "low_depth_near_zero_candidate_requires_matching_independent_same_arm_LOH_pattern"
                elif not direct_near_zero_wt_ok:
                    classification_reason = (
                        "not_in_supplied_germline_whitelist"
                        if germline_variant_set else "WT_not_clean_heterozygous"
                    )
                elif direct_near_zero_pattern and not low_depth_near_zero_basic_depth_ok:
                    if not low_depth_near_zero_total_reads_ok:
                        classification_reason = "insufficient_low_depth_candidate_total_clone_reads"
                    elif not low_depth_near_zero_mean_depth_ok:
                        classification_reason = "insufficient_low_depth_candidate_mean_depth"
                    elif not low_depth_near_zero_qualified_cells_ok:
                        classification_reason = "insufficient_low_depth_candidate_depth_qualified_cells"
                    else:
                        classification_reason = "insufficient_low_depth_candidate_support"
                elif direct_near_zero_pattern and not low_depth_near_zero_exact_evidence_ok:
                    classification_reason = "low_depth_near_zero_insufficient_exact_pooled_evidence"
                elif has_counts and not direct_near_zero_pattern:
                    classification_reason = "no_plot_aligned_near_zero_allele_depletion"
                elif not has_minimum_read_evidence:
                    classification_reason = "insufficient_clone_read_depth"
                elif not statistical_test_cell_eligible:
                    classification_reason = "insufficient_depth_qualified_cells_for_statistical_test"
                elif baf_shift_vs_wt < float(dropout_rescue_min_baf_shift_vs_wt):
                    classification_reason = "insufficient_AF_shift_vs_WT"
                elif not dropout_like_allelic_pattern:
                    classification_reason = "no_pooled_depth_weighted_depletion"
                elif not rescue_iqr_ok:
                    classification_reason = "AF_IQR_above_rescue_threshold"
                else:
                    classification_reason = "not_significant_or_not_strict"

                rows.append({
                    "variant": variant,
                    "label": lab,
                    "amplicon": _amplicon_for_variant(variant),
                    "amplicon_mapping_available": amplicon_mapping_available,
                    "amplicon_mapping_overlap": amplicon_mapping_overlap,
                    "mean_af_clone": mean_af_clone,
                    "mean_af_wt": mean_af_wt,
                    "delta_af": delta_af,
                    "wilcoxon_stat": stat,
                    "wilcoxon_p": p,
                    "significant_threshold": is_significant,
                    "significant_threshold_0.01": bool(p < 0.01),
                    "mean_alt_clone": mean_alt_clone,
                    "mean_ref_clone": mean_ref_clone,
                    "mean_depth_clone": mean_depth_clone,
                    "mean_depth_wt": mean_depth_wt,
                    "clone_total_reads": clone_total_reads,
                    "wt_total_reads": wt_total_reads,
                    "depth_ratio": depth_ratio,
                    "n_cells": n_cells,
                    "n_wt_cells": n_wt_cells,
                    "statistical_test_cell_eligible": statistical_test_cell_eligible,
                    "n_plot_cells": int(plot_clone_mask.sum()),
                    "af_iqr": af_iqr,
                    "wt_af_iqr": wt_af_iqr,
                    "baf_deviation": baf_deviation,
                    "baf_shift_vs_wt": baf_shift_vs_wt,
                    "clone_imbalanced": clone_imbalanced,
                    "has_shift_vs_wt": has_shift_vs_wt,
                    "has_imbalance": has_imbalance,
                    "is_consistent": is_consistent,
                    "clone_minor_allele_fraction": clone_minor_allele_fraction,
                    "wt_minor_allele_fraction": wt_minor_allele_fraction,
                    "clone_dropout_fraction": clone_dropout_fraction,
                    "clone_near_dropout_fraction": clone_near_dropout_fraction,
                    "clone_depth_weighted_dropout_fraction": weighted_dropout_fraction,
                    "clone_depth_weighted_near_dropout_fraction": weighted_near_dropout_fraction,
                    "pooled_depletion": pooled_depletion,
                    "weighted_dropout": weighted_dropout,
                    "allele_dropout": allele_dropout,
                    "wt_is_clean_het_for_dropout_rescue": wt_is_clean_het,
                    "clone_min_mean_allele_reads": clone_min_mean_allele_reads,
                    "plot_mean_alt_clone": plot_mean_alt_clone,
                    "plot_mean_ref_clone": plot_mean_ref_clone,
                    "plot_mean_depth_clone": plot_mean_depth_clone,
                    "plot_clone_total_reads": plot_clone_total_reads,
                    "plot_depth_qualified_cell_count": plot_depth_qualified_cell_count,
                    "plot_clone_alt_sum": plot_clone_alt_sum,
                    "plot_clone_ref_sum": plot_clone_ref_sum,
                    "plot_depleted_allele": plot_depleted_allele,
                    "plot_depleted_allele_reads": plot_depleted_allele_reads,
                    "plot_expected_depleted_allele_fraction_wt": plot_expected_depleted_allele_fraction_wt,
                    "plot_clone_min_mean_allele_reads": plot_clone_min_mean_allele_reads,
                    "plot_minor_allele_fraction": plot_minor_allele_fraction,
                    "is_germline_whitelist_variant": is_germline_whitelist_variant,
                    "direct_near_zero_reference_source": direct_near_zero_reference_source,
                    "direct_near_zero_pattern": direct_near_zero_pattern,
                    "direct_near_zero_mean_depth_ok": direct_near_zero_mean_depth_ok,
                    "direct_near_zero_total_reads_ok": direct_near_zero_total_reads_ok,
                    "direct_near_zero_qualified_cells_ok": direct_near_zero_qualified_cells_ok,
                    "direct_near_zero_has_adequate_depth": direct_near_zero_has_adequate_depth,
                    "direct_near_zero_allele_depletion": direct_near_zero_allele_depletion,
                    "direct_near_zero_wt_ok": direct_near_zero_wt_ok,
                    "low_depth_near_zero_mean_depth_ok": low_depth_near_zero_mean_depth_ok,
                    "low_depth_near_zero_total_reads_ok": low_depth_near_zero_total_reads_ok,
                    "low_depth_near_zero_qualified_cells_ok": low_depth_near_zero_qualified_cells_ok,
                    "low_depth_near_zero_basic_depth_ok": low_depth_near_zero_basic_depth_ok,
                    "low_depth_near_zero_binom_p": low_depth_near_zero_binom_p,
                    "low_depth_near_zero_depleted_allele_upper_bound": low_depth_near_zero_depleted_allele_upper_bound,
                    "low_depth_near_zero_exact_evidence_ok": low_depth_near_zero_exact_evidence_ok,
                    "low_depth_near_zero_candidate": low_depth_near_zero_candidate,
                    "low_depth_near_zero_anchor_variants": 0,
                    "low_depth_near_zero_anchor_amplicons": 0,
                    "low_depth_near_zero_supporting_anchor_variants": 0,
                    "low_depth_near_zero_supporting_anchor_amplicons": 0,
                    "low_depth_near_zero_anchor_minor_allele_fraction": np.nan,
                    "low_depth_near_zero_anchor_phase_invariant_imbalance": np.nan,
                    "low_depth_near_zero_minor_fraction_distance": np.nan,
                    "low_depth_near_zero_phase_invariant_imbalance_distance": np.nan,
                    "low_depth_near_zero_pattern_matches_arm": False,
                    "informative_low_depth_near_zero_rescue": False,
                    "informative_direct_near_zero": informative_direct_near_zero,
                    "informative_direct_plot_aligned_near_zero": informative_direct_near_zero,
                    "strong_pooled_allele_depletion": strong_pooled_allele_depletion,
                    "broad_depth_weighted_near_dropout": broad_weighted_near_dropout,
                    "dropout_like_allelic_pattern": dropout_like_allelic_pattern,
                    "informative_primary": informative_primary,
                    "informative_statistical_fallback": informative_primary,
                    "informative_dropout_rescue": informative_dropout_rescue,
                    "arm_rescue_candidate": arm_rescue_candidate,
                    "informative_arm_rescue": False,
                    "hem_antiphase_candidate": False,
                    "hem_antiphase_anchor_label": pd.NA,
                    "hem_antiphase_anchor_mean_af": np.nan,
                    "hem_antiphase_anchor_delta_af": np.nan,
                    "hem_antiphase_candidate_delta_af": delta_af,
                    "hem_antiphase_signed_shift_product": np.nan,
                    "informative_hem_antiphase_rescue": False,
                    "informative_secondary": informative_dropout_rescue,
                    "informative": bool(
                        informative_direct_near_zero
                        or informative_primary
                        or informative_dropout_rescue
                    ),
                    "informative_for_plot": bool(
                        informative_direct_near_zero
                        or informative_primary
                        or informative_dropout_rescue
                    ),
                    "informativeness_tier": (
                        "direct_plot_aligned_near_zero_ALT_or_REF" if informative_direct_near_zero
                        else "statistical_fallback" if informative_primary
                        else "direct_pooled_read_depletion" if informative_dropout_rescue
                        else "not_informative"
                    ),
                    "classification_reason": classification_reason,
                    "event_type": _event_type(depth_ratio) if (
                        informative_direct_near_zero
                        or informative_primary
                        or informative_dropout_rescue
                    ) else "",
                })

        if not rows:
            continue
        chr_results = pd.DataFrame(rows)

        # ── Low-depth near-zero corroboration rescue ───────────────────────
        # A low-depth monoallelic-looking site is not allowed to establish LOH
        # by itself. It must already have strong pooled evidence relative to WT
        # (computed above), then be corroborated by high-confidence markers on
        # the same chromosome arm from independent amplicons. This preserves
        # true small-clone LOH (for example ~5 cells × 10 reads) while blocking
        # isolated locus/amplicon dropout.
        if enable_low_depth_near_zero_rescue:
            if require_amplicon_mapping_for_corroboration and low_depth_near_zero_require_independent_amplicons and not amplicon_mapping_available:
                vprint(
                    f"    {chr_name}: low-depth near-zero rescue skipped "
                    "(variant_to_amplicon map is required)."
                )
            else:
                _high_confidence_anchor_mask = (
                    chr_results["informative_direct_near_zero"]
                    | chr_results["informative_primary"]
                    | chr_results["informative_dropout_rescue"]
                )
                for _lab, _idx in chr_results.groupby("label").groups.items():
                    _idx = list(_idx)
                    _anchors = chr_results.loc[_idx].loc[
                        _high_confidence_anchor_mask.loc[_idx]
                    ].copy()
                    if _anchors.empty:
                        continue

                    _anchor_amplicons = int(_anchors["amplicon"].dropna().nunique())
                    chr_results.loc[_idx, "low_depth_near_zero_anchor_variants"] = int(len(_anchors))
                    chr_results.loc[_idx, "low_depth_near_zero_anchor_amplicons"] = _anchor_amplicons

                    if len(_anchors) < int(low_depth_near_zero_min_anchor_variants):
                        continue
                    if (
                        low_depth_near_zero_require_independent_amplicons
                        and _anchor_amplicons < int(low_depth_near_zero_min_anchor_amplicons)
                    ):
                        continue

                    _candidates = chr_results.loc[_idx].loc[
                        chr_results.loc[_idx, "low_depth_near_zero_candidate"]
                    ].copy()
                    if _candidates.empty:
                        continue

                    for _candidate_idx, _candidate in _candidates.iterrows():
                        _candidate_amplicon = _candidate.get("amplicon", pd.NA)
                        _support = _anchors.copy()
                        if low_depth_near_zero_require_independent_amplicons:
                            if pd.isna(_candidate_amplicon):
                                continue
                            _support = _support.loc[
                                _support["amplicon"].notna()
                                & _support["amplicon"].ne(_candidate_amplicon)
                            ].copy()

                        _support_amplicons = int(_support["amplicon"].dropna().nunique())
                        chr_results.at[_candidate_idx, "low_depth_near_zero_supporting_anchor_variants"] = int(len(_support))
                        chr_results.at[_candidate_idx, "low_depth_near_zero_supporting_anchor_amplicons"] = _support_amplicons

                        if len(_support) < int(low_depth_near_zero_min_anchor_variants):
                            continue
                        if (
                            low_depth_near_zero_require_independent_amplicons
                            and _support_amplicons < int(low_depth_near_zero_min_anchor_amplicons)
                        ):
                            continue

                        # Pattern match is intentionally phase-invariant. A
                        # true LOH marker may lose ALT or REF depending on its
                        # haplotype, so compare minor-allele depletion and the
                        # magnitude of pooled AF imbalance rather than sign.
                        _support = _support.copy()
                        _support_total = pd.to_numeric(
                            _support["plot_clone_total_reads"], errors="coerce"
                        )
                        _support_alt = pd.to_numeric(
                            _support["plot_clone_alt_sum"], errors="coerce"
                        )
                        _support_phase_imbalance = (
                            _support_alt / _support_total - 0.5
                        ).abs()
                        _anchor_minor = float(
                            pd.to_numeric(
                                _support["plot_minor_allele_fraction"], errors="coerce"
                            ).median()
                        )
                        _anchor_imbalance = float(_support_phase_imbalance.median())

                        _candidate_total = float(_candidate["plot_clone_total_reads"])
                        _candidate_alt = float(_candidate["plot_clone_alt_sum"])
                        _candidate_minor = float(_candidate["plot_minor_allele_fraction"])
                        _candidate_imbalance = abs(
                            _candidate_alt / max(_candidate_total, np.finfo(float).eps) - 0.5
                        )
                        _minor_dist = abs(_candidate_minor - _anchor_minor)
                        _imbalance_dist = abs(_candidate_imbalance - _anchor_imbalance)
                        _pattern_matches_arm = bool(
                            _minor_dist <= float(low_depth_near_zero_max_minor_fraction_distance)
                            and _imbalance_dist
                            <= float(low_depth_near_zero_max_phase_invariant_imbalance_distance)
                        )

                        chr_results.at[_candidate_idx, "low_depth_near_zero_anchor_minor_allele_fraction"] = _anchor_minor
                        chr_results.at[_candidate_idx, "low_depth_near_zero_anchor_phase_invariant_imbalance"] = _anchor_imbalance
                        chr_results.at[_candidate_idx, "low_depth_near_zero_minor_fraction_distance"] = _minor_dist
                        chr_results.at[_candidate_idx, "low_depth_near_zero_phase_invariant_imbalance_distance"] = _imbalance_dist
                        chr_results.at[_candidate_idx, "low_depth_near_zero_pattern_matches_arm"] = _pattern_matches_arm

                        if not _pattern_matches_arm:
                            continue

                        chr_results.at[_candidate_idx, "informative_low_depth_near_zero_rescue"] = True
                        chr_results.at[_candidate_idx, "informative_secondary"] = True
                        chr_results.at[_candidate_idx, "informativeness_tier"] = "low_depth_near_zero_corroborated"
                        chr_results.at[_candidate_idx, "classification_reason"] = (
                            "low_depth_near_zero_matches_independent_same_arm_LOH_pattern"
                        )
                        chr_results.at[_candidate_idx, "event_type"] = _event_type(
                            _candidate["depth_ratio"]
                        )

        # ── Conservative LOH-arm pattern rescue ────────────────────────────
        # The direct/statistical paths above remain primary. This pass only
        # rescues a marker when already-called LOH markers on the same arm
        # provide independent, assay-level corroboration. ALT-versus-REF phase
        # is deliberately ignored: an unphased germline marker may sit on
        # either haplotype, so similarity is evaluated on phase-invariant
        # imbalance magnitude and pooled minor-allele depletion.
        _pattern_cols = {
            "loh_arm_pattern_anchor_variants": 0,
            "loh_arm_pattern_anchor_amplicons": 0,
            "loh_arm_pattern_supporting_anchor_variants": 0,
            "loh_arm_pattern_supporting_anchor_amplicons": 0,
            "loh_arm_pattern_anchor_minor_allele_fraction": np.nan,
            "loh_arm_pattern_anchor_phase_invariant_imbalance": np.nan,
            "loh_arm_pattern_anchor_depth_ratio": np.nan,
            "loh_arm_pattern_minor_fraction_distance": np.nan,
            "loh_arm_pattern_phase_invariant_imbalance_distance": np.nan,
            "loh_arm_pattern_depth_ratio_distance": np.nan,
            "loh_arm_pattern_eligible_candidate": False,
            "informative_loh_arm_pattern_rescue": False,
        }
        for _col, _default in _pattern_cols.items():
            chr_results[_col] = _default

        if enable_loh_arm_pattern_rescue:
            if require_amplicon_mapping_for_corroboration and loh_arm_pattern_require_independent_amplicons and not amplicon_mapping_available:
                vprint(
                    f"    {chr_name}: LOH-arm pattern rescue skipped "
                    "(variant_to_amplicon map is required)."
                )
            else:
                _base_informative = (
                    chr_results["informative_direct_near_zero"]
                    | chr_results["informative_primary"]
                    | chr_results["informative_dropout_rescue"]
                )
                _loh_anchor_mask = (
                    _base_informative
                    & chr_results["wt_is_clean_het_for_dropout_rescue"]
                    & (chr_results["baf_shift_vs_wt"] >= float(loh_arm_pattern_min_baf_shift_vs_wt))
                    & (chr_results["mean_depth_clone"] >= float(direct_near_zero_min_mean_total_depth))
                    & (chr_results["depth_ratio"].isna() | (chr_results["depth_ratio"] <= float(depth_ratio_amplification)))
                )

                for _lab, _idx in chr_results.groupby("label").groups.items():
                    _idx = list(_idx)
                    _anchors = chr_results.loc[_idx].loc[_loh_anchor_mask.loc[_idx]].copy()
                    if _anchors.empty:
                        continue

                    # A candidate must be supported by *other* amplicons, not
                    # multiple correlated calls from its own assay amplicon.
                    _anchor_amplicons = int(_anchors["amplicon"].dropna().nunique())
                    chr_results.loc[_idx, "loh_arm_pattern_anchor_variants"] = int(len(_anchors))
                    chr_results.loc[_idx, "loh_arm_pattern_anchor_amplicons"] = _anchor_amplicons

                    if len(_anchors) < int(loh_arm_pattern_min_anchor_variants):
                        continue
                    if (
                        loh_arm_pattern_require_independent_amplicons
                        and _anchor_amplicons < int(loh_arm_pattern_min_anchor_amplicons)
                    ):
                        continue

                    _uncalled = chr_results.loc[_idx].loc[~_base_informative.loc[_idx]].copy()
                    if _uncalled.empty:
                        continue

                    for _candidate_idx, _candidate in _uncalled.iterrows():
                        _candidate_amplicon = _candidate.get("amplicon", pd.NA)
                        _support = _anchors.copy()
                        if loh_arm_pattern_require_independent_amplicons:
                            if pd.isna(_candidate_amplicon):
                                continue
                            _support = _support.loc[
                                _support["amplicon"].notna()
                                & _support["amplicon"].ne(_candidate_amplicon)
                            ].copy()

                        _support_amplicons = int(_support["amplicon"].dropna().nunique())
                        if len(_support) < int(loh_arm_pattern_min_anchor_variants):
                            continue
                        if (
                            loh_arm_pattern_require_independent_amplicons
                            and _support_amplicons < int(loh_arm_pattern_min_anchor_amplicons)
                        ):
                            continue

                        _candidate_imbalance = abs(float(_candidate["mean_af_clone"]) / 100.0 - 0.5)
                        _support = _support.assign(
                            _phase_invariant_imbalance=(
                                _support["mean_af_clone"].astype(float) / 100.0 - 0.5
                            ).abs()
                        )
                        _anchor_minor = float(_support["clone_minor_allele_fraction"].median())
                        _anchor_imbalance = float(_support["_phase_invariant_imbalance"].median())
                        _anchor_depth = float(_support["depth_ratio"].median())

                        _minor_dist = abs(float(_candidate["clone_minor_allele_fraction"]) - _anchor_minor)
                        _imbalance_dist = abs(_candidate_imbalance - _anchor_imbalance)
                        _depth_dist = (
                            abs(float(_candidate["depth_ratio"]) - _anchor_depth)
                            if pd.notna(_candidate["depth_ratio"]) and pd.notna(_anchor_depth)
                            else np.nan
                        )

                        _eligible = bool(
                            _candidate["statistical_test_cell_eligible"]
                            and _candidate["wt_is_clean_het_for_dropout_rescue"]
                            and _candidate["baf_shift_vs_wt"] >= float(loh_arm_pattern_min_baf_shift_vs_wt)
                            and _candidate["af_iqr"] <= float(loh_arm_pattern_max_af_iqr)
                            and _candidate["mean_depth_clone"] >= float(direct_near_zero_min_mean_total_depth)
                            and _candidate["clone_total_reads"] >= float(min_clone_total_reads)
                            and _candidate["clone_minor_allele_fraction"] <= float(loh_arm_pattern_max_minor_allele_fraction)
                            and (
                                pd.isna(_candidate["depth_ratio"])
                                or _candidate["depth_ratio"] <= float(depth_ratio_amplification)
                            )
                        )
                        _similar = bool(
                            _minor_dist <= float(loh_arm_pattern_max_minor_fraction_distance)
                            and _imbalance_dist <= float(loh_arm_pattern_max_phase_invariant_imbalance_distance)
                            and (
                                pd.isna(_depth_dist)
                                or _depth_dist <= float(loh_arm_pattern_max_depth_ratio_distance)
                            )
                        )

                        chr_results.at[_candidate_idx, "loh_arm_pattern_supporting_anchor_variants"] = int(len(_support))
                        chr_results.at[_candidate_idx, "loh_arm_pattern_supporting_anchor_amplicons"] = _support_amplicons
                        chr_results.at[_candidate_idx, "loh_arm_pattern_anchor_minor_allele_fraction"] = _anchor_minor
                        chr_results.at[_candidate_idx, "loh_arm_pattern_anchor_phase_invariant_imbalance"] = _anchor_imbalance
                        chr_results.at[_candidate_idx, "loh_arm_pattern_anchor_depth_ratio"] = _anchor_depth
                        chr_results.at[_candidate_idx, "loh_arm_pattern_minor_fraction_distance"] = _minor_dist
                        chr_results.at[_candidate_idx, "loh_arm_pattern_phase_invariant_imbalance_distance"] = _imbalance_dist
                        chr_results.at[_candidate_idx, "loh_arm_pattern_depth_ratio_distance"] = _depth_dist
                        chr_results.at[_candidate_idx, "loh_arm_pattern_eligible_candidate"] = _eligible

                        if _eligible and _similar:
                            chr_results.at[_candidate_idx, "informative_loh_arm_pattern_rescue"] = True
                            chr_results.at[_candidate_idx, "informativeness_tier"] = "LOH_arm_pattern_rescue"
                            chr_results.at[_candidate_idx, "classification_reason"] = (
                                "matches_independent_LOH_anchor_pattern_on_same_chr_arm"
                            )
                            chr_results.at[_candidate_idx, "event_type"] = _event_type(_candidate["depth_ratio"])

        if enable_arm_rescue and amplicon_mapping_available:
            for lab, idx in chr_results.groupby("label").groups.items():
                candidates = chr_results.loc[list(idx)]
                candidates = candidates.loc[
                    candidates["arm_rescue_candidate"]
                    & ~candidates["informative_direct_near_zero"]
                    & ~candidates["informative_primary"]
                    & ~candidates["informative_dropout_rescue"]
                    & ~candidates["informative_loh_arm_pattern_rescue"]
                    & ~candidates["informative_low_depth_near_zero_rescue"]
                    & candidates["amplicon"].notna()
                ]
                n_unique = int(candidates["amplicon"].nunique())
                chr_results.loc[list(idx), "arm_rescue_unique_amplicons"] = n_unique
                if n_unique >= int(arm_rescue_min_unique_amplicons):
                    promote = candidates.index
                    chr_results.loc[promote, "informative_arm_rescue"] = True
                    chr_results.loc[promote, "informative_secondary"] = True
                    chr_results.loc[promote, "informativeness_tier"] = "arm_rescue_partial_shift"
                    chr_results.loc[promote, "classification_reason"] = "independent_amplicon_arm_rescue"

        # ── Cross-Hem anti-phase rescue ─────────────────────────────────────
        # A clone labelled as Hem can retain the opposite haplotype from another
        # Hem clone.  Therefore an otherwise uncalled pair is promoted only if
        # the same marker is already informative in another Hem clone, the two
        # clone-vs-WT AF shifts are anti-phased, and the candidate independently
        # remains statistically convincing and allelically depleted.  This
        # catches near-threshold BAF imbalance (for example ~20% vs ~84%) but
        # does not promote weak/non-significant partial shifts.
        if enable_hem_antiphase_rescue:
            _hem_token = str(hem_label_token).strip().lower()
            _hem_p_threshold = (
                float(p_threshold)
                if hem_antiphase_p_threshold is None
                else float(hem_antiphase_p_threshold)
            )

            if not _hem_token:
                vprint(
                    f"    {chr_name}: cross-Hem anti-phase rescue skipped "
                    "(hem_label_token is empty)."
                )
            else:
                _current_informative = (
                    chr_results["informative_direct_near_zero"]
                    | chr_results["informative_primary"]
                    | chr_results["informative_dropout_rescue"]
                    | chr_results["informative_loh_arm_pattern_rescue"]
                    | chr_results["informative_arm_rescue"]
                )
                _hem_mask = chr_results["label"].astype(str).str.lower().str.contains(
                    _hem_token,
                    regex=False,
                    na=False,
                )

                for _variant, _variant_idx in chr_results.loc[_hem_mask].groupby("variant").groups.items():
                    _variant_idx = list(_variant_idx)
                    _hems_here = chr_results.loc[_variant_idx].copy()
                    _anchors = _hems_here.loc[_current_informative.loc[_variant_idx]].copy()
                    if _anchors.empty:
                        continue

                    _candidates = _hems_here.loc[~_current_informative.loc[_variant_idx]].copy()
                    if _candidates.empty:
                        continue

                    for _candidate_idx, _candidate in _candidates.iterrows():
                        _candidate_delta = float(_candidate["delta_af"])
                        _reference_ok = bool(
                            _candidate["is_germline_whitelist_variant"]
                            if germline_variant_set
                            else _candidate["wt_is_clean_het_for_dropout_rescue"]
                        )
                        _candidate_evidence_ok = bool(
                            _reference_ok
                            and (
                                (not hem_antiphase_require_statistical_cell_eligibility)
                                or _candidate["statistical_test_cell_eligible"]
                            )
                            and _candidate["significant_threshold"]
                            and float(_candidate["wilcoxon_p"]) < _hem_p_threshold
                            and _candidate["is_consistent"]
                            and _candidate["baf_shift_vs_wt"] >= float(hem_antiphase_min_baf_shift_vs_wt)
                            and (
                                (not hem_antiphase_require_allele_dropout)
                                or _candidate["allele_dropout"]
                            )
                        )
                        if not _candidate_evidence_ok:
                            continue

                        _opposite = _anchors.loc[
                            _anchors["delta_af"].astype(float).mul(_candidate_delta) < 0
                        ].copy()
                        _opposite = _opposite.loc[
                            _opposite["baf_shift_vs_wt"].astype(float)
                            >= float(hem_antiphase_min_baf_shift_vs_wt)
                        ]
                        if _opposite.empty:
                            continue

                        # Use the strongest opposite-phase Hem anchor, keeping
                        # the selected relationship visible in the output table.
                        _anchor = _opposite.assign(
                            _abs_delta=_opposite["delta_af"].astype(float).abs()
                        ).sort_values("_abs_delta", ascending=False).iloc[0]
                        _signed_product = _candidate_delta * float(_anchor["delta_af"])

                        chr_results.at[_candidate_idx, "hem_antiphase_candidate"] = True
                        chr_results.at[_candidate_idx, "hem_antiphase_anchor_label"] = str(_anchor["label"])
                        chr_results.at[_candidate_idx, "hem_antiphase_anchor_mean_af"] = float(_anchor["mean_af_clone"])
                        chr_results.at[_candidate_idx, "hem_antiphase_anchor_delta_af"] = float(_anchor["delta_af"])
                        chr_results.at[_candidate_idx, "hem_antiphase_candidate_delta_af"] = _candidate_delta
                        chr_results.at[_candidate_idx, "hem_antiphase_signed_shift_product"] = _signed_product
                        chr_results.at[_candidate_idx, "informative_hem_antiphase_rescue"] = True
                        chr_results.at[_candidate_idx, "informative_secondary"] = True
                        chr_results.at[_candidate_idx, "informativeness_tier"] = "Hem_antiphase_rescue"
                        chr_results.at[_candidate_idx, "classification_reason"] = (
                            "matches_informative_Hem_clone_with_opposite_AF_shift"
                        )
                        chr_results.at[_candidate_idx, "event_type"] = _event_type(
                            _candidate["depth_ratio"]
                        )

        chr_results["informative"] = (
            chr_results["informative_direct_near_zero"]
            | chr_results["informative_primary"]
            | chr_results["informative_dropout_rescue"]
            | chr_results["informative_loh_arm_pattern_rescue"]
            | chr_results["informative_low_depth_near_zero_rescue"]
            | chr_results["informative_arm_rescue"]
            | chr_results["informative_hem_antiphase_rescue"]
        )
        chr_results["informative_for_plot"] = chr_results["informative"]
        unresolved_event = chr_results["informative"] & chr_results["event_type"].eq("")
        chr_results.loc[unresolved_event, "event_type"] = chr_results.loc[unresolved_event, "depth_ratio"].map(_event_type)
        out[chr_name] = chr_results

        vprint(
            f"    {chr_name}: {len(chr_results)} comparisons | "
            f"near-zero={int(chr_results['informative_direct_near_zero'].sum())} | "
            f"statistical_fallback={int(chr_results['informative_primary'].sum())} | "
            f"read-rescued={int(chr_results['informative_dropout_rescue'].sum())} | "
            f"LOH-arm-pattern={int(chr_results['informative_loh_arm_pattern_rescue'].sum())} | "
            f"low-depth-corroborated={int(chr_results['informative_low_depth_near_zero_rescue'].sum())} | "
            f"arm-rescued={int(chr_results['informative_arm_rescue'].sum())} | "
            f"Hem-antiphase={int(chr_results['informative_hem_antiphase_rescue'].sum())}"
        )

    if not out:
        return {}

    os.makedirs(output_dir, exist_ok=True)
    for chr_name, res_df in out.items():
        res_df.to_csv(
            os.path.join(output_dir, f"wilcoxon_results_{chr_name}_enhanced.csv"),
            index=False,
        )

    all_results = pd.concat(out.values(), ignore_index=True)
    vprint("\n" + "=" * 60)
    vprint("RESULTS SUMMARY")
    vprint("=" * 60)
    vprint(f"  Comparisons:                     {len(all_results)}")
    vprint(f"  Direct near-zero ALT/REF calls:  {int(all_results['informative_direct_near_zero'].sum())}")
    vprint(f"  Strict depth-weighted calls:     {int(all_results['informative_primary'].sum())}")
    vprint(f"  Direct pooled-depletion rescues: {int(all_results['informative_dropout_rescue'].sum())}")
    vprint(f"  Low-depth corroborated calls:    {int(all_results['informative_low_depth_near_zero_rescue'].sum())}")
    vprint(f"  Arm-rescued partial shifts:      {int(all_results['informative_arm_rescue'].sum())}")
    vprint(f"  Cross-Hem anti-phase rescues:    {int(all_results['informative_hem_antiphase_rescue'].sum())}")
    vprint(f"  Final informative:               {int(all_results['informative'].sum())}")
    if enable_arm_rescue and not amplicon_mapping_available:
        vprint("  Arm rescue skipped: no variant_to_amplicon map supplied.")

    return out


def expand_informative_variants(
    H5_PATH,
    Sample,
    cytobands,
    output_dir,
    desired_plotting_order,
    *,
    whitelist,
    germline_whitelist,
    sample_raw_size,
    num_cores,
    relaxed_filter_config=None,
    p_threshold=0.01,
    baf_deviation_threshold=0.3,
    depth_ratio_deletion=0.7,
    depth_ratio_amplification=1.3,
    max_af_iqr=0.3,
    min_cells=5,
    quality_percentile=10,
    adaptive_qc_percentile=1,
    adaptive_median_dp_cap=10.0,
    adaptive_mean_gq_cap=50.0,
    min_clone_call_rate=None,
    min_median_dp=None,
    previous_informative_variants=None,
    require_wt_het=True,
    wt_het_band=None,
    wt_het_band_percentiles=(1, 99),
    wt_label="WT",
    blacklist_paths=None,
    exclude_pathogenic=True,
    pathogenic_annotation_df=None,
    clinvar_col="ClinVar",
    annotate_final_variants=True,
    annotation_chunk_size=1000,
    plot=True,
    plot_prefix="Continuous_AF_New_vs_Previous_Informative_Variants",
    verbose=True,
):
    """
    Expand the informative germline-variant panel using a relaxed H5 reload.

    The returned ``Sample_object`` contains:
      - DNA data reloaded from the raw H5 using relaxed variant filtering;
      - exactly the same cell barcodes as the input ``Sample``;
      - clone labels and compatible row-level annotations transferred from
        the input ``Sample`` (including multi-dimensional per-cell embeddings
        such as 'umap' and 'pca');
      - the original panel plus newly retained informative variants;
      - final variant annotations added in chunks.

    The input ``Sample`` variable cannot be overwritten from within this
    function. In the notebook, replace it explicitly with:

        res = CNV_Utils.expand_informative_variants(...)
        Sample = res["Sample_object"]

    Parameters
    ----------
    H5_PATH : str
        Path to the original Mosaic H5 file.
    Sample : missionbio.mosaic.Sample
        Current analysed sample. Its barcodes, labels, and compatible DNA
        row annotations are retained.
    cytobands : pd.DataFrame
        Cytoband annotation table used by variant grouping.
    output_dir : str
        Output directory.
    desired_plotting_order : list
        Clone order for optional AF plots.
    whitelist, germline_whitelist : list
        Variant whitelists passed to Mosaic loading.
    sample_raw_size : int
        Number of raw cells, used to set the relaxed filter configuration.
    num_cores : int
        Number of worker cores for Mosaic variant filtering.
    relaxed_filter_config : dict, optional
        Mosaic variant-filter settings. Defaults to a relaxed configuration.
    p_threshold, baf_deviation_threshold, depth_ratio_deletion,
    depth_ratio_amplification, max_af_iqr, min_cells
        Parameters passed to ``test_variant_informativeness``.
    quality_percentile : float
        Baseline percentile used only for the relative clone call-rate gate.
    adaptive_qc_percentile : float
        Very permissive baseline percentile used to derive automatic depth and
        GQ thresholds before capping them.
    adaptive_median_dp_cap : float
        Upper cap for the automatic median-depth threshold. The final
        threshold is ``min(auto_threshold, adaptive_median_dp_cap)``.
    adaptive_mean_gq_cap : float
        Upper cap for the automatic mean-GQ threshold. The final threshold is
        ``min(auto_threshold, adaptive_mean_gq_cap)``.
    min_clone_call_rate, min_median_dp : float, optional
        Optional hard floors. ``min_clone_call_rate`` supplements the relative
        clone call-rate gate; ``min_median_dp`` supplements the permissive
        automatic depth threshold.
    previous_informative_variants : sequence, optional
        Variant IDs from the pre-existing informative panel. When plotting,
        these variants are passed as the grey/background set while newly
        retained variants are highlighted. Also used to derive the WT-HET band
        when ``require_wt_het`` is True and ``wt_het_band`` is None.
    require_wt_het : bool
        If True, additional variants must be heterozygous in the WT clone
        (mean WT AF inside the WT-HET band) to be retained. This enforces that
        kept additions are germline-HET LOH markers, not somatic or homozygous
        sites that can pass the informativeness test with a large AF shift.
    wt_het_band : tuple(float, float), optional
        Explicit (lo, hi) WT AF% band defining the heterozygous range. If None,
        the band is derived from the WT AF distribution of
        ``previous_informative_variants``; if that is also unavailable, falls
        back to (30, 70).
    wt_het_band_percentiles : tuple(float, float)
        Percentiles of the reference WT AF distribution used to derive the band
        when ``wt_het_band`` is None.
    wt_label : str or sequence[str]
        Exact clone label(s) treated as the WT/reference population throughout
        relaxed informativeness testing and the final germline-HET QC gate.
        A string is treated as one exact label; a sequence permits a combined
        reference population when that is biologically intended.
    blacklist_paths : list[str], optional
        BED/CSV blacklist interval files (e.g. ENCODE hg19 blacklist). Any
        addition whose locus overlaps a blacklisted interval is dropped, never
        added. BED is treated as 0-based half-open against the 1-based variant
        position. If None, no blacklist filtering is applied.
    exclude_pathogenic : bool
        If True, additions annotated as (Likely) pathogenic in ClinVar (and not
        also benign) are dropped. Uses ``pathogenic_annotation_df`` when given,
        otherwise annotates the candidate additions on the fly.
    pathogenic_annotation_df : pd.DataFrame, optional
        Pre-computed annotations with a variant ID column/index and a ClinVar
        column. Avoids a redundant annotation call when already available.
    clinvar_col : str
        Name of the ClinVar column used for the pathogenic check.
    annotate_final_variants : bool
        Retrieve annotations for final variants in chunks and add them as
        DNA column attributes.
    annotation_chunk_size : int
        Number of variants per annotation request.
    plot : bool
        Create a plot of retained original and newly selected variants.
    plot_prefix : str
        Prefix for optional plot files.
    verbose : bool
        Print progress.

    Returns
    -------
    dict
        ``Sample_object`` is the expanded sample. ``Sample`` is provided as
        an alias for convenient reassignment. Other keys: ``Sample_all``
        (relaxed, cell-matched raw sample), ``wilcoxon_results`` (output of
        ``test_variant_informativeness``), ``qc`` (per-variant quality of the
        best informative call per variant), ``additional`` (candidate
        additions with gate columns ``wt_af``, ``is_blacklisted``,
        ``is_pathogenic``, ``kept``), ``new_kept``, ``final_variant_ids``,
        ``missing_variant_ids``, ``thresholds``, ``variant_annotations``,
        ``transferred_row_attributes`` and ``skipped_row_attributes``.

    Raises
    ------
    ValueError
        If ``wt_label`` is empty, no barcodes overlap, labels cannot be
        transferred, no WT/reference cells are found, a percentile is outside
        [0, 100], a cap is negative, ``wt_het_band`` is malformed, or
        ``annotation_chunk_size`` < 1.
    RuntimeError
        If the relaxed DNA assay cannot be subset to the input barcodes.

    Notes
    -----
    Steps: (1) ``ms.load`` the H5 with ``relaxed_filter_config`` (default
    ``min_dp=5, min_gq=20, vaf_ref=5, vaf_hom=95, vaf_het=30,
    min_prct_cells=30``, ``min_mut_prct_cells`` from 1% of raw cells clamped
    to 5-20 cells); (2) keep the input cells and copy their labels;
    (3) build AF/ALT/REF matrices (ALT = round(AF/100 * DP)) and run
    ``test_variant_informativeness``; keep the lowest-p informative call per
    variant; (4) compute clone call rate, genotyping rate, median/mean DP and
    mean GQ; (5) gate variants not already in the panel by clone call rate
    (``quality_percentile`` of the baseline), capped adaptive median DP / mean
    GQ thresholds, optional hard floors, WT-HET band, blacklist overlap and
    ClinVar pathogenicity; (6) build ``Sample_object`` = original panel +
    kept additions on raw DNA with transferred row attributes; (7) optionally
    annotate variants in chunks; (8) optionally plot with
    ``visualise_variants_by_clone``.
    """
    import copy
    import inspect
    import os
    import re

    import missionbio.mosaic as ms
    import numpy as np
    import pandas as pd

    # Use one exact reference-label contract for every stage of expansion:
    # relaxed informativeness, WT-HET QC, and returned threshold metadata.
    # ``wt_label`` remains backward compatible as a single string, but accepts
    # a sequence of exact labels for an intentionally pooled reference cohort.
    wt_reference_labels = [
        str(label).strip()
        for label in _normalize_wt_labels(wt_label)
        if pd.notna(label) and str(label).strip()
    ]
    if not wt_reference_labels:
        raise ValueError("wt_label must contain at least one non-empty exact label.")

    # Sibling package functions are not automatically present in this
    # module namespace merely because they are exported by CNV_Utils.__init__.
    from .variant_grouping import group_informative_variants_by_chr_arm
    from .visualization import visualise_variants_by_clone

    def vp(*args):
        if verbose:
            print(*args)

    def _as_str_index(values):
        return pd.Index(np.asarray(values).astype(str))

    def _select_available_ids(requested_ids, available_ids):
        """Keep requested IDs in order while preserving native assay ID types."""
        lookup = {str(value): value for value in available_ids}
        selected = [lookup[str(value)] for value in requested_ids if str(value) in lookup]
        missing = [str(value) for value in requested_ids if str(value) not in lookup]
        return selected, missing

    def _transfer_row_annotations(source_sample, target_sample):
        """
        Transfer compatible DNA row attributes from source to target by barcode.

        Handles 1-D per-cell attributes and multi-dimensional per-cell arrays
        (e.g. 'umap', 'pca' embeddings of shape (n_cells, k)) by aligning along
        the first axis (cells) on barcode identity. Barcode identity is taken
        from the target assay; labels are set separately with ``set_labels``.
        """
        source_barcodes = _as_str_index(source_sample.dna.barcodes())
        target_barcodes = _as_str_index(target_sample.dna.barcodes())
        n_source = len(source_barcodes)

        # Position of each target barcode within the source barcode order.
        source_positions = (
            pd.Series(np.arange(n_source), index=source_barcodes)
            .reindex(target_barcodes)
        )
        found_mask = source_positions.notna().to_numpy()
        found_idx = source_positions[found_mask].astype(int).to_numpy()

        transferred = []
        skipped = []

        for attr_name, attr_values in source_sample.dna.row_attrs.items():
            attr_name = str(attr_name)

            if attr_name in {"barcode", "label"}:
                continue

            try:
                values = np.asarray(attr_values)
            except Exception:
                skipped.append(attr_name)
                continue

            # First axis must be cells.
            if values.ndim < 1 or values.shape[0] != n_source:
                skipped.append(attr_name)
                continue

            if values.ndim == 1:
                aligned_values = (
                    pd.Series(values, index=source_barcodes)
                    .reindex(target_barcodes)
                    .to_numpy()
                )
            else:
                # Multi-dimensional per-cell array (e.g. umap, pca). Align along
                # the first (cell) axis, filling any unmatched cells with NaN for
                # numeric data or empty values otherwise.
                out_shape = (len(target_barcodes),) + values.shape[1:]
                if np.issubdtype(values.dtype, np.number):
                    aligned_values = np.full(out_shape, np.nan, dtype=float)
                else:
                    aligned_values = np.empty(out_shape, dtype=values.dtype)
                aligned_values[found_mask] = values[found_idx]

            try:
                target_sample.dna.row_attrs[attr_name] = aligned_values
                transferred.append(attr_name)
            except Exception:
                try:
                    target_sample.dna.add_row_attr(attr_name, aligned_values)
                    transferred.append(attr_name)
                except Exception:
                    skipped.append(attr_name)

        return transferred, skipped

    def _annotate_final_variants(sample_object, chunk_size):
        """
        Retrieve annotations from final DNA variants in chunks and attach
        each returned annotation column to ``sample_object.dna``.
        """
        if chunk_size < 1:
            raise ValueError("annotation_chunk_size must be at least 1.")

        final_ids_native = list(sample_object.dna.ids())
        final_ids_str = [str(value) for value in final_ids_native]
        n_variants = len(final_ids_native)

        if n_variants == 0:
            return pd.DataFrame(index=pd.Index([], dtype=str))

        vp(
            f"Annotating {n_variants} final variants "
            f"in chunks of {chunk_size}..."
        )

        annotation_dfs = []

        for start_idx in range(0, n_variants, chunk_size):
            stop_idx = min(start_idx + chunk_size, n_variants)

            annotation_sample = copy.copy(sample_object)
            annotation_sample.dna = sample_object.dna[:, start_idx:stop_idx]

            try:
                annotation_df = annotation_sample.dna.get_annotations()
            except Exception as exc:
                vp(
                    f"  Warning: annotations failed for variants "
                    f"{start_idx}:{stop_idx}: {exc}"
                )
                continue

            if annotation_df is None or annotation_df.empty:
                continue

            annotation_df = annotation_df.copy()

            chunk_ids = [
                str(value)
                for value in annotation_sample.dna.ids()
            ]

            # get_annotations() normally indexes rows by variant ID. When it
            # does not, recover IDs from an id-like column or from chunk order.
            if "id" in annotation_df.columns:
                annotation_df["id"] = annotation_df["id"].astype(str)
                annotation_df = annotation_df.set_index("id", drop=False)
            else:
                annotation_df.index = annotation_df.index.astype(str)

                if not set(chunk_ids).issubset(annotation_df.index):
                    if len(annotation_df) == len(chunk_ids):
                        annotation_df.index = pd.Index(chunk_ids, name="id")
                    else:
                        vp(
                            f"  Warning: could not safely align annotations "
                            f"for variants {start_idx}:{stop_idx}; skipping chunk."
                        )
                        continue

            annotation_dfs.append(annotation_df)

        if not annotation_dfs:
            vp("  Warning: no variant annotations were returned.")
            return pd.DataFrame(index=pd.Index(final_ids_str, name="id"))

        merged_annotations = pd.concat(annotation_dfs, axis=0)
        merged_annotations.index = merged_annotations.index.astype(str)

        merged_annotations = (
            merged_annotations.loc[
                ~merged_annotations.index.duplicated(keep="first")
            ]
            .reindex(final_ids_str)
        )

        added_columns = []

        for column_name, column_values in merged_annotations.items():
            column_name = str(column_name)

            # The assay's native id attribute already exists and must never be
            # replaced by an annotation dataframe column.
            if column_name == "id":
                continue

            values = column_values.to_numpy()

            try:
                sample_object.dna.col_attrs[column_name] = values
            except Exception:
                try:
                    sample_object.dna.add_col_attr(column_name, values)
                except Exception as exc:
                    vp(
                        f"  Warning: could not add variant annotation "
                        f"{column_name!r}: {exc}"
                    )
                    continue

            added_columns.append(column_name)

        vp(
            f"Annotated {len(merged_annotations)} variants "
            f"with {len(added_columns)} annotation columns."
        )

        return merged_annotations

    # ── 1. Reload raw H5 with relaxed variant filtering ─────────────────────
    if relaxed_filter_config is None:
        min_mut_prct_cells = (
            max(5, min(20, round(0.01 * sample_raw_size)))
            * 100
            / sample_raw_size
        )

        relaxed_filter_config = {
            "min_dp": 5,
            "min_gq": 20,
            "vaf_ref": 5,
            "vaf_hom": 95,
            "vaf_het": 30,
            "min_prct_cells": 30,
            "min_mut_prct_cells": min_mut_prct_cells,
            "iterations": 10,
            "n_cores": num_cores,
        }

    Sample_all = ms.load(
        H5_PATH,
        filter_variants=True,
        filter_cells=False,
        variant_filter_config=relaxed_filter_config,
        whitelist=list(whitelist) + list(germline_whitelist),
    )

    vp(
        f"Reloaded raw H5: {Sample_all.dna.shape[0]} cells × "
        f"{Sample_all.dna.shape[1]} variants"
    )

    # ── 2. Match relaxed raw data to original Sample cells ──────────────────
    source_barcode_order = [
        str(barcode)
        for barcode in Sample.dna.barcodes()
    ]

    raw_barcode_lookup = {
        str(barcode): barcode
        for barcode in Sample_all.dna.barcodes()
    }

    keep_barcode_strings = [
        barcode
        for barcode in source_barcode_order
        if barcode in raw_barcode_lookup
    ]

    keep_bc = [
        raw_barcode_lookup[barcode]
        for barcode in keep_barcode_strings
    ]

    if not keep_bc:
        raise ValueError(
            "No overlapping cell barcodes were found between Sample and the "
            "relaxed raw H5 object."
        )

    if len(keep_bc) != len(source_barcode_order):
        vp(
            f"  Warning: retained {len(keep_bc)} of "
            f"{len(source_barcode_order)} original Sample cells after "
            "matching to the raw H5."
        )

    # Subset assays in Sample_all where supported. DNA must succeed.
    try:
        Sample_all.dna = Sample_all.dna[keep_bc, Sample_all.dna.ids()]
    except Exception as exc:
        raise RuntimeError(
            "Could not subset relaxed DNA data to original Sample barcodes."
        ) from exc

    for assay_name in ("cnv", "protein"):
        assay = getattr(Sample_all, assay_name, None)

        if assay is None:
            continue

        try:
            setattr(
                Sample_all,
                assay_name,
                assay[keep_bc, assay.ids()],
            )
        except Exception as exc:
            vp(f"  Warning: could not subset {assay_name}: {exc}")

    source_labels = (
        pd.Series(
            Sample.dna.get_labels(),
            index=_as_str_index(Sample.dna.barcodes()),
            dtype="object",
        )
        .astype(str)
        .str.strip()
    )

    relaxed_barcodes = _as_str_index(Sample_all.dna.barcodes())

    labels_keep = source_labels.reindex(relaxed_barcodes)

    if labels_keep.isna().any():
        raise ValueError(
            f"{int(labels_keep.isna().sum())} relaxed raw cells could not "
            "be matched to original Sample labels."
        )

    Sample_all.dna.set_labels(labels_keep.to_numpy())

    vp(
        f"Cell-matched to Sample: {Sample_all.dna.shape[0]} cells | "
        f"{labels_keep.nunique()} label groups"
    )

    labels_all = pd.Series(
        Sample_all.dna.get_labels(),
        index=_as_str_index(Sample_all.dna.barcodes()),
        dtype="object",
    ).astype(str).str.strip()

    wt_reference_mask = labels_all.isin(wt_reference_labels)
    wt_reference_cell_count = int(wt_reference_mask.sum())
    if wt_reference_cell_count == 0:
        available_labels = sorted(labels_all.dropna().unique().tolist())
        raise ValueError(
            "No WT/reference cells found after relaxed-H5 cell matching for "
            f"exact wt_label(s): {wt_reference_labels}. "
            f"Available: {available_labels}"
        )

    vp(
        "WT/reference labels for relaxed expansion: "
        f"{wt_reference_labels} ({wt_reference_cell_count} cells)"
    )

    # ── 3. Build matrices and run informativeness on relaxed raw DNA ────────
    def labelled(attribute):
        dataframe = Sample_all.dna.get_attribute(
            attribute,
            constraint="row+col",
        ).copy()

        dataframe.index = dataframe.index.astype(str)
        dataframe.columns = dataframe.columns.astype(str)

        dataframe.insert(
            0,
            "label",
            labels_all.reindex(dataframe.index).to_numpy(),
        )

        return dataframe

    df_AF = labelled("AF")
    df_DP = labelled("DP")

    af_matrix = df_AF.drop(columns=["label"]).fillna(0)
    dp_matrix = df_DP.drop(columns=["label"]).fillna(0)

    df_ALT = (af_matrix / 100.0 * dp_matrix).round().astype(int)
    df_REF = (dp_matrix - df_ALT).clip(lower=0).astype(int)

    _, grouped_by_chr_arm = group_informative_variants_by_chr_arm(
        filtered_germline_in_samples=[
            str(variant)
            for variant in Sample_all.dna.ids()
        ],
        cytobands=cytobands,
        verbose=verbose,
    )

    # Required for the small-clone arm rescue.  Only independently mapped
    # amplicons can support one another; several variants from one amplicon
    # never create a rescue on their own.
    try:
        _variant_to_amplicon = dict(
            zip(
                np.asarray(Sample_all.dna.col_attrs["id"]).astype(str),
                np.asarray(Sample_all.dna.col_attrs["amplicon"]).astype(str),
            )
        )
    except Exception:
        _variant_to_amplicon = None

    wilcoxon_results = test_variant_informativeness(
        df_AF=df_AF,
        df_ALT=df_ALT,
        df_REF=df_REF,
        grouped_by_chr_arm=grouped_by_chr_arm,
        output_dir=os.path.join(output_dir, "informativeness_relaxed"),
        p_threshold=p_threshold,
        baf_deviation_threshold=baf_deviation_threshold,
        depth_ratio_deletion=depth_ratio_deletion,
        depth_ratio_amplification=depth_ratio_amplification,
        max_af_iqr=max_af_iqr,
        min_cells=min_cells,
        wt_labels=wt_reference_labels,
        variant_to_amplicon=_variant_to_amplicon,
        verbose=verbose,
    )

    if not wilcoxon_results:
        vp(
            "No informative variants were detected after relaxed filtering. "
            "Returning the original panel on matched raw DNA."
        )
        inf_best = pd.DataFrame(
            columns=["variant", "label", "event_type", "wilcoxon_p"]
        )
    else:
        informative_tables = pd.concat(
            wilcoxon_results.values(),
            ignore_index=True,
        )

        inf = informative_tables.loc[
            informative_tables["informative"]
        ].copy()

        inf["variant"] = inf["variant"].astype(str)

        inf_best = (
            inf.sort_values("wilcoxon_p")
            .drop_duplicates("variant", keep="first")
        )

    original_panel = set(map(str, Sample.dna.ids()))

    # ── 4. Per-variant quality metrics from relaxed raw DNA ──────────────────
    NGT = Sample_all.dna.get_attribute("NGT", constraint="row+col").copy()
    DP = Sample_all.dna.get_attribute("DP", constraint="row+col").copy()

    NGT.index = NGT.index.astype(str)
    NGT.columns = NGT.columns.astype(str)

    DP.index = DP.index.astype(str)
    DP.columns = DP.columns.astype(str)

    try:
        GQ = Sample_all.dna.get_attribute("GQ", constraint="row+col").copy()
        GQ.index = GQ.index.astype(str)
        GQ.columns = GQ.columns.astype(str)
        has_gq = True
    except Exception:
        GQ = None
        has_gq = False

    quality_rows = []

    for _, result_row in inf_best.iterrows():
        variant = str(result_row["variant"])
        clone = str(result_row["label"])

        if variant not in NGT.columns:
            continue

        clone_barcodes = labels_all.index[labels_all.eq(clone)]

        clone_ngt = NGT.loc[
            NGT.index.intersection(clone_barcodes),
            variant,
        ]

        quality_row = {
            "variant": variant,
            "clone": clone,
            "event_type": result_row.get("event_type"),
            "in_panel": variant in original_panel,
            "clone_call_rate": (
                float((clone_ngt != 3).mean())
                if len(clone_ngt)
                else np.nan
            ),
            "genotyping_rate": float((NGT[variant] != 3).mean()),
            "median_DP": float(DP[variant].median()),
            "mean_DP": float(DP[variant].mean()),
        }

        if has_gq:
            quality_row["mean_GQ"] = float(GQ[variant].mean())

        quality_rows.append(quality_row)

    if quality_rows:
        qc = pd.DataFrame(quality_rows).set_index("variant")
    else:
        qc_columns = [
            "clone",
            "event_type",
            "in_panel",
            "clone_call_rate",
            "genotyping_rate",
            "median_DP",
            "mean_DP",
        ] + (["mean_GQ"] if has_gq else [])

        qc = pd.DataFrame(columns=qc_columns)
        qc.index.name = "variant"

    qc["group"] = np.where(
        qc["in_panel"].fillna(False),
        "baseline",
        "additional",
    )

    # ── 5. Quality gate for additions ───────────────────────────────────────
    # Clone call rate remains relative to the pre-existing informative panel.
    # Depth and GQ are derived from a deliberately permissive baseline
    # percentile, then capped so that an unusually high-quality baseline cannot
    # make the gate overly restrictive for newly discovered variants.
    baseline = qc.loc[qc["group"].eq("baseline")].copy()
    additional = qc.loc[qc["group"].eq("additional")].copy()

    if not 0 <= float(quality_percentile) <= 100:
        raise ValueError("quality_percentile must be between 0 and 100.")

    if not 0 <= float(adaptive_qc_percentile) <= 100:
        raise ValueError("adaptive_qc_percentile must be between 0 and 100.")

    if float(adaptive_median_dp_cap) < 0:
        raise ValueError("adaptive_median_dp_cap must be non-negative.")

    if float(adaptive_mean_gq_cap) < 0:
        raise ValueError("adaptive_mean_gq_cap must be non-negative.")

    def _finite_quantile(dataframe, column, percentile):
        values = pd.to_numeric(
            dataframe[column],
            errors="coerce",
        ).dropna()

        if values.empty:
            return -np.inf

        return float(values.quantile(percentile / 100.0))

    clone_call_rate_threshold = _finite_quantile(
        baseline,
        "clone_call_rate",
        float(quality_percentile),
    )

    median_dp_auto = _finite_quantile(
        baseline,
        "median_DP",
        float(adaptive_qc_percentile),
    )

    median_dp_threshold = (
        min(median_dp_auto, float(adaptive_median_dp_cap))
        if np.isfinite(median_dp_auto)
        else -np.inf
    )

    if has_gq:
        mean_gq_auto = _finite_quantile(
            baseline,
            "mean_GQ",
            float(adaptive_qc_percentile),
        )

        mean_gq_threshold = (
            min(mean_gq_auto, float(adaptive_mean_gq_cap))
            if np.isfinite(mean_gq_auto)
            else -np.inf
        )
    else:
        mean_gq_auto = np.nan
        mean_gq_threshold = np.nan

    thresholds = {
        "clone_call_rate_pctl": clone_call_rate_threshold,
        "clone_call_rate_percentile": float(quality_percentile),
        "median_DP_auto": median_dp_auto,
        "median_DP_final": median_dp_threshold,
        "median_DP_adaptive_percentile": float(adaptive_qc_percentile),
        "median_DP_cap": float(adaptive_median_dp_cap),
        "mean_GQ_auto": mean_gq_auto,
        "mean_GQ_final": mean_gq_threshold,
        "mean_GQ_adaptive_percentile": float(adaptive_qc_percentile),
        "mean_GQ_cap": float(adaptive_mean_gq_cap),
        "wt_reference_labels": list(wt_reference_labels),
        "wt_reference_cell_count": int(wt_reference_cell_count),
        "wt_het_band": None,
    }

    keep_new = pd.Series(True, index=additional.index, dtype=bool)

    # Relative to the existing informative panel.
    keep_new &= additional["clone_call_rate"].ge(clone_call_rate_threshold)

    # Automatic but deliberately permissive depth/GQ QC.
    keep_new &= additional["median_DP"].ge(median_dp_threshold)

    if has_gq:
        keep_new &= additional["mean_GQ"].ge(mean_gq_threshold)

    # Optional user-supplied hard floors remain available.
    if min_clone_call_rate is not None:
        keep_new &= additional["clone_call_rate"].ge(
            float(min_clone_call_rate)
        )

    if min_median_dp is not None:
        keep_new &= additional["median_DP"].ge(float(min_median_dp))

    # ── Germline-HET gate ────────────────────────────────────────────────
    # Derive WT AF and the reference band independently of whether any new
    # candidates exist. This keeps ``additional`` schema-stable (notably
    # ``wt_af``) and makes threshold metadata auditable for empty runs.
    wt_cells = df_AF.index[
        df_AF["label"].astype(str).str.strip().isin(wt_reference_labels)
    ]
    if len(wt_cells) == 0:
        raise ValueError(
            "No WT/reference cells are present in df_AF for exact labels: "
            f"{wt_reference_labels}."
        )

    wt_af_mean = (
        df_AF.loc[wt_cells]
        .drop(columns=["label"])
        .apply(pd.to_numeric, errors="coerce")
        .mean()
    )

    # Always expose the column, even for an empty candidate frame or when the
    # germline-HET gate is disabled. This makes downstream notebook summaries
    # safe to index consistently.
    additional["wt_af"] = pd.to_numeric(
        additional.index.to_series().map(wt_af_mean),
        errors="coerce",
    ).to_numpy(dtype=float)

    het_band = None
    if require_wt_het:
        if wt_het_band is not None:
            if len(wt_het_band) != 2:
                raise ValueError("wt_het_band must be a two-value (low, high) sequence.")
            het_lo, het_hi = float(wt_het_band[0]), float(wt_het_band[1])
        elif previous_informative_variants is not None:
            ref_ids = [
                str(v)
                for v in previous_informative_variants
                if str(v) in wt_af_mean.index
            ]
            ref = wt_af_mean.loc[ref_ids].dropna()
            if len(ref):
                lo_p, hi_p = wt_het_band_percentiles
                het_lo = float(ref.quantile(float(lo_p) / 100.0))
                het_hi = float(ref.quantile(float(hi_p) / 100.0))
            else:
                het_lo, het_hi = 30.0, 70.0
        else:
            het_lo, het_hi = 30.0, 70.0

        if het_lo > het_hi:
            raise ValueError("wt_het_band lower bound must not exceed upper bound.")

        het_band = (het_lo, het_hi)
        keep_new &= additional["wt_af"].between(het_lo, het_hi)
        thresholds["wt_het_band"] = het_band

        vp(
            "Germline-HET gate: reference labels "
            f"{wt_reference_labels}; AF in [{het_lo:.1f}, {het_hi:.1f}]%"
        )
    else:
        vp(
            "Germline-HET gate disabled; reference labels retained for "
            f"informativeness: {wt_reference_labels}"
        )

    # ── Blacklist / pathogenic exclusion gate ────────────────────────────
    # Additions overlapping ENCODE-style blacklist intervals, or annotated as
    # (Likely) pathogenic in ClinVar, are never added regardless of how well
    # they pass the quality and germline-HET gates. Mirrors the exclusions in
    # find_germline_variants so the two paths stay consistent.
    def _norm_chrom(value):
        value = str(value).strip()
        return value if value.startswith("chr") else f"chr{value}"

    def _parse_variant(variant_id):
        match = re.match(r"^(chr[^:]+):(\d+):", str(variant_id))
        if not match:
            return None, None
        return _norm_chrom(match.group(1)), int(match.group(2))

    def _read_blacklist(path):
        ext = os.path.splitext(path)[1].lower()
        if ext in (".bed", ".tsv", ".txt"):
            raw = pd.read_csv(path, sep="\t", header=None, comment="#")
            if raw.shape[1] < 3:
                raise ValueError(f"Blacklist has fewer than 3 columns: {path}")
            frame = raw.iloc[:, :3].copy()
            frame.columns = ["chr", "start", "end"]
        else:
            frame = pd.read_csv(path)
            rename = {}
            for column in frame.columns:
                cl = column.lower().strip()
                if cl in ("chromosome", "chrom", "chr"):
                    rename[column] = "chr"
                elif cl in ("start", "begin", "chromstart"):
                    rename[column] = "start"
                elif cl in ("stop", "end", "chromend"):
                    rename[column] = "end"
            frame = frame.rename(columns=rename)[["chr", "start", "end"]]
        frame["chr"] = frame["chr"].map(_norm_chrom)
        frame["start"] = pd.to_numeric(frame["start"], errors="coerce")
        frame["end"] = pd.to_numeric(frame["end"], errors="coerce")
        frame = frame.dropna(subset=["chr", "start", "end"])
        frame["start"] = frame["start"].astype(int)
        frame["end"] = frame["end"].astype(int)
        return frame

    blacklisted_additions = set()
    if blacklist_paths and len(additional):
        intervals = {}
        for bl_path in blacklist_paths:
            try:
                bl = _read_blacklist(bl_path)
            except Exception as exc:
                vp(f"  Warning: could not read blacklist {bl_path}: {exc}")
                continue
            for chrom, sub in bl.groupby("chr"):
                intervals.setdefault(chrom, []).extend(
                    zip(sub["start"].tolist(), sub["end"].tolist())
                )

        for variant in additional.index:
            chrom, pos = _parse_variant(variant)
            if chrom is None or chrom not in intervals:
                continue
            # BED is 0-based half-open; Tapestri POS is 1-based.
            pos0 = pos - 1
            if any(s <= pos0 < e for s, e in intervals[chrom]):
                blacklisted_additions.add(str(variant))

    pathogenic_additions = set()
    if exclude_pathogenic and len(additional):
        anno = pathogenic_annotation_df
        if anno is None:
            # Annotate just the candidate additions from the relaxed object.
            cand_ids_native, _ = _select_available_ids(
                list(additional.index),
                Sample_all.dna.ids(),
            )
            if cand_ids_native:
                anno_sample = copy.copy(Sample_all)
                anno_sample.dna = Sample_all.dna[:, cand_ids_native]
                try:
                    anno = anno_sample.dna.get_annotations()
                except Exception as exc:
                    vp(f"  Warning: pathogenic annotation lookup failed: {exc}")
                    anno = None

        if anno is not None and not anno.empty and clinvar_col in anno.columns:
            anno = anno.copy()
            # Recover variant IDs from an id-like column or the index.
            if "id" in anno.columns:
                anno_ids = anno["id"].astype(str)
            else:
                anno_ids = anno.index.astype(str)
            clinvar = anno[clinvar_col].astype(str).str.lower()
            is_pathogenic = (
                clinvar.str.contains("pathogenic", na=False)
                & ~clinvar.str.contains("benign", na=False)
            )
            pathogenic_additions = set(
                anno_ids[is_pathogenic.to_numpy()].tolist()
            )
        elif anno is not None and clinvar_col not in anno.columns:
            vp(
                f"  Warning: '{clinvar_col}' column not in annotations; "
                "pathogenic gate skipped."
            )

    excluded_additions = blacklisted_additions | pathogenic_additions
    additional["is_blacklisted"] = additional.index.to_series().astype(str).isin(
        blacklisted_additions
    )
    additional["is_pathogenic"] = additional.index.to_series().astype(str).isin(
        pathogenic_additions
    )
    if excluded_additions:
        keep_new &= ~additional.index.to_series().astype(str).isin(excluded_additions)
        thresholds["excluded_blacklisted"] = sorted(blacklisted_additions)
        thresholds["excluded_pathogenic"] = sorted(pathogenic_additions)
        vp(
            f"Blacklist/pathogenic gate: excluded "
            f"{len(blacklisted_additions)} blacklisted, "
            f"{len(pathogenic_additions)} pathogenic addition(s)."
        )

    additional["kept"] = keep_new
    new_kept = additional.index[additional["kept"]].astype(str).tolist()

    vp(
        f"\nBaseline informative variants already in panel: {len(baseline)} | "
        f"additional informative variants: {len(additional)} | "
        f"retained additions: {len(new_kept)}"
    )

    gate_message = (
        f"Quality gate: clone_call_rate >= p{quality_percentile} "
        f"({clone_call_rate_threshold:.3f}); "
        f"median_DP >= {median_dp_threshold:.2f} "
        f"(auto p{adaptive_qc_percentile}={median_dp_auto:.2f}, "
        f"cap={float(adaptive_median_dp_cap):.2f})"
    )

    if has_gq:
        gate_message += (
            f"; mean_GQ >= {mean_gq_threshold:.2f} "
            f"(auto p{adaptive_qc_percentile}={mean_gq_auto:.2f}, "
            f"cap={float(adaptive_mean_gq_cap):.2f})"
        )

    if het_band is not None:
        gate_message += (
            f"; WT-HET AF in [{het_band[0]:.1f}, {het_band[1]:.1f}]%"
        )

    if blacklist_paths or exclude_pathogenic:
        gate_message += (
            f"; excluded {int(additional['is_blacklisted'].sum())} blacklisted, "
            f"{int(additional['is_pathogenic'].sum())} pathogenic"
        )

    vp(gate_message)

    if len(additional):
        display_columns = ["clone_call_rate"]

        if "wt_af" in additional.columns:
            display_columns.append("wt_af")

        display_columns.append("median_DP")

        if has_gq:
            display_columns.append("mean_GQ")

        for flag_col in ("is_blacklisted", "is_pathogenic"):
            if flag_col in additional.columns and additional[flag_col].any():
                display_columns.append(flag_col)

        display_columns.extend([
            "clone",
            "event_type",
            "kept",
        ])

        vp(
            additional[display_columns]
            .sort_values(
                ["clone_call_rate", "median_DP"],
                ascending=[True, True],
            )
            .round(3)
            .to_string()
        )

    # ── 6. Create final raw-DNA Sample object with input annotations ─────────
    requested_variant_ids = list(
        dict.fromkeys(
            [str(variant) for variant in Sample.dna.ids()]
            + [str(variant) for variant in new_kept]
        )
    )

    final_variant_ids_native, missing_variant_ids = _select_available_ids(
        requested_variant_ids,
        Sample_all.dna.ids(),
    )

    final_variant_ids = [
        str(variant)
        for variant in final_variant_ids_native
    ]

    if missing_variant_ids:
        vp(
            f"  Warning: {len(missing_variant_ids)} requested panel variants "
            "were unavailable after relaxed loading and were excluded."
        )

    # Preserve the existing Sample container and non-DNA assays, but replace
    # DNA with raw H5 data for matched cells and final variants.
    Sample_object = copy.copy(Sample)
    Sample_object.dna = Sample_all.dna[
        list(Sample_all.dna.barcodes()),
        final_variant_ids_native,
    ]

    target_barcodes = _as_str_index(Sample_object.dna.barcodes())
    final_labels = source_labels.reindex(target_barcodes)

    if final_labels.isna().any():
        raise ValueError(
            f"{int(final_labels.isna().sum())} final cells could not be "
            "matched to input Sample labels."
        )

    Sample_object.dna.set_labels(final_labels.to_numpy())

    transferred_attrs, skipped_attrs = _transfer_row_annotations(
        Sample,
        Sample_object,
    )

    vp(
        f"Transferred {len(transferred_attrs)} compatible DNA row "
        "annotation(s) from the input Sample."
    )

    if skipped_attrs:
        vp(
            f"  Skipped {len(skipped_attrs)} incompatible row attribute(s): "
            + ", ".join(skipped_attrs)
        )

    # ── 7. Annotate final variants in chunks ─────────────────────────────────
    if annotate_final_variants:
        variant_annotations = _annotate_final_variants(
            Sample_object,
            annotation_chunk_size,
        )
    else:
        variant_annotations = pd.DataFrame(
            index=pd.Index(final_variant_ids, name="id")
        )

    vp(
        f"\nFinal Sample_object: {Sample_object.dna.shape[0]} cells × "
        f"{Sample_object.dna.shape[1]} variants"
    )

    # ── 8. Optional plot: supplied baseline plus retained additions ─────────
    if plot:
        # Use the user's original informative panel for the grey/background
        # track when supplied. Fall back to informative variants rediscovered
        # from the relaxed data that also belong to the original panel.
        if previous_informative_variants is None:
            previous = [
                str(variant)
                for variant in inf_best.get(
                    "variant",
                    pd.Series(dtype=str),
                ).astype(str)
                if str(variant) in original_panel
            ]
        else:
            previous = [
                str(variant)
                for variant in previous_informative_variants
                if str(variant) in df_AF.columns
            ]

        previous = list(dict.fromkeys(previous))

        panel_order = []

        for _, value in grouped_by_chr_arm.items():
            if isinstance(value, dict):
                values = value.get(
                    "variants",
                    value.get(
                        "variant",
                        value.get(
                            "columns",
                            value.get("vars", []),
                        ),
                    ),
                )
            else:
                values = value

            panel_order.extend([str(variant) for variant in values])

        panel_order = list(dict.fromkeys(panel_order))

        variants_to_plot_set = set(previous) | set(new_kept)

        df_plot = df_AF.loc[
            df_AF["label"].astype(str).ne("not_in_Sample")
        ].copy()

        variants_to_plot = [
            variant
            for variant in panel_order
            if (
                variant in variants_to_plot_set
                and variant in df_plot.columns
            )
        ]

        existing_plot_variants = set(variants_to_plot)

        variants_to_plot.extend(
            [
                variant
                for variant in previous + new_kept
                if (
                    variant in df_plot.columns
                    and variant not in existing_plot_variants
                )
            ]
        )

        if variants_to_plot:
            plot_kwargs = {
                "df": df_plot,
                "variants_to_plot": variants_to_plot,
                "cytobands": cytobands,
                "output_dir": output_dir,
                "output_prefix": plot_prefix,
                "label_col": "label",
                "desired_clone_plotting_order": desired_plotting_order,
                "verbose": verbose,
                "plot_separate_clones": True,
                "split_by_cytobands": False,
            }

            visualisation_params = inspect.signature(
                visualise_variants_by_clone
            ).parameters

            highlight_param = next(
                (
                    parameter
                    for parameter in [
                        "highlight_variants",
                        "highlight",
                        "emphasis_variants",
                        "variant_highlight",
                        "highlight_vars",
                    ]
                    if parameter in visualisation_params
                ),
                None,
            )

            background_param = next(
                (
                    parameter
                    for parameter in [
                        "background_variants",
                        "greyed_variants",
                        "deemphasise_variants",
                        "deemphasize_variants",
                    ]
                    if parameter in visualisation_params
                ),
                None,
            )

            if highlight_param:
                plot_kwargs[highlight_param] = new_kept

            if background_param:
                plot_kwargs[background_param] = previous

            if not highlight_param and verbose:
                vp(
                    "  visualise_variants_by_clone has no recognised "
                    "highlight argument; plotting baseline and new variants "
                    "without styling emphasis."
                )

            visualise_variants_by_clone(**plot_kwargs)
        else:
            vp(
                "  No retained baseline or additional informative variants "
                "were available for plotting."
            )

    return {
        "Sample_all": Sample_all,
        "Sample_object": Sample_object,
        "Sample": Sample_object,
        "wilcoxon_results": wilcoxon_results,
        "qc": qc,
        "additional": additional,
        "new_kept": new_kept,
        "final_variant_ids": final_variant_ids,
        "missing_variant_ids": missing_variant_ids,
        "thresholds": thresholds,
        "variant_annotations": variant_annotations,
        "transferred_row_attributes": transferred_attrs,
        "skipped_row_attributes": skipped_attrs,
    }
