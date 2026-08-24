"""
LOH Pipeline – Configuration, shared state, and helper functions.

All phases import from here so parameter definitions and utilities live in one place.
"""
from __future__ import annotations

import re
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from collections import defaultdict
from typing import Any
from scipy.stats import pearsonr


# ─────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────
LOH_TAG     = "LOH"
HET_TAG     = "Het"
ADO_LABEL   = "ADO"
SMALL_LABEL = "small"
FP_TAG      = "FP:"
SUB_TAG     = "SUB:"


def is_small_label(label: object) -> bool:
    """Return True for the common terminal sub-threshold label ``small``."""
    text = str(label).strip()
    return text == SMALL_LABEL or text.startswith(f"{SMALL_LABEL}:")


def is_fp_label(label: object) -> bool:
    """Return True for a retained false-positive clone label (``FP: ...``)."""
    return (not is_small_label(label)) and str(label).strip().startswith(FP_TAG)


def is_sub_label(label: object) -> bool:
    """Return True for an unresolved directional subclone label (``SUB: ...``)."""
    return (not is_small_label(label)) and str(label).strip().startswith(SUB_TAG)


def is_uncertain_loh_label(label: object) -> bool:
    """Return True for review-only FP or unresolved subclonal LOH labels."""
    return is_fp_label(label) or is_sub_label(label)


def is_loh_label(label: object) -> bool:
    """Return True only for active LOH labels, never review-only FP/SUB/small labels."""
    return (not is_small_label(label)) and (not is_uncertain_loh_label(label)) and (LOH_TAG in str(label))


def make_fp_label(assigned_clone_label: object) -> str:
    """Build an inert, traceable false-positive label.

    ``LOH-C<n>`` is rendered as ``C<n>`` inside the retained tail, e.g.
    ``FP: TP53_R248W_Hem C0 chr7p:p22.1-p14.1``.  This preserves the
    assigned-clone identity while ensuring the literal ``LOH`` token cannot
    be rediscovered by legacy substring scans in later phases.
    """
    label = str(assigned_clone_label).strip()
    if is_fp_label(label):
        return label
    label = re.sub(r"\bLOH-C(\d+)\b", r"C\1", label)
    return f"{FP_TAG} {label}"


def make_sub_label(assigned_clone_label: object) -> str:
    """Build an inert unresolved-subclone label after a directional split.

    ``SUB:`` is reserved for cells from a coherent parent LOH clone that could
    not be assigned to any validated directional child. It is deliberately
    distinct from technical/biological false-positive ``FP:`` labels.
    """
    label = str(assigned_clone_label).strip()
    if is_sub_label(label):
        return label
    label = re.sub(r"\bLOH-C(\d+)\b", r"C\1", label)
    return f"{SUB_TAG} {label}"


def make_small_label(assigned_clone_label: object) -> str:
    """Return the common terminal small-clone label.

    Small groups are deliberately collapsed into the single label ``small``
    during final cleanup so 1-4-cell technical remnants do not masquerade as
    named biological populations.
    """
    return SMALL_LABEL


def clone_category(label: object) -> str:
    """Classify active LOH, SUB, FP, small, or non-LOH labels."""
    if is_small_label(label):
        return "small"
    if is_sub_label(label):
        return "subclonal_LOH"
    if is_fp_label(label):
        return "possible_LOH"
    if is_loh_label(label):
        return "LOH"
    return "non_LOH"


def fp_base_name(label: object) -> str:
    """Recover the base genotype from an FP label.

    ``FP: TP53_R248W_Hem C0 chr7p:p22.1-p14.1`` -> ``TP53_R248W_Hem``.
    The rendered ``C<n> <arm>`` tokens and the ``FP:`` prefix are stripped.
    """
    s = re.sub(rf"^{re.escape(SMALL_LABEL)}:\s*", "", str(label).strip())
    s = re.sub(rf"^(?:{re.escape(FP_TAG)}|{re.escape(SUB_TAG)})\s*", "", s)
    s = re.sub(r"\bC\d+\s+\S+", "", s)
    return re.sub(r"\s+", " ", s).strip()



# ─────────────────────────────────────────────────────────────────────
# Automatic nested parent-child LOH rescue configuration
# ─────────────────────────────────────────────────────────────────────
#
# Phase 3b detects nested LOH relationships from the labels already produced
# by Phases 1–3. No clone, chromosome arm, cytoband, or mutation is listed
# here. A candidate is considered only when:
#
#   1. an FP label and an active LOH label have the same base genotype;
#   2. the FP arm set is a strict subset of the active LOH arm set;
#   3. their shared-arm clone-level AF profiles are concordant; and
#   4. there is one unambiguous compatible child clone.
#
# Shared-arm concordant cells are then partitioned using the child-only LOH
# arm(s): extra-arm concordant cells join the child; the others become the
# rescued parent. Ambiguous parent–child relationships remain FP.

# ─────────────────────────────────────────────────────────────────────
# Configuration dataclass
# ─────────────────────────────────────────────────────────────────────
@dataclass
class LOHConfig:
    """All tuneable parameters for the current LOH pipeline phases (0–5) + plotting."""

    # Phase 1 – per-arm detection (core thresholds)
    low_allele_threshold: float = 10.0
    high_allele_threshold: float = 90.0
    min_contiguous_variants_initial: int = 4
    min_extreme_fraction: float = 0.90
    min_shared_variants: int = 4
    min_cluster_size: int = 5
    min_final_clone_size: int = 5
    correlation_threshold: float = 0.80
    detect_loh_verbose: bool = False

    # Phase 6 – iterative post-Phase-5 residual subclone scan.
    #
    # Phase 6 is a targeted refinement, not a high-sensitivity discovery pass.
    # Keep the same minimum cell count as the primary LOH workflow so sparse
    # ADO-like patterns cannot inflate the final clone table into many 2–4-cell
    # fragments. Any genuinely smaller pattern remains visible in the audit but
    # is not promoted to a final LOH child.
    phase6_residual_min_child_size: int = 5
    phase6_residual_min_cluster_size: int = 5

    # Phase 6 – primary-child similarity expansion / nested-arm handling.
    # A strict LOH core is always required first. These thresholds only admit
    # close members to that validated representative; they never create a child
    # from a weak pattern alone.
    phase6_expand_child_members: bool = True
    phase6_expand_min_corr_to_representative: float = 0.85
    phase6_expand_max_median_abs_delta: float = 15.0
    phase6_expand_min_block_extreme_fraction: float = 0.75
    phase6_expand_min_signed_agreement: float = 0.75
    phase6_expand_min_compared_variants: int = 4
    phase6_expand_constant_profile_max_abs_delta: float = 15.0

    # Phase 6 – partial-block family attachment.
    # A strict primary child can recruit a residual cell that supports a
    # contiguous signed sub-block of the validated core event. This captures
    # genuine members with edge dropout or a shorter shared LOH span while
    # retaining an independent block, amplicon, and representative-similarity
    # requirement. It never creates a new child from residual cells alone.
    phase6_expand_partial_block_members: bool = True
    phase6_expand_partial_min_corr_to_representative: float = 0.80
    phase6_expand_partial_max_median_abs_delta: float = 15.0
    phase6_expand_partial_min_block_variants: int = 4
    phase6_expand_partial_min_independent_amplicons: int = 3
    phase6_expand_partial_min_signed_fraction: float = 1.00
    phase6_expand_partial_constant_profile_max_abs_delta: float = 15.0

    # A secondary arm present in only this fraction of a primary Phase-6 child
    # is retained as row-level nested evidence instead of fragmenting the flat
    # Clone label into a small secondary-arm child and a prunable remainder.
    phase6_secondary_arm_label_min_fraction: float = 0.90

    # Clone-mean LOH block edge tolerance.  A contiguous extreme core may
    # include up to this many adjacent non-extreme variants at either terminal
    # edge for span/length calls.  Internal gaps are never bridged.
    block_edge_tolerance: int = 1

    # Phase 1 – display
    phase1_show_top_clones: bool = True
    phase1_show_failed_clones: bool = False

    # Phase 1 – localized / cytoband QC
    phase1_localized_min_cells: int = 5
    phase1_localized_min_mean_r: float = 0.85
    phase1_localized_min_good_frac: float = 0.85
    phase1_localized_good_cell_r: float = 0.85

    # NEW: localized cytoband logic
    # Require >= this many *extreme-mean* variants within the *contiguous run* of passing cytobands.
    # (Default 5 = "over 4".)
    phase1_localized_min_extreme_variants_passing_bands: int = 4

    # Phase 1 – clone-size gate applied to both arm-wide and localised calls
    phase1_min_clone_cells_all: int = 5

    # Phase 1 – extreme-AF variance QC
    phase1_extreme_sd_qc_enabled: bool = True
    phase1_extreme_sd_qc_min_cells: int = 5
    phase1_extreme_sd_qc_max_median_sd: float = 10.0
    phase1_extreme_sd_qc_max_p90_sd: float = 15.0
    phase1_sd_dynamic_factor: float = 1.2

    # Phase 1 – NGT/no-call exclusion inside candidate LOH blocks.
    # After a candidate subclone has an accepted LOH block, cells that are
    # NGT/no-call for variants in that block are stripped from the LOH subclone.
    # The default is strict: any NGT/no-call in the block removes the cell.
    phase1_exclude_ngt_cells_in_loh_block: bool = True
    phase1_ngt_max_fraction_in_block: float = 0.0
    phase1_ngt_treat_nan_as_ngt: bool = True
    phase1_ngt_no_call_values: tuple = (
        "NGT", "NO_CALL", "NO CALL", "NOCALL", "./.", ".", "", "NONE", "NAN"
    )
    phase1_ngt_numeric_no_call_values: tuple = (3,)

    # Legacy/manual only: Phase 2 p/q merging is skipped by the default full pipeline.
    post_merge_correlation_threshold: float = 0.80

    # Phase 3 – LOH clone QC
    skip_homogeneity_check: bool = False
    min_homogeneity_fraction: float = 0.80
    phase3_outlier_k: float = 2.0

    # Phase 3 Step 1 – small-clone outlier policy.
    # Relative mean-k×SD trimming is unstable when a clone contains only a
    # handful of cells: one otherwise coherent cell can fall narrowly below a
    # very tight clone-specific cutoff. For clones at or below this size, use
    # only the absolute correlation floor below. Set to None to restore legacy
    # relative trimming for every clone.
    phase3_outlier_small_clone_max_cells: int | None = 10
    phase3_outlier_small_clone_abs_corr_floor: float = 0.80

    phase3_min_variants_for_cell_corr: int = 4
    phase3_min_mean_corr_to_mean: float = 0.80
    min_variants_per_arm_for_corr: int = 4

    # Directional consistency check (applied in Phase 1 QC and Phase 3
    # homogeneity).  For each variant in the LOH block, look at the cells
    # that are extreme on that variant and compute the fraction agreeing
    # on direction (LOW vs HIGH).  Mid-range cells are treated as missing.
    # The block passes if at least
    # ``directional_min_block_pass_frac`` of variants meet
    # ``directional_min_per_variant_agreement``.
    directional_check_enabled: bool = True
    directional_min_per_variant_agreement: float = 0.95
    directional_min_block_pass_frac: float = 1.00
    directional_min_extreme_cells_per_variant: int = 5

    # Phase 3 – ADO/dropout-aware directional tolerance.
    # Applies only during Phase 3 homogeneity, not Phase 1 discovery.
    # It rescues strong LOH blocks that narrowly fail the hard directional
    # consistency gate when the discordance is explainable by at most one
    # bad amplicon, e.g. 7/8 directionally consistent variants in a TP53 Hem
    # chr17p block. Small blocks remain protected by the minimum block size.
    phase3_allow_single_ado_amplicon_direction_fail: bool = True
    phase3_direction_ado_min_block_variants: int = 5
    phase3_direction_ado_max_bad_amplicons: int = 1
    phase3_direction_ado_min_good_variants: int = 4

    # Phase 3 – split directional failures before false-positive labelling.
    # Only clones with per-variant directional discordance are split. Other
    # homogeneity failures are retained as ``FP: <assigned clone>``.
    phase3_split_directional_mixed: bool = True
    phase3_directional_split_min_cells: int | None = None
    phase3_directional_split_min_discordant_variants: int = 1

    # Retained cells from a successful directional split that cannot be placed
    # in a validated child are labelled ``SUB:`` rather than ``FP:``.
    phase3_directional_split_use_sub_label: bool = True

    # Phase 3a – multi-arm LOH validation.
    #
    # Multi-arm LOH clones are validated arm by arm using only the persisted
    # Phase-1 LOH block for each arm. A clone passes only when every represented
    # arm has a contiguous block of at least four informative variants and passes
    # its own cell-to-centroid homogeneity check. Directional orientation is not
    # pooled across arms and is not used in this multi-arm pathway.
    phase3_multi_arm_homogeneity_per_arm: bool = True
    phase3_multi_arm_require_all_arms_pass: bool = True
    phase3_multi_arm_min_block_variants: int = 4

    # When a multi-arm clone has coherent support on at least one arm but fails
    # one or more other arm-specific homogeneity checks, retain it as ``SUB:``
    # (review-only unresolved subclonal LOH) rather than labelling the whole
    # population as a technical false positive. A clone for which every arm
    # fails remains ``FP:``.
    phase3_multi_arm_partial_fail_use_sub_label: bool = True

    # Phase 3b – automatic nested parent-child LOH rescue.
    #
    # This is discovery-based, not rule-based: candidate lineages are inferred
    # from same-base FP parents whose LOH arm set is a strict subset of one
    # active same-base LOH child. No labels or arms are hard-coded.
    phase3_nested_loh_rescue_enabled: bool = True
    phase3_nested_loh_rescue_require_single_child_only_arm: bool = True
    phase3_nested_loh_rescue_min_block_variants: int = 4
    phase3_nested_loh_rescue_min_child_extreme_fraction: float = 0.85
    phase3_nested_loh_rescue_min_shared_arm_r: float = 0.80
    phase3_nested_loh_rescue_max_shared_arm_median_abs_diff: float = 15.0
    phase3_nested_loh_rescue_min_shared_arm_extreme_match: float = 0.75
    phase3_nested_loh_rescue_min_child_only_arm_r: float = 0.85
    phase3_nested_loh_rescue_max_child_only_arm_median_abs_diff: float = 15.0
    phase3_nested_loh_rescue_min_child_only_arm_extreme_match: float = 0.80
    phase3_nested_loh_rescue_min_child_only_arm_called_variants: int = 4
    phase3_nested_loh_rescue_min_profile_dynamic_range_for_corr: float = 20.0
    phase3_nested_loh_rescue_min_retained_parent_cells: int | None = None

    # Phase 4 – non-LOH ↔ LOH merge
    # Preserve Phase 3b rescued parent-child lineages. Without this guard,
    # the generic Phase 4d superset-arm merge can collapse a rescued parent
    # straight back into its child.
    phase4_preserve_nested_loh_lineages: bool = True

    # Phase 4d – iterative cell-level reassignment to a fixed point.
    # Each pass rebuilds clone profiles from the assignments made in the
    # previous pass. The loop stops when no labels change, a repeated label
    # state is detected, or the bounded iteration cap is reached.
    phase4_iterate_cell_merge: bool = True
    phase4_cell_merge_max_iters: int = 10
    phase4_cell_merge_detect_cycles: bool = True

    merge_min_extreme_frac: float = 0.85

    # Phase 4 – per-cell rescue
    # Defaults are intentionally conservative. Phase 4 should expand only
    # strong, block-consistent LOH seeds; broad rescue is a major source of
    # false-positive subclone inflation.
    phase4_percell_rescue: bool = True
    phase4_percell_min_corr: float = 0.80
    phase4_percell_min_extreme: float = 0.85

    # Phase 4 – block-aware / ADO-tolerant per-cell rescue.
    # Rescue is scored on the discovered LOH block, grouped by independent
    # amplicon.  A cell may fail/miss up to `phase4_rescue_max_ado_amplicons`
    # amplicons in the block, but only once the block has enough independent
    # informative amplicons. By default: 3/3 and 4/4 are required; ≥5-amplicon
    # blocks may tolerate one non-supporting/ADO-like amplicon.
    phase4_rescue_use_loh_block_variants: bool = True
    # Legacy/manual only: retained for compatibility; default rescue uses explicit
    # amplicon support rules, not an ADO probability model.
    phase4_rescue_expected_amplicon_ado: float = 0.10
    # Strict default: do not accept 2/3 or 3/4 support. One failed/ADO
    # amplicon is only considered safe for larger blocks, controlled by
    # phase4_rescue_min_amplicons_to_allow_ado.
    phase4_rescue_max_ado_amplicons: int | None = 1
    phase4_rescue_min_amplicons_to_allow_ado: int = 5
    phase4_rescue_min_required_amplicons: int = 3
    phase4_rescue_amplicon_variant_extreme_frac: float = 1.0
    phase4_rescue_require_direction_match: bool = True
    phase4_rescue_min_corr_variants: int = 4
    phase4_rescue_constant_profiles_are_match: bool = False
    # Legacy/manual only: ignored when phase4_rescue_constant_profiles_are_match=False.
    phase4_rescue_constant_profile_max_abs_diff: float = 15.0
    phase4_rescue_allow_amplicon_support_without_corr: bool = False
    # Exclude somatic/context variants from Phase 4 rescue amp_support when
    # cfg.informative_variants is supplied. The variants can remain in the
    # plotted LOH block, but rescue support must come from germline-informative
    # amplicons only.
    phase4_rescue_exclude_somatic_from_amp_support: bool = True

    # NEW: trigger per-cell rescue when there's a "hint of similarity"
    phase4_percell_trigger_mean_corr: float = 0.70
    phase4_percell_trigger_any_arm_corr: float = 0.85

    # Phase 4c – arm-level pooled rescue (Fix 1)
    # Builds a pooled AF profile per arm from all LOH cells, then rescues
    # non-LOH cells of matching base-genotype that show the arm's signal.
    # Disabled by default: pooled rescue can turn small recurrent ADO-like
    # blocks into very large false-positive LOH clones. Enable only after
    # inspecting Phase 1/3 seeds and with ADO probability QC active.
    phase4_arm_rescue_enabled: bool = False
    phase4_arm_rescue_min_cells: int = 5   # min rescued cells to accept a new clone

    # Phase 4c-ii – LOH arm expansion
    # Second pass over existing LOH clones: for each LOH clone, check
    # whether its cells also carry LOH on arms not yet in the clone label.
    # If all (or nearly all) cells pass, relabel the clone to include the
    # new arm.  If only a subset pass, split off those cells into a new
    # expanded clone.
    # Disabled by default: second-pass expansion can create many compound
    # micro-subclones when only a tiny subset of cells passes an extra-arm test.
    phase4_arm_expansion_enabled: bool = False
    phase4_arm_expansion_min_corr: float = 0.80       # per-cell r vs pooled arm profile
    phase4_arm_expansion_min_extreme: float = 0.85     # per-cell extreme-AF fraction
    phase4_arm_expansion_unanimous_frac: float = 0.95  # fraction of clone cells that must pass to relabel whole clone (else split)
    phase4_arm_expansion_min_cells: int = 5            # min cells carrying extra arm to accept expansion

    # Phase 4b – WT/Het → Hem/Hom ADO/somatic-amplicon check
    # Compare candidate WT/Het cells to Hem/Hom target profiles across the
    # full informative AF space, excluding only the amplicon that carries the
    # target somatic mutation. This is profile-similarity based only; no ADO
    # probability model is used.
    ado_check_enabled: bool = True
    ado_min_corr: float = 0.80
    ado_max_median_abs_diff: float = 15.0
    ado_min_comparison_variants: int = 20
    ado_exclude_somatic_amplicon: bool = True
    ado_include_hom_targets: bool = True
    ado_wt_compare_all_hemhom_loh_targets: bool = True
    # Retained for backward compatibility; not used by the new Phase 4b logic.
    ado_min_extreme_frac: float = 0.80

    # Phase 4c diagnostic – cross-LOH clone similarity/manual merge review.
    # This is diagnostic only: it prints yellow suggestions and writes
    # state.results["phase4_cross_loh_similarity"], but does not relabel cells.
    phase4_cross_loh_similarity_enabled: bool = True
    phase4_cross_loh_min_clone_size: int | None = None
    phase4_cross_loh_min_shared_variants: int = 4
    phase4_cross_loh_suggest_corr: float = 0.95
    phase4_cross_loh_suggest_median_abs_delta: float = 10.0
    phase4_cross_loh_suggest_extreme_frac: float = 0.85
    phase4_cross_loh_min_block_overlap: float = 0.50
    phase4_cross_loh_require_same_base: bool = False
    phase4_cross_loh_constant_profiles_are_match: bool = True

    # Phase 4d – automatic final small-LOH → larger-LOH merge.
    # This runs after Phase 4 annotation/ADO/cross-LOH diagnostics.  It is
    # stricter than the cross-LOH diagnostic and actually relabels cells only
    # when a small LOH clone has one unambiguous, larger, highly similar target.
    phase4_merge_small_loh_after_annotation_enabled: bool = True
    phase4_merge_small_loh_max_source_cells: int | None = None
    phase4_merge_small_loh_min_target_cells: int | None = None
    phase4_merge_small_loh_require_same_base: bool = True
    # LOH source arm-set must be a proper subset of the LOH target arm-set.
    # Equal-complexity and arm-losing merges are prohibited in the ordinary
    # Phase-4 workflow. Phase 6 may temporarily allow exact same-arm sibling
    # reabsorption into a larger target after strict cell-level validation.
    phase4_merge_small_loh_require_source_arms_subset: bool = True
    phase4_merge_small_loh_allow_equal_arm_set_to_larger_target: bool = False
    phase4_merge_small_loh_compare_all_target_arms: bool = True
    # LOH outlier reassignment. When require_larger_target is False, an LOH cell
    # may move to any same-base LOH sibling it strongly matches, regardless of
    # relative clone size, and the source-arms-subset rule is replaced by an
    # arm-overlap rule (the sibling must share at least one arm). The strict
    # per-cell similarity gates (corr/MAD/extreme/overlap/direction) still apply,
    # so only genuine outliers move. Set require_larger_target True to restore
    # the legacy merge-into-larger-only behaviour.
    phase4_merge_small_loh_loh_to_loh_require_larger_target: bool = True
    phase4_merge_small_loh_loh_to_loh_require_arm_overlap: bool = True
    phase4_merge_small_loh_min_shared_variants: int = 4
    phase4_merge_small_loh_min_corr: float = 0.95
    phase4_merge_small_loh_max_median_abs_delta: float = 10.0
    phase4_merge_small_loh_min_extreme_frac: float = 0.85
    phase4_merge_small_loh_min_block_overlap: float = 0.50
    phase4_merge_small_loh_ambiguous_corr_delta: float = 0.02
    phase4_merge_small_loh_constant_profiles_are_match: bool = True
    phase4_merge_small_loh_include_base_non_loh_target: bool = False
    # Also score cells from the same-base non-LOH parent clone against LOH clone
    # profiles. This promotes only individual base cells that match a retained
    # LOH pattern; it does not relabel the whole base clone.
    phase4_merge_small_loh_include_base_non_loh_sources: bool = False
    phase4_merge_small_loh_base_source_min_cells: int | None = None
    # Base non-LOH → LOH promotion is deliberately cell-level and uses its own
    # thresholds.  Single-cell Pearson against a clone mean is noisier than
    # clone-to-clone correlation, so these defaults are less strict than the
    # LOH→LOH cleanup while still requiring LOH-like AF over the target block.
    phase4_merge_base_to_loh_min_corr: float = 0.80
    phase4_merge_base_to_loh_max_median_abs_delta: float = 15.0
    phase4_merge_base_to_loh_min_cell_extreme_frac: float = 0.70
    phase4_merge_base_to_loh_min_target_extreme_frac: float = 0.85
    phase4_merge_base_to_loh_require_direction_match: bool = True
    phase4_merge_base_to_loh_min_matching_arms_frac: float = 1.00
    # Allow non-LOH Het cells to be promoted into same-mutation Hem/Hom LOH
    # clones when the only base-name difference is zygosity, e.g.
    # TP53_R248W_Het -> TP53_R248W_Hem LOH-C...
    phase4_merge_base_to_loh_allow_het_to_hem: bool = True
    phase4_merge_small_loh_base_target_require_low_extreme: bool = False
    phase4_merge_small_loh_base_target_max_cell_extreme_frac: float = 0.50
    phase4_merge_small_loh_base_target_max_target_extreme_frac: float = 0.50

    # Possible-LOH (FP) → LOH rescue. When enabled, cells in retained FP
    # clones are offered to the same cell-level promotion path as base cells:
    # an FP cell that individually matches a validated same-base LOH clone
    # over all of that clone's arms is moved into it; the rest stay FP.
    # Opt-in (legacy: FP stays inert).
    phase4_promote_fp_to_loh: bool = False
    phase4_promote_fp_min_source_cells: int = 1
    # FP cells are not allowed to lose or gain LOH arms during generic Phase
    # 4 rescue. Complex arm loss/addition must be resolved in Phase 3a/3b.
    phase4_promote_fp_require_exact_arm_set: bool = True

    # Phase 4e – residual base-clone LOH discovery.
    # After matching cells have been rescued to established LOH targets,
    # rescan the remaining same-base population for a de novo coherent LOH
    # subgroup. This can recover a genuine LOH subclone that has a distinct
    # arm/block and therefore cannot match an existing target profile.
    phase4_residual_base_loh_discovery_enabled: bool = False
    phase4_residual_base_loh_min_candidate_cells: int = 5
    phase4_residual_base_loh_min_contiguous_variants: int = 4
    phase4_residual_base_loh_min_extreme_fraction: float = 0.90
    phase4_residual_base_loh_min_shared_variants: int = 4
    phase4_residual_base_loh_min_mean_corr: float = 0.80
    phase4_residual_base_loh_max_cell_median_abs_diff: float = 15.0

    # Phase 4f – single conservative residual base-clone validation.
    # This runs once immediately before Phase 5. It only creates a new active
    # LOH clone at n>=min_final_clone_size; sub-threshold residual patterns
    # stay in their base clone and are never manufactured as de novo smalls.
    phase4_final_residual_refinement_enabled: bool = True
    phase4_final_residual_refinement_max_iterations: int = 1
    phase4_final_residual_refinement_min_active_cells: int = 5
    # Compatibility-only: ignored by the conservative one-pass implementation.
    phase4_final_residual_refinement_min_small_cells: int = 5
    phase4_final_residual_refinement_min_contiguous_variants: int = 4
    phase4_final_residual_refinement_min_extreme_fraction: float = 0.90
    phase4_final_residual_refinement_min_shared_variants: int = 4
    phase4_final_residual_refinement_min_mean_corr: float = 0.80
    phase4_final_residual_refinement_max_cell_median_abs_diff: float = 15.0
    phase4_final_residual_refinement_min_different_amplicons: int = 3
    phase4_final_residual_refinement_require_ado_unlikely: bool = True
    phase4_final_residual_refinement_ado_pvalue_threshold: float = 0.01
    phase4_final_residual_refinement_ado_rate_per_amplicon: float = 0.01

    # Phase 5 – final LOH clone cleanup before summary/plotting
    # Phase 4 can create or split LOH subclones after Phase 3 cleanup.
    # These options merge highly similar LOH subclones that share the same
    # base clone and LOH arm set, then rename remaining small LOH clones
    # to `phase5_small_loh_target_label` before Phase 5 summary and plotting.
    # This keeps tiny final LOH splits from being merged back into the baseline clone.
    # Phase 5 must not merge equal-complexity arm sets; that decision belongs
    # to Phase 4 diagnostics/reassignment. Disabled by default.
    phase5_merge_similar_loh_clones: bool = False
    phase5_merge_similar_loh_min_corr: float = 0.80
    # Every terminal label below min_final_clone_size is recoded as small:
    # <original label>, preserving provenance while excluding it from active
    # LOH / SUB / FP calculations.
    phase5_prune_all_small_clones: bool = True
    phase5_small_loh_target_label: str = SMALL_LABEL

    # Phase 5 – QC alerting for borderline-minimum LOH-block support.
    # A clone is retained if it passes min_different_amplicons, but if the
    # final LOH block is supported by exactly 3 independent informative
    # amplicons, Phase 5 prints a QC alert and writes it to phase5_summary.
    # The alert also includes a simple ADO-plausibility calculation: assuming
    # independent per-amplicon ADO at phase5_qc_alert_ado_rate_per_amplicon,
    # could the observed LOH subclone size arise within the original/base
    # clone lineage by all 3 amplicons dropping out together?
    phase5_alert_three_amplicon_loh_blocks: bool = True
    phase5_qc_alert_amplicons: int = 3
    phase5_qc_alert_explain_ado_plausibility: bool = True
    # Run/report the empirical ADO plausibility calculation for all final LOH
    # confidence tiers (low, mid, high), not only 3-amplicon boundary calls.
    phase5_qc_explain_ado_all_loh_confidences: bool = True
    # ADO plausibility can use empirical same-base non-LOH background rates
    # computed directly from the AF matrix for each boundary-support block.
    # This is the default because a fixed 20% amplicon ADO prior can be too low
    # for noisy panels/samples. Set phase5_qc_alert_ado_rate_source="fixed" to
    # force the old fixed-rate calculation.
    phase5_qc_alert_ado_rate_source: str = "empirical_background"
    phase5_qc_alert_ado_rate_per_amplicon: float = 0.20
    phase5_qc_alert_empirical_min_background_cells: int = 50
    phase5_qc_alert_empirical_use_dp: bool = True
    phase5_qc_alert_empirical_min_depth: int = 10
    phase5_qc_alert_empirical_amplicon_extreme_frac: float = 1.0
    phase5_qc_alert_empirical_beta_alpha: float = 1.0
    phase5_qc_alert_empirical_beta_beta: float = 9.0
    phase5_qc_alert_ado_unlikely_pvalue: float = 0.01

    # Phase 1b / 5 – amplicon validation
    validate_segment_length: bool = True
    min_contiguous_variants_final: int = 4
    min_different_amplicons: int = 3

    # Somatic variants may stay in the AF matrix for plotting/context, but they
    # do not count as germline-informative LOH support when cfg.informative_variants
    # is supplied.  A LOH block containing somatic/non-informative variants is
    # accepted only when the somatic positions are themselves extreme and the
    # block also contains enough germline-informative variants flanking the
    # somatic span.
    enforce_somatic_block_flanking: bool = True
    min_informative_variants_flanking_somatic: int = 3
    require_informative_on_both_sides_of_somatic: bool = True

    # Legacy/manual only: ADO-combination probability QC is disabled and not run
    # by the default Phase 5 workflow. No amplicon_ado_rates are required.
    ado_probability_qc_enabled: bool = False
    amplicon_ado_rates: Any = None
    ado_probability_rate_col: str = "smooth_ado_any_rate"
    ado_probability_amplicon_col: str = "amplicon_id"
    ado_probability_default_amplicon_rate: float = 0.01
    ado_probability_required_amplicons: int | None = None
    ado_probability_pvalue_threshold: float = 0.01
    # Probability QC is optional. The default workflow uses profile/block
    # similarity rescue only and does not require amplicon-level ADO rates.
    ado_probability_require_rates: bool = False

    # Plotting / external references
    informative_variants: list | None = None
    cytobands: Any = None
    output_dir: str | None = None
    save_pdf: bool = True  # also save every exported figure as .pdf

    # Verbosity
    verbose: bool = True

    # Diagnostic – keep clusters that fail intra-cluster coherence so
    # they flow through Phase 1 QC and appear in the plots.
    plot_dropped_clusters: bool = False


# ─────────────────────────────────────────────────────────────────────
# Shared pipeline state
# ─────────────────────────────────────────────────────────────────────
class PipelineState:
    """
    Holds all mutable data threaded through pipeline phases.

    Call ``refresh()`` after any modification to Sample clone labels so
    that derived mappings (barcode_to_clone, clone_to_cells) stay in sync.
    """

    def __init__(
        self,
        Sample,
        df: pd.DataFrame,
        grouped_by_chr_arm: dict,
        chr_arms: list[str],
        config: LOHConfig,
        df_dp: pd.DataFrame | None = None,
    ):
        self.Sample = Sample
        self.df = df
        self.df_dp = df_dp
        self.grouped_by_chr_arm = grouped_by_chr_arm
        self.chr_arms = chr_arms
        self.config = config
        self.results: dict = {}

        # Cytobands DataFrame (from config; used in cytoband-validated LOH detection)
        self.cytobands_df = config.cytobands

        # Barcodes – immutable reference series
        self.barcodes = pd.Series(Sample.dna.row_attrs["barcode"]).astype(str)

        # Variant-like columns (contain ':' in name, e.g. "chr1:12345:A/G")
        metadata_cols = {"barcode", "Clone", "Clone_backup"}
        self.variant_cols: list[str] = [
            c for c in df.columns if c not in metadata_cols and ":" in str(c)
        ]

        # Numeric AF matrix
        if len(self.variant_cols) == 0:
            self.df_numeric = pd.DataFrame(index=df.index)
        elif (
            df[self.variant_cols[0]].dtype == "object"
            or str(df[self.variant_cols[0]].dtype).startswith("str")
        ):
            self.df_numeric = df[self.variant_cols].apply(
                pd.to_numeric, errors="coerce"
            )
        else:
            self.df_numeric = df[self.variant_cols].copy()

        # Optional numeric DP matrix aligned to df_numeric.  This is used by
        # Phase 5 empirical ADO-like QC so low/no coverage is treated as
        # missingness, not as ADO.  If df_dp is not explicitly supplied, try to
        # recover it from the Tapestri Sample object.
        if df_dp is None:
            try:
                df_dp = Sample.dna.get_attribute("DP", constraint="row+col")
            except Exception:
                df_dp = None
        self.df_dp = df_dp
        if df_dp is None or len(self.variant_cols) == 0:
            self.df_dp_numeric = pd.DataFrame(index=df.index, columns=self.variant_cols, dtype=float)
        else:
            dp_tmp = df_dp.copy()
            # Align columns first; ignore metadata/label columns.
            dp_cols = [c for c in self.variant_cols if c in dp_tmp.columns]
            if dp_cols:
                # Align rows robustly.  If indexes already match, this is direct;
                # otherwise fall back to barcode column if available.
                if set(df.index).issubset(set(dp_tmp.index)):
                    dp_num = dp_tmp.reindex(index=df.index, columns=self.variant_cols)
                elif "barcode" in dp_tmp.columns and "barcode" in df.columns:
                    dp_num = (
                        dp_tmp.set_index(dp_tmp["barcode"].astype(str))
                        .reindex(df["barcode"].astype(str).values)[dp_cols]
                    )
                    dp_num.index = df.index
                    dp_num = dp_num.reindex(columns=self.variant_cols)
                else:
                    dp_num = dp_tmp.reindex(index=df.index, columns=self.variant_cols)
                self.df_dp_numeric = dp_num.apply(pd.to_numeric, errors="coerce")
            else:
                self.df_dp_numeric = pd.DataFrame(index=df.index, columns=self.variant_cols, dtype=float)

        # Cell <-> barcode mapping (stable across phases)
        bc_set = set(self.barcodes.values.astype(str))
        if str(df.index[0]) in bc_set:
            self.cell_to_barcode = {c: str(c) for c in df.index}
        elif "barcode" in df.columns:
            self.cell_to_barcode = df["barcode"].astype(str).to_dict()
        else:
            self.cell_to_barcode = dict(zip(df.index, self.barcodes.astype(str)))

        # Barcode -> positional index in Sample arrays (for fast label updates)
        self.barcode_to_sample_idx: dict[str, int] = {
            str(bc): i for i, bc in enumerate(self.barcodes.values.astype(str))
        }

        # Clone mappings – populated by refresh()
        self.barcode_to_clone: dict[str, str] = {}
        self.clone_to_cells: dict[str, list] = defaultdict(list)
        self.refresh()

    def refresh(self) -> None:
        """Re-derive barcode_to_clone and clone_to_cells from current labels."""
        current = pd.Series(self.Sample.dna.row_attrs["Clone"]).astype(str)
        self.barcode_to_clone = dict(
            zip(self.barcodes.astype(str), current.astype(str))
        )
        c2c: dict[str, list] = defaultdict(list)
        for cell_id in self.df.index:
            bc = self.cell_to_barcode.get(cell_id)
            if bc is None:
                continue
            cl = self.barcode_to_clone.get(str(bc), "")
            c2c[cl].append(cell_id)
        self.clone_to_cells = c2c

    def set_clone_labels(self, new_labels) -> None:
        """Overwrite Sample clone labels and refresh all mappings."""
        if isinstance(new_labels, pd.Series):
            self.Sample.dna.row_attrs["Clone"] = new_labels.values
        else:
            self.Sample.dna.row_attrs["Clone"] = new_labels
        self.refresh()

    @property
    def current_clones(self) -> pd.Series:
        return pd.Series(self.Sample.dna.row_attrs["Clone"]).astype(str)

    def sample_idx_for_barcode(self, barcode: str) -> int | None:
        """Positional index in the Sample arrays for a barcode string."""
        return self.barcode_to_sample_idx.get(str(barcode))

    def loh_clone_labels(self) -> list[str]:
        """Return active LOH labels, excluding review-only ``FP:`` and ``SUB:`` labels."""
        return [c for c in self.clone_to_cells.keys() if is_loh_label(c)]

    def loh_cell_count(self) -> int:
        """Number of active LOH cells, excluding review-only FP/SUB labels."""
        return int(self.current_clones.apply(is_loh_label).sum())


# ─────────────────────────────────────────────────────────────────────
# Helper functions (stateless)
# ─────────────────────────────────────────────────────────────────────
def vprint(msg: str, cfg: LOHConfig) -> None:
    """Verbose print – only outputs when cfg.verbose is True."""
    if cfg.verbose:
        print(msg)
        sys.stdout.flush()


def edge_tolerant_extreme_runs(
    extreme_mask,
    max_edge_tolerance: int = 0,
) -> list[dict[str, list[int] | int]]:
    """
    Return contiguous extreme cores with optional *terminal-edge* padding.

    The helper deliberately does **not** bridge internal gaps.  A pattern
    ``[1, 1, 0, 1, 1]`` therefore remains two blocks, whereas terminal
    patterns ``[1, 1, 0]`` and ``[0, 1, 1]`` can each absorb the single
    flanking non-extreme variant when ``max_edge_tolerance=1``.

    Each record has ``core_indices`` (strict extreme support),
    ``span_indices`` (core plus permitted terminal padding), and
    ``n_edge_padded``.  Downstream QC should use the core for AF purity and
    independent-amplicon support, while it may use the span for block length.
    """
    arr = np.asarray(extreme_mask, dtype=bool).ravel()
    n = int(arr.size)
    if n == 0:
        return []

    cores: list[tuple[int, int]] = []
    start = None
    for i, value in enumerate(arr):
        if value and start is None:
            start = i
        elif (not value) and start is not None:
            cores.append((start, i - 1))
            start = None
    if start is not None:
        cores.append((start, n - 1))
    if not cores:
        return []

    tol = max(0, int(max_edge_tolerance or 0))
    if tol == 0:
        return [
            {
                "core_indices": list(range(left, right + 1)),
                "span_indices": list(range(left, right + 1)),
                "n_edge_padded": 0,
            }
            for left, right in cores
        ]

    # Candidate pads are generated independently, then any contested index
    # (the single gap between two extreme cores) is removed from both blocks.
    candidates: list[set[int]] = []
    for k, (left, right) in enumerate(cores):
        prev_end = cores[k - 1][1] if k > 0 else -1
        next_start = cores[k + 1][0] if k + 1 < len(cores) else n
        pad: set[int] = set()
        for j in range(left - 1, max(prev_end, left - tol - 1), -1):
            if j < 0 or arr[j]:
                break
            pad.add(j)
        for j in range(right + 1, min(next_start, right + tol + 1)):
            if j >= n or arr[j]:
                break
            pad.add(j)
        candidates.append(pad)

    usage: dict[int, int] = {}
    for pad in candidates:
        for idx in pad:
            usage[idx] = usage.get(idx, 0) + 1

    out: list[dict[str, list[int] | int]] = []
    for (left, right), pad in zip(cores, candidates):
        clean_pad = {idx for idx in pad if usage.get(idx, 0) == 1}
        core = list(range(left, right + 1))
        span = sorted(set(core).union(clean_pad))
        out.append({
            "core_indices": core,
            "span_indices": span,
            "n_edge_padded": len(clean_pad),
        })
    return out


def sort_chromosome_arms(chr_list: list[str]) -> list[str]:
    """Sort chromosome arm labels naturally: chr1p, chr1q, chr2p, ..."""
    def _key(chr_arm):
        m = re.match(r"chr(\d+|X|Y)([pq])", str(chr_arm))
        if not m:
            return (999, 0, chr_arm)
        num, arm = m.groups()
        if num == "X":
            chr_order = 23
        elif num == "Y":
            chr_order = 24
        else:
            chr_order = int(num)
        return (chr_order, 0 if arm == "p" else 1, chr_arm)
    return sorted(chr_list, key=_key)


def get_base_clone(clone_label: str) -> str:
    """Strip all ' LOH-C<n> <arm>' suffixes from a clone label."""
    return re.sub(r" LOH-C\d+ \S+", "", str(clone_label))


def extract_chr_arms(clone_label: str) -> list[str]:
    """Return all chrNp / chrNq tokens in a clone label."""
    return [
        m.group(1)
        for m in re.finditer(r"(chr\d+[pq]|chrX[pq]|chrY[pq])", str(clone_label))
    ]


def get_chr_number(chr_arm: str) -> str | None:
    """Extract chromosome number/letter from an arm label."""
    m = re.match(r"chr(\d+|X|Y)", str(chr_arm))
    return m.group(1) if m else None


def is_wt_clone(label: str, cfg: LOHConfig) -> bool:
    """
    Check whether *label* should be treated as a wild-type clone.

    By default a clone is WT if its label starts with ``"WT"``.  A deprecated
    runtime attribute ``cfg.wt_labels`` is still honoured if present.
    """
    s = str(label)
    if s.startswith("WT"):
        return True
    wt_labels = getattr(cfg, "wt_labels", None)
    if wt_labels:
        return any(s.startswith(prefix) for prefix in wt_labels)
    return False


def is_hem_like(label: str) -> bool:
    """Detect Hem/HEM/hem even when delimited by underscores/punctuation."""
    return (
        re.search(
            r"(^|[^A-Za-z0-9])hem([^A-Za-z0-9]|$)", str(label), flags=re.IGNORECASE
        )
        is not None
    )


def arm_variants(arm: str, state: PipelineState) -> list[str]:
    """Variants for *arm* that exist as columns in state.df_numeric."""
    return [
        v for v in state.grouped_by_chr_arm.get(arm, [])
        if v in state.df_numeric.columns
    ]


def pearson_safe(x, y, min_n: int = 5) -> float | None:
    """Pearson r returning None instead of NaN for degenerate inputs."""
    if x is None or y is None:
        return None
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < min_n:
        return None
    xx, yy = x[mask], y[mask]
    if np.nanstd(xx) == 0 or np.nanstd(yy) == 0:
        return None
    r, _ = pearsonr(xx, yy)
    if isinstance(r, float) and (np.isnan(r) or np.isinf(r)):
        return None
    return float(r)


def check_directional_consistency(
    af_block: np.ndarray,
    low_thr: float,
    high_thr: float,
    min_per_var_agreement: float = 0.85,
    min_block_pass_frac: float = 0.80,
    min_extreme_cells_per_var: int = 3,
) -> dict:
    """
    Per-variant directional cleanliness check for an LOH block.

    For each variant column in *af_block* (cells × variants):
      • count cells that are extreme LOW (≤ low_thr) and extreme HIGH (≥ high_thr)
      • if total extreme cells ≥ min_extreme_cells_per_var,
        compute agreement = max(low_count, high_count) / (low_count + high_count)
      • the variant "passes" if agreement ≥ min_per_var_agreement
    Mid-range cells are treated as missing/ADO and don't vote.

    The block "passes" overall if at least min_block_pass_frac of variants
    with enough extreme cells pass the per-variant check.

    Returns
    -------
    dict with keys:
        passes : bool
        n_variants_evaluated : int
        n_variants_passing : int
        block_pass_frac : float
        per_variant_agreements : list[float]   # NaN if too few extreme cells
        per_variant_directions : list[str]     # 'LOW', 'HIGH', or 'NA'
        dominant_direction : str               # 'LOW', 'HIGH', 'MIXED', or 'NA'
    """
    af = np.asarray(af_block, dtype=float)
    if af.ndim != 2 or af.size == 0:
        return {
            "passes": False,
            "n_variants_evaluated": 0,
            "n_variants_passing": 0,
            "block_pass_frac": float("nan"),
            "per_variant_agreements": [],
            "per_variant_directions": [],
            "dominant_direction": "NA",
        }

    n_vars = af.shape[1]
    agreements: list[float] = []
    directions: list[str] = []
    n_passing = 0
    n_evaluated = 0
    low_dom = 0
    high_dom = 0

    for j in range(n_vars):
        col = af[:, j]
        finite = col[np.isfinite(col)]
        n_low = int(np.sum(finite <= low_thr))
        n_high = int(np.sum(finite >= high_thr))
        n_ext = n_low + n_high

        if n_ext < min_extreme_cells_per_var:
            agreements.append(float("nan"))
            directions.append("NA")
            continue

        n_evaluated += 1
        agree = max(n_low, n_high) / n_ext
        agreements.append(float(agree))

        if n_low > n_high:
            directions.append("LOW")
            if agree >= min_per_var_agreement:
                low_dom += 1
                n_passing += 1
        elif n_high > n_low:
            directions.append("HIGH")
            if agree >= min_per_var_agreement:
                high_dom += 1
                n_passing += 1
        else:
            directions.append("TIE")

    if n_evaluated == 0:
        block_pass_frac = float("nan")
        passes = False
        dominant = "NA"
    else:
        block_pass_frac = n_passing / n_evaluated
        passes = block_pass_frac >= min_block_pass_frac
        if low_dom > 0 and high_dom == 0:
            dominant = "LOW"
        elif high_dom > 0 and low_dom == 0:
            dominant = "HIGH"
        elif low_dom == 0 and high_dom == 0:
            dominant = "NA"
        else:
            dominant = "MIXED"

    return {
        "passes": passes,
        "n_variants_evaluated": n_evaluated,
        "n_variants_passing": n_passing,
        "block_pass_frac": block_pass_frac,
        "per_variant_agreements": agreements,
        "per_variant_directions": directions,
        "dominant_direction": dominant,
    }


def mean_profile(
    df_numeric: pd.DataFrame, cells: list, variants: list
) -> np.ndarray | None:
    """Mean AF profile across *cells* for *variants*; None if empty."""
    if not cells or not variants:
        return None
    rows = [
        df_numeric.loc[c, variants].values.astype(float)
        for c in cells
        if c in df_numeric.index
    ]
    if not rows:
        return None
    m = np.nanmean(np.vstack(rows), axis=0)
    return m if np.any(np.isfinite(m)) else None


def fmt_float(x, nd: int = 3) -> str:
    """Format a float for display; returns 'NA' for None/NaN/Inf."""
    try:
        if x is None:
            return "NA"
        if isinstance(x, float) and (np.isnan(x) or np.isinf(x)):
            return "NA"
        return f"{float(x):.{nd}f}"
    except Exception:
        return "NA"


def fmt_pct(x) -> str:
    """Format a fraction as a percentage string; returns 'NA' for bad values."""
    try:
        if x is None:
            return "NA"
        if isinstance(x, float) and (np.isnan(x) or np.isinf(x)):
            return "NA"
        return f"{100.0 * float(x):.1f}%"
    except Exception:
        return "NA"


def dedupe_preserve_order(items) -> list[str]:
    """De-duplicate a list while preserving insertion order."""
    return list(dict.fromkeys([str(x) for x in items if str(x).strip() != ""]))


def build_cytoband_maps(
    variants_sorted: list[str],
    cytobands_df: pd.DataFrame | None,
    target_chr: str,
) -> tuple[dict[int, str] | None, dict[str, list[int]] | None]:
    """
    Build bidirectional variant-index <-> cytoband mappings for one chromosome arm.

    cytobands_df must have columns: 'chromosome arm', 'start', 'end', 'cytoband'.
    """
    if cytobands_df is None or len(cytobands_df) == 0 or len(variants_sorted) == 0:
        return None, None

    arm_bands = cytobands_df[cytobands_df["chromosome arm"] == target_chr]
    if len(arm_bands) == 0:
        return None, None

    intervals = [
        (int(row["start"]), int(row["end"]), str(row["cytoband"]))
        for _, row in arm_bands.iterrows()
    ]

    var_idx_to_band: dict[int, str] = {}
    band_to_var_indices: dict[str, list[int]] = defaultdict(list)

    for idx, variant in enumerate(variants_sorted):
        try:
            pos = int(str(variant).split(":")[1].split("-")[0])
        except (IndexError, ValueError):
            continue
        for band_start, band_end, band_name in intervals:
            if band_start <= pos <= band_end:
                var_idx_to_band[idx] = band_name
                band_to_var_indices[band_name].append(idx)
                break

    if not var_idx_to_band:
        return None, None

    return var_idx_to_band, dict(band_to_var_indices)
