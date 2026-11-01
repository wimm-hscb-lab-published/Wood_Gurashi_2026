"""Somatic variant discovery and its visualisation.

Main entry points
-----------------
find_somatic_variants
    Select candidate somatic variants from a cells × variants AF table by
    combining Spearman correlation with whitelist targets, a WT vs non-WT
    Mann-Whitney test, a within-WT subclone scan and forced inclusion of
    whitelist / (likely) pathogenic variants, then germline / intronic /
    synonymous filtering.
visualize_found_differential_variants
    Per-label clustered heatmaps plus a protein-UMAP grid with one panel per
    variant.
visualize_found_informative_variants
    Per-label clustered heatmaps for an informative-variant list.
plot_somatic_variant_umap_grid
    Plotly grid: protein UMAP coloured by annotation, then one carrier /
    background UMAP panel per variant.
"""

import numpy as np
import pandas as pd

from .utils import extract_chr, extract_chr_num
from .germline import get_real_somatic_variants


# ═══════════════════════════════════════════════════════════════════════════════
# find_somatic_variants
# ═══════════════════════════════════════════════════════════════════════════════

def find_somatic_variants(
    df: pd.DataFrame,
    whitelist: list,
    filtered_germline_in_samples,
    Sample,
    annotation_df: pd.DataFrame = None,
    always_keep_pathogenic: bool = True,
    pathogenic_bypass_germline_filter: bool = False,
    whitelist_bypass_germline_filter: bool = True,
    pathogenicity_cols: list = None,
    wt_label: str = "WT",
    significance_threshold: float = 0.05,
    correction_method: str = "fdr_bh",
    min_samples: int = 3,
    min_carrier_frac: float = 0.10,
    min_het_hom_frac: float = 0.15,
    min_wt_subclone_carrier_frac: float = 0.01,
    max_wt_subclone_carrier_frac: float = 0.50,
    max_nonwt_carrier_frac: float = 0.05,
    min_wt_subclone_carrier_count: int = 5,
    min_zero_mutant_carrier_af: float = 30.0,
    trace_variants: list = None,
    verbose: bool = True,
):
    """Find somatic variants through correlation and differential testing.

    Whitelist variants detected in ``df`` are force-retained when
    ``whitelist_bypass_germline_filter=True``. This bypasses germline,
    intronic, and synonymous filtering, but cannot add variants absent from
    the AF dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Cells × variants AF table (AF in percent, 0–100). The *first* column
        is taken as the cell label; every other column is a variant.
    whitelist : list
        Target variant IDs (``chr:pos:ref/alt``, optionally prefixed
        ``Gene:``). Matched to ``df`` columns on the canonical
        ``chr...`` part of the ID. None is treated as empty.
    filtered_germline_in_samples
        Germline variant IDs, forwarded to
        :func:`DNA_Utils.germline.get_real_somatic_variants`.
    Sample
        Mosaic Sample, forwarded to ``get_real_somatic_variants``.
    annotation_df : pd.DataFrame, optional
        Variant annotations, keyed by a ``'Variant ID'`` column or by the
        index. Used for pathogenic force-inclusion and for
        ``Function`` / ``Coding impact`` filtering.
    always_keep_pathogenic : bool
        Add detected variants annotated (likely) pathogenic to the candidates.
    pathogenic_bypass_germline_filter : bool
        Also protect those pathogenic variants from the germline, intronic and
        synonymous filters.
    whitelist_bypass_germline_filter : bool
        Protect detected whitelist variants from those filters.
    pathogenicity_cols : list, optional
        Annotation columns to scan for pathogenicity. None auto-selects
        columns whose name contains ``pathogen``, ``clinvar``, ``clnsig``,
        ``significance`` or ``acmg``.
    wt_label : str
        Label value identifying wild-type cells (default ``'WT'``).
    significance_threshold : float
        Adjusted-p cutoff for all three tests (default 0.05).
    correction_method : str
        ``statsmodels.multipletests`` method (default ``'fdr_bh'``).
    min_samples : int
        Minimum non-NaN cells per group / pair for a test (default 3).
    min_carrier_frac : float
        Minimum ``|carrier_frac_mut - carrier_frac_wt|`` for a Wilcoxon hit
        (default 0.10).
    min_het_hom_frac : float
        Minimum carrier fraction in non-WT cells for a Wilcoxon hit
        (default 0.15).
    min_wt_subclone_carrier_frac, max_wt_subclone_carrier_frac : float
        WT carrier-fraction window for the subclone scan (0.01–0.50).
    max_nonwt_carrier_frac : float
        Maximum non-WT carrier fraction for the subclone scan (0.05).
    min_wt_subclone_carrier_count : int
        Absolute WT carrier count that passes the subclone scan's lower bound
        on its own, and the count needed for the zero-in-mutant rescue
        (default 5).
    min_zero_mutant_carrier_af : float
        Minimum mean carrier AF for the zero-in-mutant rescue (default 30.0).
    trace_variants : list, optional
        Exact ``df`` column names whose fate is printed at a few checkpoints
        (printed even when ``verbose=False``).
    verbose : bool
        Print progress and result tables.

    Returns
    -------
    differential_variants : np.ndarray of str
        Final variant column names (all present in ``df``).
    significant_corr_results : pd.DataFrame
        Significant Spearman tests (``target, column, n, rho, rho_squared,
        p_value, bonferroni_p, adj_p``), sorted by ``adj_p``; empty if none
        were run.
    significant_wilcox_results : pd.DataFrame
        Mann-Whitney hits passing the significance, carrier-difference and
        mutant carrier-fraction filters; empty if none / skipped.

    Notes
    -----
    A cell is a *carrier* of a variant when its AF is > 10. Steps:

    0. Pathogenic force-include: a value counts as pathogenic when it matches
       the whole word ``pathogenic`` or ``likely[_- ]*pathogenic`` and not
       the whole phrase ``(no|not|non)[_- ]*pathogenic`` (lower-cased).
    1. Spearman between each detected whitelist target and every
       non-target variant on cells where both are non-NaN (``n >=
       min_samples``); multiple-testing correction over all pairs.
    2. Mann-Whitney U (two-sided), WT vs non-WT, per variant (skipped if
       either group has < 2 cells); NaN p-values become 1.0 before
       correction. A hit needs ``adj_p < threshold``, ``|carrier_diff| >=
       min_carrier_frac`` and ``carrier_frac_mut >= min_het_hom_frac``.
    2b. Within-WT subclone scan (only if >= 10 WT cells, and per variant
       >= 10 non-NaN WT cells): keep variants with (WT carrier fraction >=
       min frac OR WT carriers >= min count), WT fraction <= max frac, at
       least 3 WT carriers and non-WT carrier fraction <= max; one-sided
       Fisher exact test (WT enriched) and correction. Non-significant
       variants are rescued when non-WT carrier fraction is 0, WT carriers >=
       ``min_wt_subclone_carrier_count`` and mean carrier AF >=
       ``min_zero_mutant_carrier_af``.
    3. Union of hits from 1, 2, 2b, detected whitelist and forced pathogenic,
       sorted by chromosome number (stable sort on an alphabetical list), then
       ``get_real_somatic_variants``; protected variants it removed are added
       back.
    4. If ``annotation_df`` is given: drop variants whose ``Function``
       contains ``intron`` or whose ``Coding impact`` contains
       ``synonymous``, unless protected. Finally keep only exact ``df``
       column names. Bonferroni columns are reported but not used for
       filtering. The within-WT subclone hits feed the union but are not
       returned.
    """
    from scipy.stats import spearmanr, mannwhitneyu, fisher_exact
    from statsmodels.stats.multitest import multipletests

    whitelist = list(whitelist or [])

    def vprint(msg):
        if verbose:
            print(msg)

    def canonical_variant_key(variant):
        """Convert optional Gene:chr:pos:ref/alt IDs to chr:pos:ref/alt."""
        parts = str(variant).strip().split(":")

        for i, part in enumerate(parts):
            if part.lower().startswith("chr"):
                return ":".join(parts[i:])

        return str(variant).strip()

    _trace = set(trace_variants) if trace_variants else set()

    def _tr(variant, msg):
        if variant in _trace:
            print(f"  🔍 TRACE {variant}: {msg}")

    vprint("\n" + "=" * 60)
    vprint("FINDING SOMATIC VARIANTS")
    vprint("=" * 60)

    label_col = df.columns[0]
    variant_cols = [c for c in df.columns if c != label_col]

    # Map canonical variant IDs back to the exact column names in df.
    variant_column_by_key = {}

    for variant in variant_cols:
        key = canonical_variant_key(variant)

        if key not in variant_column_by_key:
            variant_column_by_key[key] = variant

    whitelist_keys = {
        canonical_variant_key(variant)
        for variant in whitelist
    }

    # Only retain whitelist variants that have AF data / exist as df columns.
    detected_whitelist = {
        variant_column_by_key[key]
        for key in whitelist_keys
        if key in variant_column_by_key
    }

    vprint(f"  Label column: '{label_col}'")
    vprint(f"  Variant columns: {len(variant_cols)}")
    vprint(f"  Unique labels: {sorted(df[label_col].astype(str).unique())}")
    vprint(
        f"  Whitelist targets detected in df: "
        f"{len(detected_whitelist)} / {len(whitelist)}"
    )

    for tv in _trace:
        if tv in variant_cols:
            _tr(tv, f"found in df columns ({len(variant_cols)} variants)")
        else:
            _tr(tv, "NOT FOUND in df columns — cannot be detected")

    # ==================================================================
    # 0. (LIKELY) PATHOGENIC FORCE-INCLUDE
    # ==================================================================

    forced_pathogenic = set()

    if (
        always_keep_pathogenic
        and annotation_df is not None
        and not annotation_df.empty
    ):
        anno = annotation_df.copy()

        annotation_variant_ids = (
            anno["Variant ID"]
            if "Variant ID" in anno.columns
            else anno.index
        )

        annotation_variant_ids = (
            annotation_variant_ids
            .astype(str)
            .str.strip()
        )

        if pathogenicity_cols is None:
            candidate_pathogenicity_cols = [
                column
                for column in anno.columns
                if any(
                    token in str(column).lower()
                    for token in [
                        "pathogen",
                        "clinvar",
                        "clnsig",
                        "significance",
                        "acmg",
                    ]
                )
            ]
        else:
            candidate_pathogenicity_cols = [
                column
                for column in pathogenicity_cols
                if column in anno.columns
            ]

        pathogenic_mask = pd.Series(False, index=anno.index)

        for column in candidate_pathogenicity_cols:
            values = anno[column].astype(str).str.lower()

            negative = values.str.contains(
                r"\b(?:no|not|non)[_\-\s]*pathogenic\b",
                regex=True,
                na=False,
            )

            positive = values.str.contains(
                r"\bpathogenic\b|likely[_\-\s]*pathogenic",
                regex=True,
                na=False,
            )

            pathogenic_mask = pathogenic_mask | (positive & ~negative)

        pathogenic_keys = {
            canonical_variant_key(variant)
            for variant in annotation_variant_ids[pathogenic_mask]
        }

        forced_pathogenic = {
            variant_column_by_key[key]
            for key in pathogenic_keys
            if key in variant_column_by_key
        }

        vprint(
            "  Force-include (likely) pathogenic detected variants: "
            f"{len(forced_pathogenic)} "
            f"(from columns: {candidate_pathogenicity_cols})"
        )

        for tv in _trace:
            _tr(
                tv,
                "pathogenic force-include: "
                f"{'YES' if tv in forced_pathogenic else 'no'}",
            )

    # ==================================================================
    # 1. SPEARMAN CORRELATION AGAINST WHITELIST TARGETS
    # ==================================================================

    vprint("\n" + "-" * 60)
    vprint("PART 1: SPEARMAN CORRELATION")
    vprint("-" * 60)

    targets = [
        variant_column_by_key[canonical_variant_key(target)]
        for target in whitelist
        if canonical_variant_key(target) in variant_column_by_key
    ]

    targets = list(dict.fromkeys(targets))

    vprint(
        f"  Whitelist targets found in df: "
        f"{len(targets)} / {len(whitelist)}"
    )

    correlation_rows = []
    exclude = set(targets) | {label_col}

    for target in targets:
        x = pd.to_numeric(df[target], errors="coerce")

        for column in (c for c in variant_cols if c not in exclude):
            y = pd.to_numeric(df[column], errors="coerce")

            mask = x.notna() & y.notna()
            n = int(mask.sum())

            if n < min_samples:
                continue

            rho, p_value = spearmanr(
                x[mask].to_numpy(),
                y[mask].to_numpy(),
            )

            correlation_rows.append(
                {
                    "target": target,
                    "column": column,
                    "n": n,
                    "rho": rho,
                    "rho_squared": rho ** 2,
                    "p_value": p_value,
                }
            )

    if not correlation_rows:
        vprint("  No correlation results (not enough overlapping data).")

        corr_results = pd.DataFrame()
        significant_corr_results = pd.DataFrame()

    else:
        corr_results = (
            pd.DataFrame(correlation_rows)
            .sort_values(["target", "p_value"])
        )

        n_tests = len(corr_results)

        corr_results["bonferroni_p"] = np.minimum(
            corr_results["p_value"] * n_tests,
            1.0,
        )

        _, adjusted_p, _, _ = multipletests(
            corr_results["p_value"],
            alpha=significance_threshold,
            method=correction_method,
        )

        corr_results["adj_p"] = adjusted_p

        significant_corr_results = (
            corr_results
            .query("adj_p < @significance_threshold")
            .sort_values("adj_p")
        )

        vprint(f"  Tests performed: {n_tests}")
        vprint(
            f"  Significant ({correction_method} < "
            f"{significance_threshold}): "
            f"{len(significant_corr_results)}"
        )

        if verbose and not significant_corr_results.empty:
            vprint(
                significant_corr_results[
                    ["target", "column", "rho", "adj_p", "n"]
                ]
                .head(15)
                .to_string(index=False)
            )

    # ==================================================================
    # 2. MANN-WHITNEY U: WT VS NON-WT
    # ==================================================================

    vprint("\n" + "-" * 60)
    vprint("PART 2: MANN-WHITNEY U  (WT vs non-WT)")
    vprint("-" * 60)

    labels = df[label_col].astype(str)

    is_wt = labels == str(wt_label)
    n_wt = int(is_wt.sum())
    n_mut = int((~is_wt).sum())

    vprint(f"  WT cells ('{wt_label}'): {n_wt}")
    vprint(f"  Non-WT cells: {n_mut}")

    if n_wt < 2 or n_mut < 2:
        vprint("  ⚠️  Not enough cells in one group — skipping Wilcoxon.")

        significant_wilcox_results = pd.DataFrame()

    else:
        wt_df = df.loc[is_wt]
        mut_df = df.loc[~is_wt]

        wilcox_rows = []

        for column in variant_cols:
            x_wt = pd.to_numeric(
                wt_df[column],
                errors="coerce",
            ).dropna()

            x_mut = pd.to_numeric(
                mut_df[column],
                errors="coerce",
            ).dropna()

            if len(x_wt) < min_samples or len(x_mut) < min_samples:
                continue

            carrier_frac_wt = (x_wt > 10).mean()
            carrier_frac_mut = (x_mut > 10).mean()

            carrier_diff = carrier_frac_mut - carrier_frac_wt

            mean_af_wt = x_wt.mean()
            mean_af_mut = x_mut.mean()
            mean_af_diff = mean_af_mut - mean_af_wt

            if mean_af_wt > 0:
                lfc = np.log2(
                    (mean_af_mut + 0.01) /
                    (mean_af_wt + 0.01)
                )
            else:
                lfc = np.nan

            try:
                _, p_value = mannwhitneyu(
                    x_wt.to_numpy(),
                    x_mut.to_numpy(),
                    alternative="two-sided",
                )
            except Exception:
                p_value = np.nan

            wilcox_rows.append(
                {
                    "column": column,
                    "n_wt": len(x_wt),
                    "n_mut": len(x_mut),
                    "mean_af_wt": mean_af_wt,
                    "mean_af_mut": mean_af_mut,
                    "mean_af_diff": mean_af_diff,
                    "carrier_frac_wt": carrier_frac_wt,
                    "carrier_frac_mut": carrier_frac_mut,
                    "carrier_diff": carrier_diff,
                    "lfc": lfc,
                    "p_value": p_value,
                }
            )

        if not wilcox_rows:
            vprint("  No tests could be run.")

            significant_wilcox_results = pd.DataFrame()

        else:
            wilcox_results = pd.DataFrame(wilcox_rows)

            valid_p = (
                wilcox_results["p_value"]
                .fillna(1.0)
                .to_numpy()
            )

            wilcox_results["bonferroni_p"] = np.minimum(
                valid_p * len(valid_p),
                1.0,
            )

            _, adjusted_p, _, _ = multipletests(
                valid_p,
                alpha=significance_threshold,
                method=correction_method,
            )

            wilcox_results["adj_p"] = adjusted_p
            wilcox_results = wilcox_results.sort_values("adj_p")

            significance_mask = (
                wilcox_results["adj_p"] < significance_threshold
            )

            carrier_mask = (
                wilcox_results["carrier_diff"].abs()
                >= min_carrier_frac
            )

            het_hom_mask = (
                wilcox_results["carrier_frac_mut"]
                >= min_het_hom_frac
            )

            significant_wilcox_results = wilcox_results[
                significance_mask
                & carrier_mask
                & het_hom_mask
            ].copy()

            vprint(f"  Tests performed: {len(wilcox_results)}")
            vprint(
                f"  Significant (adj_p < {significance_threshold}): "
                f"{int(significance_mask.sum())}"
            )
            vprint(
                f"  + carrier-diff ≥ {min_carrier_frac:.0%}: "
                f"{int((significance_mask & carrier_mask).sum())}"
            )
            vprint(
                f"  + carrier-frac in mutant ≥ {min_het_hom_frac:.0%}: "
                f"{len(significant_wilcox_results)}"
            )

            if verbose and not significant_wilcox_results.empty:
                vprint(
                    significant_wilcox_results[
                        [
                            "column",
                            "mean_af_wt",
                            "mean_af_mut",
                            "carrier_diff",
                            "carrier_frac_mut",
                            "adj_p",
                        ]
                    ]
                    .head(20)
                    .to_string(index=False)
                )

    # ==================================================================
    # 2B. WITHIN-WT SUBCLONAL VARIANT SCAN
    # ==================================================================

    vprint("\n" + "-" * 60)
    vprint("PART 2b: WITHIN-WT SUBCLONAL VARIANT SCAN")
    vprint("-" * 60)

    vprint(
        f"  Looking for variants carried by "
        f"{min_wt_subclone_carrier_frac:.0%}–"
        f"{max_wt_subclone_carrier_frac:.0%} of WT cells "
        f"(or ≥ {min_wt_subclone_carrier_count} absolute carriers)"
    )

    vprint(
        f"  and by ≤ {max_nonwt_carrier_frac:.0%} of non-WT cells"
    )

    wt_subclone_hits = []

    if n_wt >= 10:
        wt_afs = df.loc[is_wt]
        mut_afs = df.loc[~is_wt]

        for column in variant_cols:
            x_wt = pd.to_numeric(
                wt_afs[column],
                errors="coerce",
            ).dropna()

            if len(x_wt) < 10:
                continue

            wt_carriers = int((x_wt > 10).sum())
            wt_carrier_frac = wt_carriers / len(x_wt)

            x_mut = pd.to_numeric(
                mut_afs[column],
                errors="coerce",
            ).dropna()

            mut_carrier_frac = (
                (x_mut > 10).mean()
                if len(x_mut) > 0
                else 0.0
            )

            passes_fraction = (
                wt_carrier_frac >= min_wt_subclone_carrier_frac
            )

            passes_count = (
                wt_carriers >= min_wt_subclone_carrier_count
            )

            if not passes_fraction and not passes_count:
                continue

            if wt_carrier_frac > max_wt_subclone_carrier_frac:
                continue

            if wt_carriers < 3:
                continue

            if mut_carrier_frac > max_nonwt_carrier_frac:
                continue

            wt_noncarriers = len(x_wt) - wt_carriers

            mut_carriers = (
                int((x_mut > 10).sum())
                if len(x_mut) > 0
                else 0
            )

            mut_noncarriers = (
                len(x_mut) - mut_carriers
                if len(x_mut) > 0
                else 0
            )

            contingency_table = [
                [wt_carriers, wt_noncarriers],
                [mut_carriers, mut_noncarriers],
            ]

            try:
                odds_ratio, p_value = fisher_exact(
                    contingency_table,
                    alternative="greater",
                )
            except Exception:
                odds_ratio = np.nan
                p_value = 1.0

            carrier_mean_af = (
                x_wt[x_wt > 10].mean()
                if wt_carriers > 0
                else 0.0
            )

            wt_subclone_hits.append(
                {
                    "column": column,
                    "wt_carriers": wt_carriers,
                    "wt_total": len(x_wt),
                    "wt_carrier_frac": wt_carrier_frac,
                    "carrier_mean_af": carrier_mean_af,
                    "mut_carrier_frac": mut_carrier_frac,
                    "odds_ratio": odds_ratio,
                    "p_value": p_value,
                }
            )

        if wt_subclone_hits:
            wt_sub_df = pd.DataFrame(wt_subclone_hits)

            valid_p = (
                wt_sub_df["p_value"]
                .fillna(1.0)
                .to_numpy()
            )

            _, adjusted_p, _, _ = multipletests(
                valid_p,
                alpha=significance_threshold,
                method=correction_method,
            )

            wt_sub_df["adj_p"] = adjusted_p
            wt_sub_df = wt_sub_df.sort_values("adj_p")

            significance_mask = (
                wt_sub_df["adj_p"] < significance_threshold
            )

            significant_wt_sub = wt_sub_df[
                significance_mask
            ].copy()

            rescue_mask = (
                ~significance_mask
                & (wt_sub_df["mut_carrier_frac"] == 0)
                & (
                    wt_sub_df["wt_carriers"]
                    >= min_wt_subclone_carrier_count
                )
                & (
                    wt_sub_df["carrier_mean_af"]
                    >= min_zero_mutant_carrier_af
                )
            )

            rescued = wt_sub_df[rescue_mask].copy()

            if not rescued.empty:
                rescued["rescued"] = True
                significant_wt_sub["rescued"] = False

                significant_wt_sub = (
                    pd.concat(
                        [significant_wt_sub, rescued],
                        ignore_index=True,
                    )
                    .sort_values("adj_p")
                )

            elif "rescued" not in significant_wt_sub.columns:
                significant_wt_sub["rescued"] = False

            vprint(f"  Candidates tested: {len(wt_sub_df)}")
            vprint(
                f"  Significant (adj_p < {significance_threshold}): "
                f"{int(significance_mask.sum())}"
            )

            if int(rescue_mask.sum()) > 0:
                vprint(
                    f"  Rescued (zero-in-mutant): "
                    f"{int(rescue_mask.sum())}"
                )

        else:
            significant_wt_sub = pd.DataFrame()
            vprint("  No WT-subclonal candidates found.")

    else:
        significant_wt_sub = pd.DataFrame()
        vprint(
            f"  ⚠️  Too few WT cells ({n_wt}) — "
            "skipping subclonal scan."
        )

    # ==================================================================
    # 3. COMBINE, GERMLINE FILTER, AND FORCE-RETAIN PROTECTED VARIANTS
    # ==================================================================

    vprint("\n" + "-" * 60)
    vprint("PART 3: COMBINE & FILTER")
    vprint("-" * 60)

    corr_cols = (
        significant_corr_results["column"].tolist()
        if not significant_corr_results.empty
        else []
    )

    wilcox_cols = (
        significant_wilcox_results["column"].tolist()
        if not significant_wilcox_results.empty
        else []
    )

    wt_sub_cols = (
        significant_wt_sub["column"].tolist()
        if not significant_wt_sub.empty
        else []
    )

    combined = sorted(
        set(
            corr_cols
            + wilcox_cols
            + wt_sub_cols
            + list(detected_whitelist)
            + list(forced_pathogenic)
        )
    )

    vprint(f"  From correlation: {len(corr_cols)}")
    vprint(f"  From Wilcoxon: {len(wilcox_cols)}")
    vprint(f"  From WT-subclonal scan: {len(wt_sub_cols)}")
    vprint(f"  Whitelist detected in df: {len(detected_whitelist)}")
    vprint(f"  Force-included pathogenic: {len(forced_pathogenic)}")
    vprint(
        "  Combined (union + detected whitelist + pathogenic): "
        f"{len(combined)}"
    )

    differential_variants = np.array(
        sorted(combined, key=extract_chr_num),
        dtype=str,
    )

    differential_variants = get_real_somatic_variants(
        differential_variants,
        filtered_germline_in_samples,
        Sample,
    )

    # Variants protected from all downstream exclusion steps.
    protected_variants = set()

    if whitelist_bypass_germline_filter:
        protected_variants |= detected_whitelist

    if pathogenic_bypass_germline_filter:
        protected_variants |= forced_pathogenic

    removed_by_germline_filter = (
        set(combined) - set(differential_variants)
    )

    protected_removed_by_germline = (
        removed_by_germline_filter & protected_variants
    )

    if protected_removed_by_germline:
        differential_variants = np.array(
            sorted(
                set(differential_variants)
                | protected_removed_by_germline,
                key=extract_chr_num,
            ),
            dtype=str,
        )

        whitelist_restored = (
            protected_removed_by_germline
            & detected_whitelist
        )

        pathogenic_restored = (
            protected_removed_by_germline
            & forced_pathogenic
        )

        if whitelist_restored:
            vprint(
                "  Re-added "
                f"{len(whitelist_restored)} whitelist variant(s) "
                "removed by germline/intronic filtering: "
                f"{sorted(whitelist_restored)}"
            )

        if pathogenic_restored:
            vprint(
                "  Re-added "
                f"{len(pathogenic_restored)} pathogenic variant(s) "
                "removed by germline/intronic filtering: "
                f"{sorted(pathogenic_restored)}"
            )

    vprint(
        "\n  After removing germline + intronic "
        f"(with enabled bypasses): {len(differential_variants)} variants"
    )

    # ==================================================================
    # 4. ANNOTATION-BASED INTRONIC / SYNONYMOUS FILTERING
    # ==================================================================

    if annotation_df is not None and len(differential_variants) > 0:
        anno = annotation_df.copy()

        if "Variant ID" in anno.columns:
            annotation_ids = (
                anno["Variant ID"]
                .astype(str)
                .str.strip()
            )
        else:
            annotation_ids = (
                anno.index
                .astype(str)
                .str.strip()
            )

        annotation_keys = annotation_ids.map(canonical_variant_key)
        protected_keys = {
            canonical_variant_key(variant)
            for variant in protected_variants
        }

        if "Function" in anno.columns:
            function_values = (
                anno["Function"]
                .astype(str)
                .str.lower()
                .fillna("")
            )

            intronic_keys = set(
                annotation_keys[
                    function_values.str.contains(
                        "intron",
                        na=False,
                    )
                ]
            )

            n_before = len(differential_variants)

            differential_variants = np.array(
                [
                    variant
                    for variant in differential_variants
                    if (
                        canonical_variant_key(variant)
                        not in intronic_keys
                    )
                    or (
                        canonical_variant_key(variant)
                        in protected_keys
                    )
                ],
                dtype=str,
            )

            n_removed = n_before - len(differential_variants)

            if n_removed > 0:
                vprint(
                    "  Annotation filter "
                    f"(intronic via Function): removed {n_removed}"
                )

        if "Coding impact" in anno.columns:
            coding_values = (
                anno["Coding impact"]
                .astype(str)
                .str.lower()
                .fillna("")
            )

            synonymous_keys = set(
                annotation_keys[
                    coding_values.str.contains(
                        "synonymous",
                        na=False,
                    )
                ]
            )

            n_before = len(differential_variants)

            differential_variants = np.array(
                [
                    variant
                    for variant in differential_variants
                    if (
                        canonical_variant_key(variant)
                        not in synonymous_keys
                    )
                    or (
                        canonical_variant_key(variant)
                        in protected_keys
                    )
                ],
                dtype=str,
            )

            n_removed = n_before - len(differential_variants)

            if n_removed > 0:
                vprint(
                    "  Annotation filter "
                    f"(synonymous via Coding impact): removed {n_removed}"
                )

    # Ensure returned variants remain exact dataframe column names.
    differential_variants = np.array(
        [
            variant
            for variant in differential_variants
            if variant in variant_cols
        ],
        dtype=str,
    )

    vprint(f"\n  ✓ Final somatic variants: {len(differential_variants)}")

    if verbose and len(differential_variants) > 0:
        for variant in differential_variants:
            tags = []

            if variant in detected_whitelist:
                tags.append("whitelist")

            if variant in forced_pathogenic:
                tags.append("pathogenic")

            tag_text = (
                f" [{' | '.join(tags)}]"
                if tags
                else ""
            )

            vprint(f"    • {variant}{tag_text}")

    vprint("\n" + "=" * 60)

    return (
        differential_variants,
        significant_corr_results,
        significant_wilcox_results,
    )

# ═══════════════════════════════════════════════════════════════════════════════
# Shared heatmap builder
# ═══════════════════════════════════════════════════════════════════════════════

def _variant_heatmap(
    df: pd.DataFrame,
    variant_list,
    Sample_subset,
    title_prefix: str,
    output_filename: str,
    wd: str,
    MBio_batch: str,
    Patient_ID: str,
    vmin_percentile: float = 2,
    vmax_percentile: float = 98,
    save_plot: bool = True,
    show_plot: bool = True,
):
    """Draw one clustered AF heatmap panel per label for ``variant_list``.

    Labels come from ``Sample_subset.dna.get_labels()`` and are aligned to
    ``df`` rows by position. Rows within each label are ordered by average-
    linkage Euclidean clustering (when >= 2 rows); colour limits are the
    ``vmin_percentile`` / ``vmax_percentile`` of all values after NaN → 0.
    Dashed white lines mark chromosome changes. When ``save_plot`` the figure
    is saved as PNG and PDF under
    ``{wd}/results/per_sample/{MBio_batch}/{Patient_ID}/dna_cnv/variants/somatic_variants``.
    Returns ``(fig, df_filtered)`` where ``df_filtered`` is the selected
    variant columns with a leading ``label`` column. Raises ``ValueError`` if
    no variant is a column of ``df``.
    """
    import os
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy.cluster.hierarchy import linkage, leaves_list

    dv = [c for c in list(variant_list) if c in df.columns]
    if not dv:
        raise ValueError("No matching variants found in df.")

    df_filtered = df.loc[:, dv].copy()
    labels = pd.Series(Sample_subset.dna.get_labels(), index=df.index)
    df_filtered.insert(0, "label", labels)

    X = (
        df_filtered.drop(columns="label")
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
    )
    vmin = np.nanpercentile(X.values, vmin_percentile)
    vmax = np.nanpercentile(X.values, vmax_percentile)

    chr_list = [extract_chr(c) for c in X.columns]
    unique_labels = df_filtered["label"].unique()

    n_panels = len(unique_labels)
    fig, axes = plt.subplots(
        n_panels, 1,
        figsize=(16, max(4, 3.5 * n_panels)),
        sharex=True,
    )
    if n_panels == 1:
        axes = [axes]

    for i, (ax, label) in enumerate(zip(axes, unique_labels)):
        X_lab = X.loc[df_filtered["label"].values == label]

        if X_lab.shape[0] >= 2:
            try:
                Z = linkage(X_lab.values, method="average", metric="euclidean")
                X_show = X_lab.iloc[leaves_list(Z), :]
            except Exception:
                X_show = X_lab
        else:
            X_show = X_lab

        sns.set(font_scale=0.9)
        sns.heatmap(
            X_show,
            cmap="mako",
            ax=ax,
            vmin=vmin,
            vmax=vmax,
            cbar=True,
            xticklabels=(i == n_panels - 1),
            yticklabels=False,
        )

        prev_chr = None
        ymax = X_show.shape[0]
        for j, chr_num in enumerate(chr_list):
            if chr_num != prev_chr:
                ax.axvline(x=j, color="white", linestyle="--", linewidth=2)
                ax.text(
                    j + 0.5, ymax + 0.2, chr_num,
                    ha="center", va="top", fontsize=10,
                    fontweight="bold", rotation=90,
                )
                prev_chr = chr_num

        ax.set_title(f"Clustered Heatmap of {title_prefix} Variants • {label}")
        ax.set_ylabel(
            str(label), fontsize=6, fontweight="bold", rotation=90, labelpad=8
        )

    axes[-1].set_xlabel("Variants", labelpad=20)
    plt.setp(axes[-1].get_xticklabels(), y=-0.20)
    plt.tight_layout()

    if save_plot:
        outdir = (
            f"{wd}/results/per_sample/{MBio_batch}/{Patient_ID}/dna_cnv/variants/somatic_variants"
        )
        os.makedirs(outdir, exist_ok=True)
        out = os.path.join(outdir, output_filename)
        fig.savefig(out, dpi=300, bbox_inches="tight")
        fig.savefig(out.replace('.png', '.pdf'), dpi=300, bbox_inches="tight")
        print(f"    ✓ Saved: {out}")

    if show_plot:
        plt.show()
    else:
        plt.close()

    return fig, df_filtered


def visualize_found_differential_variants(
    df, differential_variants, Sample_subset,
    wd, MBio_batch, Patient_ID,
    Sample=None,
    experiment: str = None,
    variant_metadata=None,
    resolution: str = "Final_annotation",
    umap_attribute: str = "umap",
    plot_heatmap: bool = True,
    plot_protein_umap: bool = True,
    plot_variant_grid: bool = True,
    ncols: int = 4,
    grid_kwargs: dict = None,
    show_plot: bool = True,
    **kwargs,
):
    """Plot heatmaps and a variant UMAP grid for differential variants.

    Draws a per-label clustered heatmap (optional) plus a single UMAP grid
    figure with the protein UMAP (coloured by ``resolution``) top-left and
    one panel per differential variant.

    Parameters
    ----------
    df : pd.DataFrame
        Cells × variants AF table (barcode-indexed for the UMAP grid).
    differential_variants : array-like
        Variant column names to plot.
    Sample_subset
        Sample whose DNA labels align row-wise with ``df`` (heatmap).
    wd, MBio_batch, Patient_ID : str
        Build the heatmap output directory (see :func:`_variant_heatmap`).
    Sample : optional
        Sample used for the UMAPs; defaults to ``Sample_subset``.
    experiment : str, optional
        Batch name for the grid output directory
        ``{wd}/results/per_sample/{experiment}/{Patient_ID}/dna_cnv/clonotypes``;
        defaults to ``MBio_batch``.
    variant_metadata : pd.DataFrame, optional
        Used to prefix grid panel titles with gene / protein names.
    resolution : str
        Protein row attribute used to colour the UMAP.
    umap_attribute : str
        Protein UMAP attribute name.
    plot_heatmap, plot_protein_umap, plot_variant_grid : bool
        Toggle each figure.
    ncols : int
        Grid columns.
    grid_kwargs : dict, optional
        Overrides for :func:`plot_somatic_variant_umap_grid` arguments.
    show_plot : bool
        Show the heatmap and the grid.
    **kwargs
        Forwarded to :func:`_variant_heatmap`.

    Returns
    -------
    dict
        ``heatmap_fig``, ``df_filtered``, ``protein_umap_fig``,
        ``variant_grid_fig`` (None for figures not drawn).

    Notes
    -----
    The standalone protein UMAP is built from
    ``Sample.protein.scatterplot`` but never shown or saved; failures there
    are printed and skipped. The grid is called with ``verbose=True``,
    panels of 200 × 200 px, carriers in ``#191919`` (size 6) over
    ``#D3D3D3`` background (size 3), output prefix ``Variant_UMAP``.
    """
    print("\n" + "=" * 60)
    print("VISUALIZING DIFFERENTIAL VARIANTS")
    print("=" * 60)

    sample_for_umap = Sample if Sample is not None else Sample_subset
    exp = experiment if experiment is not None else MBio_batch

    heatmap_fig = df_filtered = None
    if plot_heatmap:
        kwargs.pop("show_plot", None)
        heatmap_fig, df_filtered = _variant_heatmap(
            df, differential_variants, Sample_subset,
            title_prefix="Differential",
            output_filename="Differential_significant_variants.png",
            wd=wd, MBio_batch=MBio_batch, Patient_ID=Patient_ID,
            show_plot=show_plot, **kwargs,
        )

    protein_umap_fig = None
    if plot_protein_umap and sample_for_umap is not None:
        try:
            import plotly.graph_objects as go
            protein_umap_fig = go.Figure(
                sample_for_umap.protein.scatterplot(attribute=umap_attribute, colorby=resolution)
            )
        except Exception as e:
            print(f"  ⚠️  Standalone protein UMAP skipped: {e}")

    variant_grid_fig = None
    if plot_variant_grid and sample_for_umap is not None:
        outdir = f"{wd}/results/per_sample/{exp}/{Patient_ID}/dna_cnv/clonotypes"
        gkw = dict(
            variant_metadata=variant_metadata,
            resolution=resolution,
            umap_attribute=umap_attribute,
            ncols=ncols,
            variant_color="#191919",
            background_color="#D3D3D3",
            variant_size=6,
            background_size=3,
            width=200, height=200,
            outdir=outdir,
            output_prefix="Variant_UMAP",
            show_plot=show_plot,
            verbose=True,
        )
        # map width/height to panel_* names
        gkw["panel_width"] = gkw.pop("width")
        gkw["panel_height"] = gkw.pop("height")
        if grid_kwargs:
            gkw.update(grid_kwargs)
        variant_grid_fig = plot_somatic_variant_umap_grid(
            Sample=sample_for_umap, df=df, differential_variants=differential_variants, **gkw,
        )

    return {
        "heatmap_fig": heatmap_fig,
        "df_filtered": df_filtered,
        "protein_umap_fig": protein_umap_fig,
        "variant_grid_fig": variant_grid_fig,
    }


def visualize_found_informative_variants(
    df, combined_useful_variants_sorted, Sample_subset,
    wd, MBio_batch, Patient_ID, **kwargs,
):
    """Plot per-label clustered heatmaps for informative variants.

    Thin wrapper around :func:`_variant_heatmap` with title prefix
    ``Informative`` and file name ``Informative_significant_variants.png``;
    ``**kwargs`` (e.g. percentiles, ``save_plot``, ``show_plot``) are
    forwarded. Returns ``(fig, df_filtered)``.
    """
    print("\n" + "=" * 60)
    print("VISUALIZING INFORMATIVE VARIANTS")
    print("=" * 60)
    return _variant_heatmap(
        df, combined_useful_variants_sorted, Sample_subset,
        title_prefix="Informative",
        output_filename="Informative_significant_variants.png",
        wd=wd, MBio_batch=MBio_batch, Patient_ID=Patient_ID, **kwargs,
    )

def _protein_umap_frame(Sample, resolution="Final_annotation", umap_attribute="umap"):
    """Return protein UMAP coordinates and annotation as a barcode-indexed frame.

    Coordinates come from ``row_attrs[umap_attribute]`` or
    ``get_attribute(umap_attribute, constraint='row')`` (first 2-D result
    with >= 2 columns; ``ValueError`` otherwise). The annotation comes from
    ``row_attrs[resolution]`` or ``get_attribute``; if none has one value per
    barcode, every cell is annotated ``'all'``. Columns: ``umap_x``,
    ``umap_y``, ``annot`` (str); index name ``barcode``.
    """
    assay = Sample.protein
    try:
        barcodes = np.asarray(assay.barcodes()).astype(str).ravel()
    except Exception:
        barcodes = np.asarray(assay.row_attrs.get("barcode", [])).astype(str).ravel()

    coords = None
    for getter in (
        lambda: np.asarray(assay.row_attrs[umap_attribute]),
        lambda: np.asarray(assay.get_attribute(umap_attribute, constraint="row")),
    ):
        try:
            c = getter()
            if c.ndim == 2 and c.shape[1] >= 2:
                coords = c
                break
        except Exception:
            continue
    if coords is None:
        raise ValueError(f"Could not extract 2D '{umap_attribute}' from Sample.protein")

    annot = None
    for getter in (
        lambda: np.asarray(assay.row_attrs[resolution]).ravel(),
        lambda: np.asarray(assay.get_attribute(resolution, constraint="row")).to_numpy().ravel(),
        lambda: np.asarray(assay.get_attribute(resolution, constraint="row")).ravel(),
    ):
        try:
            a = getter()
            if len(a) == len(barcodes):
                annot = a
                break
        except Exception:
            continue
    if annot is None:
        annot = np.array(["all"] * len(barcodes))

    return pd.DataFrame(
        {"umap_x": coords[:, 0], "umap_y": coords[:, 1], "annot": annot.astype(str)},
        index=pd.Index(barcodes, name="barcode"),
    )


def plot_somatic_variant_umap_grid(
    Sample,
    df,
    differential_variants,
    variant_metadata=None,
    resolution="Final_annotation",
    umap_attribute="umap",
    ncols=4,
    carrier_af_threshold=10.0,
    color_by_af=False,
    variant_color="#191919",
    background_color="#D3D3D3",
    variant_size=6,
    background_size=3,
    panel_width=200,
    panel_height=200,
    title=None,
    outdir=None,
    output_prefix="Variant_UMAP",
    save_plot=True,
    show_plot=True,
    verbose=True,
):
    """Plot a protein UMAP grid with one carrier panel per variant.

    Panel (1, 1) is the protein UMAP coloured by ``resolution`` (plotly
    Dark24 + Light24 palette); each following panel shows cells with
    ``AF > carrier_af_threshold`` for one variant on top of the other cells.

    Parameters
    ----------
    Sample
        Sample whose ``protein`` assay provides UMAP and annotation
        (see :func:`_protein_umap_frame`).
    df : pd.DataFrame
        Barcode-indexed cells × variants AF table; reindexed to the protein
        barcodes.
    differential_variants : array-like
        Variants to plot; those not in ``df`` are skipped with a warning.
    variant_metadata : pd.DataFrame, optional
        If given, panel titles become ``"<Gene|Protein|Coding impact><br>id"``
        using the first non-null of ``Gene``, ``gene``, ``Protein``,
        ``protein``, ``Coding impact``.
    resolution, umap_attribute : str
        Protein annotation and UMAP attribute names.
    ncols : int
        Grid columns; rows = ``ceil((1 + n_variants) / ncols)``.
    carrier_af_threshold : float
        AF (percent) above which a cell is a carrier (default 10.0).
    color_by_af : bool
        Colour carriers by AF (Viridis, 0–100) instead of ``variant_color``.
    variant_color, background_color : str
        Carrier / background marker colours.
    variant_size, background_size : int
        Marker sizes (annotation panel uses ``background_size + 1``).
    panel_width, panel_height : int
        Pixel size per grid cell.
    title : str, optional
        Figure title.
    outdir : str, optional
        Where to write ``<output_prefix>_grid.html`` and (if kaleido is
        available) ``<output_prefix>_grid.png``.
    output_prefix : str
        File name prefix.
    save_plot, show_plot, verbose : bool
        Save (only when ``outdir`` is set), show, and print.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    import os
    import math
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.colors as pcolors

    umap_df = _protein_umap_frame(Sample, resolution=resolution, umap_attribute=umap_attribute)

    variants = [v for v in list(differential_variants) if v in df.columns]
    missing = [v for v in differential_variants if v not in df.columns]
    if verbose and missing:
        print(f"  ⚠️  {len(missing)} variant(s) not in df, skipped: {missing[:5]}{'...' if len(missing) > 5 else ''}")

    af = df.reindex(umap_df.index)  # align DNA AF to protein barcodes
    if af[variants].notna().to_numpy().sum() == 0 and verbose:
        print("  ⚠️  No AF values aligned to protein barcodes — check df is barcode-indexed.")

    def _vlabel(v):
        if variant_metadata is None:
            return v
        meta = variant_metadata
        try:
            if "Variant ID" in meta.columns:
                row = meta.loc[meta["Variant ID"].astype(str).str.strip() == v]
            else:
                row = meta.loc[[v]] if v in meta.index else meta.iloc[0:0]
            if row.empty:
                return v
            for col in ("Gene", "gene", "Protein", "protein", "Coding impact"):
                if col in row.columns and pd.notna(row.iloc[0][col]):
                    return f"{row.iloc[0][col]}<br>{v}"
        except Exception:
            pass
        return v

    n_panels = 1 + len(variants)
    nrows = math.ceil(n_panels / ncols)
    subplot_titles = [str(resolution)] + [_vlabel(v) for v in variants]

    fig = make_subplots(
        rows=nrows, cols=ncols,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.01, vertical_spacing=0.04,
    )

    # Panel (1,1): protein UMAP coloured by annotation
    cats = [c for c in pd.unique(umap_df["annot"]) if c is not None]
    palette = pcolors.qualitative.Dark24 + pcolors.qualitative.Light24
    for k, cat in enumerate(cats):
        sub = umap_df.loc[umap_df["annot"] == cat]
        fig.add_trace(
            go.Scattergl(
                x=sub["umap_x"], y=sub["umap_y"], mode="markers",
                marker=dict(size=background_size + 1, color=palette[k % len(palette)]),
                name=str(cat), legendgroup="annot", showlegend=True,
            ),
            row=1, col=1,
        )

    # Variant panels
    for i, v in enumerate(variants, start=1):
        r = i // ncols + 1
        c = i % ncols + 1
        a = pd.to_numeric(af[v], errors="coerce")
        is_carrier = a > carrier_af_threshold
        bg = umap_df.loc[~is_carrier.reindex(umap_df.index).fillna(False)]
        fg = umap_df.loc[is_carrier.reindex(umap_df.index).fillna(False)]
        fig.add_trace(
            go.Scattergl(
                x=bg["umap_x"], y=bg["umap_y"], mode="markers",
                marker=dict(size=background_size, color=background_color),
                showlegend=False, hoverinfo="skip",
            ),
            row=r, col=c,
        )
        if color_by_af:
            fmarker = dict(size=variant_size, color=a.reindex(fg.index), colorscale="Viridis",
                           cmin=0, cmax=100, showscale=False)
        else:
            fmarker = dict(size=variant_size, color=variant_color)
        fig.add_trace(
            go.Scattergl(
                x=fg["umap_x"], y=fg["umap_y"], mode="markers",
                marker=fmarker, name=v, showlegend=False,
                text=[f"{v}: {val:.0f}%" for val in a.reindex(fg.index)], hoverinfo="text",
            ),
            row=r, col=c,
        )

    fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False)
    fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False)
    for ann in fig.layout.annotations:
        ann.font.size = 9
    fig.update_layout(
        width=ncols * panel_width,
        height=nrows * panel_height,
        title=title or "",
        legend=dict(font=dict(size=8), itemsizing="constant"),
        margin=dict(l=10, r=10, t=40, b=10),
        plot_bgcolor="white",
    )

    if save_plot and outdir is not None:
        os.makedirs(outdir, exist_ok=True)
        html_out = os.path.join(outdir, f"{output_prefix}_grid.html")
        fig.write_html(html_out)
        if verbose:
            print(f"    ✓ Saved: {html_out}")
        try:
            fig.write_image(os.path.join(outdir, f"{output_prefix}_grid.png"), scale=2)
        except Exception as e:
            if verbose:
                print(f"    (PNG export skipped — needs kaleido: {e})")

    if show_plot:
        fig.show()
    return fig