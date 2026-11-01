"""Cell-level QC for Tapestri DNA assays: doublets, low-quality cells and AF heatmaps.

The functions here work on per-cell allele-frequency (AF) matrices, either a
DataFrame of ``[label] + variant columns`` or a Mosaic ``Sample`` / dna assay.

Main entry points
-----------------
Doublet detection
    :func:`flag_doublet_change_cells` (per-cell doublet metrics),
    :func:`piecewise_elbow_threshold` (two-line elbow on a sorted curve),
    :func:`define_doublet_cutoff_per_donor` (per-donor elbow threshold with a
    bootstrap CI) and :func:`apply_doublet_calls_with_corr_reassign`
    (correlation-based resolution of borderline cells).
Low-quality cells
    :func:`flag_amplicon_loss_cells` (deviation from donor consensus at
    germline-stable loci), :func:`flag_spurious_gain_cells` (AF spikes at
    expected-silent loci) and :func:`apply_low_quality_qc` (gated relabelling
    of cells to ``Low_quality`` on a Sample).
Plotting
    :func:`fast_af_heatmap` (rasterised, grouped/clustered matplotlib AF
    heatmap) and :func:`palette_from_plotly_scatter`.
"""

from __future__ import annotations

import os
import re
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap, TwoSlopeNorm


# Scale factor turning a MAD into a normal-consistent SD estimate.
_MAD_TO_SD = 1.4826
# Scale factor of the Iglewicz–Hoaglin modified z-score.
_ROBUST_Z_SCALE = 0.6745
# Small constant added to MADs / variances to avoid division by zero.
_EPS = 1e-12
# Labels that are never treated as donors by the amplicon-loss scorer.
_NON_DONOR_LABELS = frozenset({"Mixed", "Low_quality"})
# Median AF from which a variant counts as "mutant-like" for AF dilution.
_MUTANT_LIKE_MIN_AF = 0.30


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════


def _robust_z(x) -> np.ndarray:
    """Modified z-score ``0.6745 * (x - median) / (MAD + 1e-12)``, NaN-aware."""
    x = np.asarray(x, dtype=float)
    medx = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - medx)) + _EPS
    return _ROBUST_Z_SCALE * (x - medx) / mad


def _af_dataframe(assay, attribute="AF") -> pd.DataFrame:
    """Return an assay attribute as a DataFrame regardless of Mosaic's return type."""
    df = assay.get_attribute(attribute, constraint="row+col")
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)
    return df


def _corr_complete_obs(x, y, min_shared=40):
    """Pearson correlation over positions finite in both ``x`` and ``y``.

    Returns NaN when fewer than ``min_shared`` positions are shared or either
    vector is (near) constant on them.
    """
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < int(min_shared):
        return np.nan
    xv, yv = x[m], y[m]
    if np.std(xv) < 1e-12 or np.std(yv) < 1e-12:
        return np.nan
    return np.corrcoef(xv, yv)[0, 1]


# ═══════════════════════════════════════════════════════════════════════════════
# DOUBLET DETECTION
# ═══════════════════════════════════════════════════════════════════════════════


def flag_doublet_change_cells(
    df: pd.DataFrame,
    label_col: str = "label",
    af_change_cut: float = 0.30,
    min_variant_af: float = 0.10,
    max_variant_af: float = 0.90,
    z_k: float = 3.5,
    min_informative: int = 30,
    return_scores: bool = True,
    min_gain_frac: float = 0.01,
    min_loss_frac: float = 0.01,
    cross_donor_refs: dict | None = None,
    donor_name: str | None = None,
    disc_af_diff: float = 0.20,
    min_disc_variants: int = 5,
):
    """Score cells of one donor for doublet-like AF profiles.

    Each cell's AF profile is compared with the donor's per-variant median
    profile. Per active variant, a deviation ``>= +af_change_cut`` is a *gain*
    and ``<= -af_change_cut`` a *loss*; per cell the fractions of gained, lost
    and changed active variants are reported. Two flags are derived:

    * ``fail`` (returned first): a legacy score ``robust_z(frac_changed) +
      robust_z(af_dilution)`` above ``median + z_k * 1.4826 * MAD`` of the score.
    * ``qc["fail_bidir"]``: ``frac_changed_bidir = 2 * min(frac_gain,
      frac_loss)`` at or above its median, with ``frac_gain >= min_gain_frac``
      and ``frac_loss >= min_loss_frac``.

    Parameters
    ----------
    df : pd.DataFrame
        ``[label_col]`` + AF variant columns, one row per cell (all cells are
        treated as one donor; ``label_col`` is only dropped).
    label_col : str
        Name of the label column to drop before scoring.
    af_change_cut : float
        Per-variant absolute AF deviation from the median that counts as a
        gain/loss.
    min_variant_af, max_variant_af : float
        Median-AF range for a variant to be informative.
    z_k : float
        MAD multiplier for the legacy score threshold.
    min_informative : int
        If fewer variants are informative, all variants are used.
    return_scores : bool
        Return ``(fail, qc)`` instead of only ``fail``.
    min_gain_frac, min_loss_frac : float
        Minimum gain/loss fractions for the bidirectional flag.
    cross_donor_refs : dict, optional
        ``{donor_name: pd.Series}`` median AF profiles of all donors. Enables
        discriminating-variant mode when it has more than one entry.
    donor_name : str, optional
        Key of this donor in ``cross_donor_refs``.
    disc_af_diff : float
        Minimum absolute median-AF difference between this donor and any other
        donor for a variant to be discriminating.
    min_disc_variants : int
        Minimum number of discriminating variants (before and after
        intersecting with the informative set) to use discriminating mode.

    Returns
    -------
    fail : np.ndarray of bool
        Legacy score flag.
    qc : dict
        Only if ``return_scores``. Keys ``frac_changed``, ``frac_gain``,
        ``frac_loss``, ``frac_changed_bidir``, ``af_dilution``,
        ``change_score`` (arrays), ``threshold`` (float), ``fail_bidir``
        (bool array), ``n_disc_variants`` (number of active variants, in either
        mode) and ``used_discriminating`` (bool).

    Notes
    -----
    * Discriminating mode restricts the active variants to informative
      variants whose AF differs from at least one other donor's consensus by
      ``>= disc_af_diff``; otherwise all informative variants are active.
    * ``af_dilution`` is the mean positive drop below the median at
      "mutant-like" variants (median AF ``>= 0.30``, fixed, independent of
      ``af_change_cut``); it is 0 when there are none.
    * The thresholds assume a 0–1 AF scale; no rescaling of 0–100 AF is done.
    """
    af = df.drop(columns=[label_col]).astype(float)
    med = af.median(axis=0, skipna=True)

    informative = (med >= min_variant_af) & (med <= max_variant_af)
    mutant_like = med >= _MUTANT_LIKE_MIN_AF

    if informative.sum() < min_informative:
        informative[:] = True

    # ── Identify discriminating variants (cross-donor mode) ──
    use_disc = False
    disc_mask = informative.copy()

    if cross_donor_refs is not None and donor_name is not None and len(cross_donor_refs) > 1:
        this_ref = cross_donor_refs.get(donor_name)
        if this_ref is not None:
            disc_variants = set()
            for other_name, other_ref in cross_donor_refs.items():
                if other_name == donor_name:
                    continue
                common = this_ref.index.intersection(other_ref.index).intersection(af.columns)
                diff = (this_ref.reindex(common) - other_ref.reindex(common)).abs()
                disc_variants.update(diff[diff >= disc_af_diff].index)

            if len(disc_variants) >= min_disc_variants:
                candidate = pd.Series(False, index=af.columns)
                candidate[candidate.index.isin(disc_variants)] = True
                candidate = candidate & informative
                if candidate.sum() >= min_disc_variants:
                    disc_mask = candidate
                    use_disc = True

    active_mask = disc_mask if use_disc else informative

    # ── Signed change decomposition ──
    d = af.sub(med, axis=1).loc[:, active_mask]
    gain = d >= af_change_cut
    loss = d <= -af_change_cut
    changed = gain | loss

    n_active = max(1, int(active_mask.sum()))
    frac_gain = gain.sum(axis=1) / n_active
    frac_loss = loss.sum(axis=1) / n_active
    frac_changed = changed.sum(axis=1) / n_active

    frac_changed_bidir = 2.0 * np.minimum(frac_gain.values, frac_loss.values)

    # ── AF dilution at mutant-like loci ──
    if mutant_like.sum() > 0:
        drop = (med[mutant_like] - af.loc[:, mutant_like]).clip(lower=0)
        af_dilution = drop.mean(axis=1)
    else:
        af_dilution = pd.Series(0.0, index=af.index)

    score = _robust_z(frac_changed.values) + _robust_z(af_dilution.values)

    s_med = np.nanmedian(score)
    s_mad = np.nanmedian(np.abs(score - s_med)) + _EPS
    thr = s_med + z_k * _MAD_TO_SD * s_mad
    fail = score > thr

    finite_bidir = np.isfinite(frac_changed_bidir)
    if finite_bidir.any():
        bidir_base_thr = np.nanmedian(frac_changed_bidir[finite_bidir])
    else:
        bidir_base_thr = np.nan

    fail_bidir = (
        (frac_changed_bidir >= bidir_base_thr)
        & (frac_gain.values >= float(min_gain_frac))
        & (frac_loss.values >= float(min_loss_frac))
    ) if np.isfinite(bidir_base_thr) else np.zeros(len(af), dtype=bool)

    qc = {
        "frac_changed": np.asarray(frac_changed),
        "frac_gain": np.asarray(frac_gain),
        "frac_loss": np.asarray(frac_loss),
        "frac_changed_bidir": np.asarray(frac_changed_bidir),
        "af_dilution": np.asarray(af_dilution),
        "change_score": np.asarray(score),
        "threshold": float(thr),
        "fail_bidir": np.asarray(fail_bidir, dtype=bool),
        "n_disc_variants": int(active_mask.sum()),
        "used_discriminating": use_disc,
    }
    return (fail, qc) if return_scores else fail


def _two_line_fit(x, y, k):
    """Fit separate lines to ``[:k]`` and ``[k:]``; return ``(m1, m2, sse)``."""
    x1, y1 = x[:k], y[:k]
    x2, y2 = x[k:], y[k:]
    m1, b1 = np.polyfit(x1, y1, 1)
    m2, b2 = np.polyfit(x2, y2, 1)
    sse = np.sum((y1 - (m1 * x1 + b1)) ** 2) + np.sum((y2 - (m2 * x2 + b2)) ** 2)
    return m1, m2, sse


def piecewise_elbow_threshold(values, min_frac=0.02, max_frac=0.40, slope_ratio_min=3.0):
    """Find the elbow of a descending curve by fitting two straight lines.

    The finite values are sorted high→low and, for every split ``k`` in
    ``[max(5, int(min_frac*n)), min(n-5, int(max_frac*n))]``, one line is fitted
    to the first ``k`` points and one to the rest. Among splits where both
    slopes are negative and ``|m1| / (|m2| + 1e-12) >= slope_ratio_min``, the
    split with the lowest total squared error wins. If none qualifies, the
    lowest-SSE split over all candidates is used.

    Parameters
    ----------
    values : array-like
        Values to threshold (NaN/inf are dropped after sorting).
    min_frac, max_frac : float
        Bounds of the candidate split position as fractions of ``n``.
    slope_ratio_min : float
        Minimum steep/shallow slope ratio for a preferred split.

    Returns
    -------
    threshold : float
        ``y_sorted[k]`` at the chosen split, or NaN.
    k : int
        Split index into the descending curve, or -1.
    info : dict
        ``{"k", "thr", "sse", "m1", "m2", "ratio"}`` of the chosen split, or
        ``{}`` when fewer than 20 finite values remain or the candidate range
        is empty.
    """
    y = np.sort(np.asarray(values, dtype=float))[::-1]
    y = y[np.isfinite(y)]
    n = len(y)
    if n < 20:
        return np.nan, -1, {}

    x = np.arange(n, dtype=float)
    kmin = max(5, int(min_frac * n))
    kmax = min(n - 5, int(max_frac * n))
    if kmax <= kmin:
        return np.nan, -1, {}

    best = None
    for k in range(kmin, kmax + 1):
        m1, m2, sse = _two_line_fit(x, y, k)
        if m1 < 0 and m2 < 0:
            ratio = abs(m1) / (abs(m2) + 1e-12)
            if ratio >= slope_ratio_min:
                if (best is None) or (sse < best["sse"]):
                    best = {"k": k, "thr": y[k], "sse": sse, "m1": m1, "m2": m2, "ratio": ratio}

    if best is None:
        # No split with a clear slope change: fall back to the best plain fit.
        for k in range(kmin, kmax + 1):
            m1, m2, sse = _two_line_fit(x, y, k)
            ratio = abs(m1) / (abs(m2) + 1e-12)
            if (best is None) or (sse < best["sse"]):
                best = {"k": k, "thr": y[k], "sse": sse, "m1": m1, "m2": m2, "ratio": ratio}

    return best["thr"], best["k"], best


def _draw_elbow_panel(ax, fc, thr, k, ci_low, ci_high, ci, title, metric_name, legend_fontsize):
    """Draw the sorted-metric curve with threshold, CI band and elbow marker on ``ax``."""
    y = np.sort(fc)[::-1]
    x = np.arange(len(y))
    ax.plot(x, y, lw=1.2)
    ax.axhline(thr, color="red", ls="--", lw=1, label=f"thr={thr:.3f}")
    if np.isfinite(ci_low) and np.isfinite(ci_high):
        ax.axhspan(ci_low, ci_high, color="red", alpha=0.12,
                   label=f"{int(ci * 100)}% CI")
    if k >= 0:
        ax.axvline(k, color="red", ls=":", lw=1)
    ax.set_title(title)
    ax.set_xlabel("cells (sorted high→low)")
    ax.set_ylabel(metric_name)
    ax.legend(fontsize=legend_fontsize, loc="best")


def _tiny_heatmap_matrix(df_af, cell_ids, fc, heatmap_max_cells):
    """Rows of ``df_af`` sorted by descending ``fc``, evenly thinned, NaN-imputed and column-clustered.

    Returns ``(A, fc_sorted)``.
    """
    df_plot = df_af.reindex(cell_ids).copy()
    row_order = np.argsort(-fc)
    df_plot = df_plot.iloc[row_order, :]
    fc_sorted = np.asarray(fc)[row_order]

    if len(df_plot) > int(heatmap_max_cells):
        keep = np.linspace(0, len(df_plot) - 1, int(heatmap_max_cells)).astype(int)
        df_plot = df_plot.iloc[keep, :]
        fc_sorted = fc_sorted[keep]

    A = df_plot.to_numpy(dtype=float)
    col_med = np.nanmedian(A, axis=0)
    col_med[~np.isfinite(col_med)] = 0.0
    bad = ~np.isfinite(A)
    if bad.any():
        A[bad] = np.take(col_med, np.where(bad)[1])

    col_order = np.arange(A.shape[1])
    try:
        from scipy.cluster.hierarchy import linkage, leaves_list
        from scipy.spatial.distance import pdist
        Z = linkage(pdist(A.T, metric="euclidean"), method="average")
        col_order = leaves_list(Z)
    except Exception:
        pass
    return A[:, col_order], fc_sorted


def _plot_doublet_elbow(
    donor, fc, thr, k, ci_low, ci_high, ci, metric_name, disc_label, flagged, n,
    figsize, plot_tiny_heatmap, df_af, cell_ids, heatmap_max_cells, heatmap_cmap,
):
    """Show the per-donor elbow plot, optionally with metric strip and AF heatmap."""
    title = (
        f"{donor} | {metric_name}{disc_label} | "
        f"flagged {flagged}/{n} ({100 * flagged / max(n, 1):.1f}%)"
    )

    if not plot_tiny_heatmap:
        plt.figure(figsize=figsize)
        _draw_elbow_panel(plt.gca(), fc, thr, k, ci_low, ci_high, ci, title, metric_name,
                          legend_fontsize=8)
        plt.tight_layout()
        plt.show()
        return

    A, fc_sorted = _tiny_heatmap_matrix(df_af, cell_ids, fc, heatmap_max_cells)

    fig = plt.figure(figsize=(max(figsize[0], 7.0), figsize[1]))
    gs = fig.add_gridspec(
        1, 4, width_ratios=[1.2, 0.12, 0.18, 1.8], wspace=0.0,
    )
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 2])
    ax2 = fig.add_subplot(gs[0, 3])

    _draw_elbow_panel(ax0, fc, thr, k, ci_low, ci_high, ci, title, metric_name,
                      legend_fontsize=7)

    sns.heatmap(
        fc_sorted.reshape(-1, 1), ax=ax1, cmap="magma", cbar=False,
        xticklabels=[metric_name], yticklabels=False, linewidths=0,
    )
    ax1.tick_params(axis="x", labelrotation=90, labelsize=7)
    ax1.set_ylabel("")

    sns.heatmap(
        A, ax=ax2, cmap=heatmap_cmap, cbar=False,
        xticklabels=False, yticklabels=False, linewidths=0,
    )
    ax2.set_title(f"cells ordered by {metric_name}", fontsize=8)
    fig.subplots_adjust(
        left=0.06, right=0.98, top=0.90, bottom=0.18, wspace=0.0,
    )
    plt.show()


def define_doublet_cutoff_per_donor(
    sample_qc,
    label_key: str = "label",
    exclude_labels=("Mixed", "Low_quality"),
    min_frac: float = 0.01,
    max_frac: float = 0.55,
    slope_ratio_min: float = 3.0,
    n_boot: int = 200,
    ci: float = 0.95,
    near_ci_mult: float = 1.0,
    plot: bool = True,
    figsize=(5.0, 2.6),
    plot_tiny_heatmap: bool = False,
    heatmap_max_cells: int = 250,
    heatmap_cmap="viridis",
    use_bidirectional: bool = True,
    min_gain_frac: float = 0.01,
    min_loss_frac: float = 0.01,
    disc_af_diff: float = 0.20,
    min_disc_variants: int = 5,
):
    """Per-donor doublet calls from an elbow threshold with a bootstrap CI.

    For every donor label (sorted, excluding ``exclude_labels``) the AF matrix
    of that donor's cells is scored with :func:`flag_doublet_change_cells`
    (with its own default cut-offs) and the chosen metric is thresholded with
    :func:`piecewise_elbow_threshold`. Cells near the threshold are marked
    borderline using the width of a bootstrap CI of the threshold.

    Parameters
    ----------
    sample_qc : mosaic Sample
        Sample whose ``dna.row_attrs[label_key]`` holds donor labels.
    label_key : str
        Row attribute containing donor labels.
    exclude_labels : tuple
        Labels that are not processed as donors.
    min_frac, max_frac, slope_ratio_min : float
        Passed to :func:`piecewise_elbow_threshold`.
    n_boot : int
        Number of bootstrap resamples of the metric (RNG seed fixed at 42 per
        donor).
    ci : float
        Bootstrap confidence level.
    near_ci_mult : float
        Cells with ``|metric - thr| <= near_ci_mult * max(CI half-width, 1e-6)``
        are borderline.
    plot : bool
        Show one elbow plot per donor.
    figsize : tuple
        Elbow-plot size (width is raised to at least 7 with the tiny heatmap).
    plot_tiny_heatmap : bool
        Add a metric strip and an AF heatmap (cells sorted by metric, at most
        ``heatmap_max_cells`` rows, columns clustered) next to the elbow plot.
    heatmap_max_cells : int
        Row cap of the tiny heatmap.
    heatmap_cmap : str
        Colormap of the tiny heatmap.
    use_bidirectional : bool
        Threshold ``frac_changed_bidir`` (True) or ``frac_changed`` (False).
    min_gain_frac, min_loss_frac, disc_af_diff, min_disc_variants
        Passed to :func:`flag_doublet_change_cells`.

    Returns
    -------
    summary_df : pd.DataFrame
        One row per donor, sorted by donor: ``donor, n_cells, threshold,
        ci_low, ci_high, n_flagged, pct_flagged, n_borderline, metric,
        n_disc_variants, used_discriminating``.
    donor_calls : dict
        ``{donor: dict}`` with ``cell_ids``, ``frac_changed`` / ``metric_values``
        (the thresholded metric, finite cells only), ``metric_name``,
        ``threshold``, ``k``, ``ci_low``, ``ci_high``, counts, boolean
        ``fail`` / ``borderline``, ``class_label`` (``Doublet_HighConf``,
        ``Doublet_Borderline``, ``Singlet_Borderline``, ``Singlet_HighConf``),
        ``info`` and ``boot_thresholds``; plus ``frac_gain`` / ``frac_loss``
        for donors with at least 20 finite cells. Input for
        :func:`apply_doublet_calls_with_corr_reassign`.

    Notes
    -----
    * Cross-donor consensus profiles (median AF) are built for donors with at
      least 10 cells; discriminating mode is only enabled when at least two
      such donors exist. The number of discriminating variants over all donor
      pairs and, per donor, the number with consensus AF ``< 0.05`` ("silent")
      are printed.
    * Donors with fewer than 20 finite metric values get a NaN threshold and
      no flagged cells.
    * Fewer than 10 finite bootstrap thresholds collapse the CI to the point
      threshold.
    * A cell is flagged when ``metric >= threshold``.
    """
    donors = sorted([d for d in set(sample_qc.dna.row_attrs[label_key])
                     if d not in set(exclude_labels)])

    # ── Build cross-donor consensus profiles ──
    all_af = _af_dataframe(sample_qc.dna)
    all_labels = pd.Series(
        np.asarray(sample_qc.dna.row_attrs[label_key]).astype(str),
        index=all_af.index,
    )

    cross_donor_refs = {}
    for d in donors:
        mask = all_labels == d
        if mask.sum() >= 10:
            cross_donor_refs[d] = all_af.loc[mask].median(axis=0, skipna=True)

    n_donors_with_ref = len(cross_donor_refs)

    if n_donors_with_ref > 1:
        all_disc = set()
        for d1, d2 in combinations(cross_donor_refs.keys(), 2):
            common = cross_donor_refs[d1].index.intersection(cross_donor_refs[d2].index)
            diff = (cross_donor_refs[d1].reindex(common) - cross_donor_refs[d2].reindex(common)).abs()
            all_disc.update(diff[diff >= disc_af_diff].index)
        print(f"Cross-donor discriminating variants (AF diff ≥ {disc_af_diff}): "
              f"{len(all_disc)} / {len(all_af.columns)}")
        for d in donors:
            if d in cross_donor_refs:
                n_silent = sum(1 for v in all_disc
                               if v in cross_donor_refs[d].index
                               and cross_donor_refs[d][v] < 0.05)
                print(f"  {d}: {n_silent} silent discriminating variants")

    summary_rows = []
    donor_calls = {}

    for donor in donors:
        donor_mask = sample_qc.dna.row_attrs[label_key] == donor
        assay = sample_qc.dna[donor_mask, :]

        df_af = _af_dataframe(assay)

        labels = pd.Series(
            np.asarray(assay.get_labels(), dtype=object),
            index=df_af.index, name="label",
        )
        df = pd.concat([labels, df_af], axis=1)

        _, qc = flag_doublet_change_cells(
            df,
            label_col="label",
            return_scores=True,
            min_gain_frac=min_gain_frac,
            min_loss_frac=min_loss_frac,
            cross_donor_refs=cross_donor_refs if n_donors_with_ref > 1 else None,
            donor_name=donor,
            disc_af_diff=disc_af_diff,
            min_disc_variants=min_disc_variants,
        )

        if qc.get("used_discriminating", False):
            print(f"  {donor}: using {qc['n_disc_variants']} discriminating variants")
        else:
            print(f"  {donor}: using all informative variants "
                  f"(discriminating mode not applicable)")

        metric_name = "frac_changed_bidir" if use_bidirectional else "frac_changed"
        fc_all = np.asarray(qc[metric_name], dtype=float)
        valid = np.isfinite(fc_all)
        fc = fc_all[valid]
        cell_ids = np.asarray(df_af.index)[valid]

        if len(fc) < 20:
            summary_rows.append({
                "donor": donor,
                "n_cells": int(len(fc)),
                "threshold": np.nan,
                "ci_low": np.nan,
                "ci_high": np.nan,
                "n_flagged": 0,
                "pct_flagged": 0.0,
                "n_borderline": 0,
                "metric": metric_name,
                "n_disc_variants": qc.get("n_disc_variants", 0),
                "used_discriminating": qc.get("used_discriminating", False),
            })
            donor_calls[donor] = {
                "cell_ids": cell_ids,
                "frac_changed": fc,
                "metric_name": metric_name,
                "metric_values": fc,
                "threshold": np.nan,
                "k": -1,
                "ci_low": np.nan,
                "ci_high": np.nan,
                "n_flagged": 0,
                "pct_flagged": 0.0,
                "n_borderline": 0,
                "fail": np.zeros(len(fc), dtype=bool),
                "borderline": np.zeros(len(fc), dtype=bool),
                "class_label": np.array(["Singlet_HighConf"] * len(fc), dtype=object),
                "info": {},
                "boot_thresholds": np.array([], dtype=float),
            }
            continue

        thr, k, info = piecewise_elbow_threshold(
            fc, min_frac=min_frac, max_frac=max_frac, slope_ratio_min=slope_ratio_min
        )

        rng = np.random.default_rng(42)
        boot_thr = []
        n = len(fc)
        for _ in range(int(n_boot)):
            bs = fc[rng.integers(0, n, n)]
            t, _, _ = piecewise_elbow_threshold(
                bs, min_frac=min_frac, max_frac=max_frac,
                slope_ratio_min=slope_ratio_min,
            )
            if np.isfinite(t):
                boot_thr.append(float(t))

        if len(boot_thr) >= 10:
            alpha = (1 - ci) / 2
            ci_low, ci_high = np.quantile(boot_thr, [alpha, 1 - alpha])
        else:
            ci_low, ci_high = thr, thr

        fail = fc >= thr
        half_w = max((ci_high - ci_low) / 2, 1e-6)
        borderline = np.abs(fc - thr) <= near_ci_mult * half_w

        class_label = np.where(
            fail & ~borderline,
            "Doublet_HighConf",
            np.where(
                fail & borderline, "Doublet_Borderline",
                np.where(borderline, "Singlet_Borderline", "Singlet_HighConf"),
            ),
        )

        flagged = int(fail.sum())
        n_borderline = int(borderline.sum())

        summary_rows.append({
            "donor": donor,
            "n_cells": int(n),
            "threshold": float(thr),
            "ci_low": float(ci_low),
            "ci_high": float(ci_high),
            "n_flagged": flagged,
            "pct_flagged": 100.0 * flagged / max(n, 1),
            "n_borderline": n_borderline,
            "metric": metric_name,
            "n_disc_variants": qc.get("n_disc_variants", 0),
            "used_discriminating": qc.get("used_discriminating", False),
        })

        donor_calls[donor] = {
            "cell_ids": cell_ids,
            "frac_changed": fc,
            "metric_name": metric_name,
            "metric_values": fc,
            "frac_gain": np.asarray(qc["frac_gain"], dtype=float)[valid],
            "frac_loss": np.asarray(qc["frac_loss"], dtype=float)[valid],
            "threshold": float(thr),
            "k": int(k),
            "ci_low": float(ci_low),
            "ci_high": float(ci_high),
            "n_flagged": flagged,
            "pct_flagged": 100.0 * flagged / max(n, 1),
            "n_borderline": n_borderline,
            "fail": fail,
            "borderline": borderline,
            "class_label": class_label,
            "info": info,
            "boot_thresholds": np.asarray(boot_thr, dtype=float),
        }

        if plot:
            disc_label = (f" [{qc['n_disc_variants']} disc. vars]"
                          if qc.get("used_discriminating") else "")
            _plot_doublet_elbow(
                donor, fc, thr, k, ci_low, ci_high, ci, metric_name, disc_label, flagged, n,
                figsize, plot_tiny_heatmap, df_af, cell_ids, heatmap_max_cells, heatmap_cmap,
            )

    summary_df = pd.DataFrame(summary_rows).sort_values("donor").reset_index(drop=True)
    return summary_df, donor_calls


def apply_doublet_calls_with_corr_reassign(
    sample_qc,
    donor_calls: dict,
    donors=None,
    label_key: str = "label",
    band: float = 0.02,
    min_shared: int = 40,
    delta_corr: float = 0.00,
    mixed_label: str = "Mixed",
    write_back: bool = True,
):
    """Finalise doublet calls, resolving cells near the threshold by correlation.

    For each donor in ``donor_calls`` (from
    :func:`define_doublet_cutoff_per_donor`), cells are split by the stored
    metric into *low* (``< thr - band``), *high* (``>= thr + band``) and *mid*
    (the band around the threshold). High cells are doublets, low cells are
    not. Each mid cell's AF profile is correlated (complete observations,
    at least ``min_shared`` shared variants) with the donor's median profile
    and with the global median profile of ``mixed_label`` cells; it becomes a
    doublet when ``corr_mixed - corr_donor >= delta_corr``, when only the Mixed
    correlation is finite, and never when the Mixed correlation is NaN
    (including when no ``mixed_label`` cells exist).

    Parameters
    ----------
    sample_qc : mosaic Sample
        Sample whose ``dna.row_attrs[label_key]`` holds donor labels.
    donor_calls : dict
        Per-donor results with ``threshold``, ``cell_ids`` and
        ``metric_values`` / ``frac_changed`` (and optionally ``metric_name``).
    donors : list, optional
        Donors to process (default: sorted keys of ``donor_calls``); donors
        missing from ``donor_calls`` are skipped.
    label_key : str
        Row attribute with donor labels.
    band : float
        Half-width of the borderline band around the threshold.
    min_shared : int
        Minimum shared finite variants for a correlation.
    delta_corr : float
        Required correlation advantage of Mixed over donor.
    mixed_label : str
        Label of Mixed cells and of cells called doublets.
    write_back : bool
        Write labels into the per-donor subset assays (see Notes).

    Returns
    -------
    reassign_results : dict[str, pd.DataFrame]
        Per donor, indexed by cell id: the metric, ``threshold``, ``group``
        (``LowChange``, ``HighChange``, ``Borderline→Donor``,
        ``Borderline→Mixed``), ``fail_final``, ``corr_donor``, ``corr_mixed``.
    final_doublet_calls : dict[str, np.ndarray of bool]
        ``fail_final`` per donor, ordered as ``cell_ids``.
    sample_subsets : dict[str, assay]
        Per-donor dna sub-assays.

    Notes
    -----
    With ``write_back=True`` the subset assays get ``label_original`` (copy
    of their labels, only if absent), ``label_key`` (doublets set to
    ``mixed_label``) and ``doublet_group``. ``sample_qc`` itself is NOT
    modified; the caller must propagate the labels back (e.g. by barcode).
    The write-back indexes the subset's labels with ``fail_final`` by
    position, so it assumes ``cell_ids`` lists every subset row in order.
    """
    if donors is None:
        donors = sorted(list(donor_calls.keys()))

    all_af = _af_dataframe(sample_qc.dna)
    all_labels = pd.Series(
        np.asarray(sample_qc.dna.row_attrs[label_key]).astype(str),
        index=all_af.index,
    )
    has_mixed = (all_labels == mixed_label).any()
    mixed_ref_global = (
        all_af.loc[all_labels == mixed_label].median(axis=0, skipna=True)
        if has_mixed else None
    )

    reassign_results = {}
    final_doublet_calls = {}
    sample_subsets = {}

    for donor in donors:
        if donor not in donor_calls:
            continue

        dc = donor_calls[donor]
        thr = float(dc["threshold"])

        donor_mask = sample_qc.dna.row_attrs[label_key] == donor
        assay = sample_qc.dna[donor_mask, :]

        df_af = _af_dataframe(assay)

        cell_ids = pd.Index(dc["cell_ids"])
        df_af = df_af.reindex(cell_ids)

        metric_name = dc.get("metric_name", "frac_changed")
        metric_vals = np.asarray(
            dc.get("metric_values", dc.get("frac_changed", np.full(len(cell_ids), np.nan))),
            dtype=float,
        )
        metric = pd.Series(metric_vals, index=cell_ids)

        low_mask = metric < (thr - float(band))
        high_mask = metric >= (thr + float(band))
        mid_mask = ~(low_mask | high_mask)

        donor_ref = df_af.median(axis=0, skipna=True).reindex(df_af.columns)
        mixed_ref = (
            mixed_ref_global.reindex(df_af.columns)
            if mixed_ref_global is not None
            else pd.Series(np.nan, index=df_af.columns)
        )

        mid_idx = cell_ids[mid_mask.values]
        mid_to_mixed, sim_donor, sim_mixed = [], [], []

        for idx in mid_idx:
            x = df_af.loc[idx].values.astype(float)
            cd = _corr_complete_obs(x, donor_ref.values.astype(float), min_shared=min_shared)
            cm = (
                _corr_complete_obs(x, mixed_ref.values.astype(float), min_shared=min_shared)
                if mixed_ref_global is not None else np.nan
            )

            if not np.isfinite(cm):
                is_mixed = False
            elif not np.isfinite(cd):
                is_mixed = True
            else:
                is_mixed = (cm - cd) >= float(delta_corr)

            mid_to_mixed.append(is_mixed)
            sim_donor.append(cd)
            sim_mixed.append(cm)

        mid_to_mixed = pd.Series(mid_to_mixed, index=mid_idx, dtype=bool)

        fail_final = high_mask.copy()
        fail_final.loc[mid_idx] = mid_to_mixed.values

        group = pd.Series("LowChange", index=cell_ids, dtype=object)
        group.loc[high_mask] = "HighChange"
        group.loc[mid_idx[~mid_to_mixed.values]] = "Borderline→Donor"
        group.loc[mid_idx[mid_to_mixed.values]] = "Borderline→Mixed"

        if write_back:
            if "label_original" not in assay.row_attrs:
                assay.row_attrs["label_original"] = (
                    np.asarray(assay.get_labels(), dtype=object).copy()
                )
            out_labels = np.asarray(assay.row_attrs["label_original"], dtype=object).copy()
            out_labels[np.asarray(fail_final.values, dtype=bool)] = mixed_label
            assay.row_attrs[label_key] = out_labels
            assay.row_attrs["doublet_group"] = np.asarray(group.values, dtype=object)

        reassign_results[donor] = pd.DataFrame(
            {
                metric_name: metric,
                "threshold": thr,
                "group": group,
                "fail_final": fail_final,
                "corr_donor": pd.Series(sim_donor, index=mid_idx).reindex(cell_ids),
                "corr_mixed": pd.Series(sim_mixed, index=mid_idx).reindex(cell_ids),
            },
            index=cell_ids,
        )
        final_doublet_calls[donor] = np.asarray(fail_final.values, dtype=bool)
        sample_subsets[donor] = assay

    return reassign_results, final_doublet_calls, sample_subsets


# ═══════════════════════════════════════════════════════════════════════════════
# AMPLICON LOSS / LOW-QUALITY CELL DETECTION
# ═══════════════════════════════════════════════════════════════════════════════


def flag_amplicon_loss_cells(
    df: pd.DataFrame,
    label_col: str = "label",
    informative_range=(0.10, 0.90),
    stability_quantile: float = 0.35,
    min_stable_loci: int = 25,
    fallback_quantile: float = 0.50,
    min_shared_corr: int = 20,
    score_weights=(1.0, 1.0, 0.7),
    k_mad: float = 5.0,
    q: float = 0.995,
    min_cells: int = 20,
    return_scores: bool = True,
):
    """Flag noisy / low-quality cells by deviation from donor consensus at stable loci.

    For each donor label (``Mixed`` and ``Low_quality`` are skipped, as are
    donors with fewer than ``min_cells`` cells), the per-variant median AF is
    the consensus. *Stable* loci are informative variants (median within
    ``informative_range``) whose per-variant MAD is at or below the
    ``stability_quantile`` of those MADs; if fewer than ``min_stable_loci``
    result, all variants with MAD at or below the ``fallback_quantile`` are
    used. Per cell, on the stable loci:

    1. ``noise_mad``: median absolute deviation from the consensus,
    2. ``one_minus_corr``: 1 − Pearson correlation with the consensus
       (non-finite values are replaced by the donor median, or 1.0),
    3. ``nan_frac``: fraction of non-finite AF values.

    The composite score is ``w_dev*z(1) + w_corr*z(2) + w_nan*z(3)`` with
    robust z-scores; cells with ``score >= threshold`` are flagged.

    Parameters
    ----------
    df : pd.DataFrame
        ``[label_col]`` + AF variant columns, one row per cell. AF on a 0–100
        scale (99th percentile > 1.5) is divided by 100.
    label_col : str
        Donor label column.
    informative_range : tuple(float, float)
        Median-AF range (0–1 scale) for informative variants.
    stability_quantile, fallback_quantile : float
        Quantiles of per-variant MAD used to select stable loci.
    min_stable_loci : int
        Minimum stable loci before the fallback selection is used. Donors with
        fewer than 3 stable loci are skipped.
    min_shared_corr : int
        Minimum shared finite loci for the correlation.
    score_weights : tuple(float, float, float)
        Weights of deviation, correlation and NaN-fraction z-scores.
    k_mad, q : float
        Threshold ``max(median + k_mad*1.4826*MAD, quantile(q))`` of the
        donor's finite scores (when at least ``min_cells`` are finite;
        otherwise ``nanmedian + 5*nanstd``).
    min_cells : int
        Minimum donor size.
    return_scores : bool
        Also return the per-cell score table.

    Returns
    -------
    fail : np.ndarray of bool
        Flag per row of ``df`` (False for skipped donors).
    scores : pd.DataFrame
        Only if ``return_scores``. Indexed like ``df``; columns
        ``noise_mad``, ``one_minus_corr``, ``nan_frac``, ``composite_score``,
        ``threshold`` (NaN for skipped rows) and ``n_stable_loci`` (0).

    Raises
    ------
    AssertionError
        If ``label_col`` is not a column of ``df``.
    """
    assert label_col in df.columns, f"'{label_col}' not found in columns"

    labels = df[label_col].astype(str)
    af = df.drop(columns=[label_col]).apply(pd.to_numeric, errors="coerce")

    with np.errstate(all="ignore"):
        p99 = np.nanpercentile(af.values, 99)
    if np.isfinite(p99) and p99 > 1.5:
        af = af / 100.0

    donors = sorted(d for d in labels.unique() if d not in _NON_DONOR_LABELS)
    lo_af, hi_af = informative_range
    w_dev, w_corr, w_nan = score_weights

    n_total = len(af)
    fail = np.zeros(n_total, dtype=bool)
    noise_mad_out = np.full(n_total, np.nan)
    corr_out = np.full(n_total, np.nan)
    nan_frac_out = np.full(n_total, np.nan)
    score_out = np.full(n_total, np.nan)
    thr_out = np.full(n_total, np.nan)
    n_stable_out = np.zeros(n_total, dtype=int)

    def _corr(x, y):
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < min_shared_corr:
            return np.nan
        xv, yv = x[m], y[m]
        if np.std(xv) < _EPS or np.std(yv) < _EPS:
            return np.nan
        return float(np.corrcoef(xv, yv)[0, 1])

    for d in donors:
        mask = (labels == d).values
        idx = np.where(mask)[0]

        if len(idx) < min_cells:
            continue

        A = af.iloc[idx].values.astype(float)
        med = np.nanmedian(A, axis=0)
        var_mad = np.nanmedian(np.abs(A - med[None, :]), axis=0)

        informative = (med >= lo_af) & (med <= hi_af)
        if informative.sum() > 0:
            q_val = np.nanquantile(var_mad[informative], stability_quantile)
            stable = informative & (var_mad <= q_val)
        else:
            q_val = np.nanquantile(var_mad, stability_quantile)
            stable = var_mad <= q_val

        if stable.sum() < min_stable_loci:
            q_fb = np.nanquantile(var_mad, fallback_quantile)
            stable = var_mad <= q_fb

        stable_idx = np.where(stable)[0]
        n_stable = len(stable_idx)
        if n_stable < 3:
            continue

        X = A[:, stable_idx]
        ref = med[stable_idx]

        noise_mad = np.nanmedian(np.abs(X - ref[None, :]), axis=1)
        nan_frac = np.mean(~np.isfinite(X), axis=1)
        one_minus_corr = 1.0 - np.array([_corr(X[i], ref) for i in range(len(X))])

        finite = np.isfinite(one_minus_corr)
        if finite.any():
            one_minus_corr[~finite] = np.nanmedian(one_minus_corr[finite])
        else:
            one_minus_corr[:] = 1.0

        score = (
            w_dev * _robust_z(noise_mad)
            + w_corr * _robust_z(one_minus_corr)
            + w_nan * _robust_z(nan_frac)
        )

        s = score[np.isfinite(score)]
        if len(s) >= min_cells:
            s_med = np.nanmedian(s)
            s_mad = np.nanmedian(np.abs(s - s_med)) + _EPS
            thr = max(s_med + k_mad * _MAD_TO_SD * s_mad, np.nanquantile(s, q))
        else:
            thr = np.nanmedian(score) + 5.0 * np.nanstd(score)

        donor_fail = score >= thr

        fail[idx] = donor_fail
        noise_mad_out[idx] = noise_mad
        corr_out[idx] = one_minus_corr
        nan_frac_out[idx] = nan_frac
        score_out[idx] = score
        thr_out[idx] = thr
        n_stable_out[idx] = n_stable

    if return_scores:
        scores = pd.DataFrame(
            {
                "noise_mad": noise_mad_out,
                "one_minus_corr": corr_out,
                "nan_frac": nan_frac_out,
                "composite_score": score_out,
                "threshold": thr_out,
                "n_stable_loci": n_stable_out,
            },
            index=af.index,
        )
        return fail, scores
    return fail


# ═══════════════════════════════════════════════════════════════════════════════
# SPURIOUS GAIN
# ═══════════════════════════════════════════════════════════════════════════════


def flag_spurious_gain_cells(
    df: pd.DataFrame,
    label_col: str = "label",
    silent_max: float = 0.02,
    k_mad: float = 5.0,
    high_floor: float = 0.15,
    frac_spike_threshold: float = 0.30,
    min_silent: int = 10,
    return_scores: bool = True,
):
    """Flag cells with spurious AF spikes in variants expected to be silent.

    Per label group, a variant is *expected silent* for a cell when the
    group's median AF is ``< silent_max``. A spike is an AF above
    ``clip(median + k_mad*1.4826*MAD, high_floor, 1.0)`` (group median and MAD
    per variant) at an expected-silent variant. Cells whose fraction of spiking
    silent variants is ``>= frac_spike_threshold`` and that have at least
    ``min_silent`` silent variants are flagged.

    Parameters
    ----------
    df : pd.DataFrame
        ``[label_col]`` + AF variant columns. AF whose maximum exceeds 1.8 is
        divided by 100.
    label_col : str
        Group label column (all labels, including Mixed, are used as groups).
    silent_max, k_mad, high_floor, frac_spike_threshold, min_silent
        See above.
    return_scores : bool
        Also return the score table.

    Returns
    -------
    fail : np.ndarray of bool
    scores : pd.DataFrame
        Only if ``return_scores``; columns ``frac_spike_vs_silent``,
        ``silent_count``, ``flag_fail``, indexed like ``df``.

    Raises
    ------
    AssertionError
        If ``label_col`` is not a column of ``df``.
    """
    assert label_col in df.columns, f"'{label_col}' not found"
    labels = df[label_col].astype(str)
    af = df.drop(columns=[label_col]).apply(pd.to_numeric, errors="coerce")

    with np.errstate(all="ignore"):
        vmax = np.nanmax(af.values)
    if np.isfinite(vmax) and vmax > 1.8:
        af = af / 100.0

    med = af.groupby(labels, sort=False).median()
    med_mat = med.reindex(labels.values).to_numpy()
    A = af.to_numpy()

    expected_silent = med_mat < float(silent_max)

    abs_dev = np.abs(A - med_mat)
    mad_by_cluster = (
        pd.DataFrame(abs_dev, index=af.index, columns=af.columns)
        .groupby(labels, sort=False)
        .median()
        .reindex(labels.values)
        .to_numpy()
    )
    thresh = med_mat + float(k_mad) * 1.4826 * mad_by_cluster
    thresh = np.clip(thresh, float(high_floor), 1.0)

    spikes = (A > thresh) & expected_silent

    silent_counts = expected_silent.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        frac_spike = spikes.sum(axis=1) / np.maximum(silent_counts, 1)

    enough_silent = silent_counts >= int(min_silent)
    fail = (frac_spike >= float(frac_spike_threshold)) & enough_silent

    scores = pd.DataFrame(
        {
            "frac_spike_vs_silent": frac_spike,
            "silent_count": silent_counts,
            "flag_fail": fail,
        },
        index=af.index,
    )

    return (fail, scores) if return_scores else fail


# ═══════════════════════════════════════════════════════════════════════════════
# FAST AF HEATMAP (matplotlib, rasterized) — drop-in for sample.dna.heatmap()
# ═══════════════════════════════════════════════════════════════════════════════


def _prep_for_clustering(A, axis="column"):
    """
    Return a numeric matrix safe for correlation/euclidean clustering.

    NaNs are imputed with column medians for distance computation only. Constant
    rows/columns are handled by the caller so they can be appended to the end of
    an order rather than breaking correlation distances.
    """
    A = np.asarray(A, dtype=float).copy()
    if A.size == 0:
        return A

    col_med = np.nanmedian(A, axis=0)
    col_med[~np.isfinite(col_med)] = 0.0
    bad = ~np.isfinite(A)
    if bad.any():
        A[bad] = np.take(col_med, np.where(bad)[1])
    return A


def _cluster_columns(df, max_cells=2000, metric="correlation", method="average"):
    """
    Return a variant/column order from hierarchical clustering on a cell subsample.

    This clusters variants by their AF pattern across cells, analogous to the
    variant clustering used by interactive Mosaic/Tapestri heatmaps, but avoids
    rendering the full Plotly object.
    """
    try:
        from scipy.cluster.hierarchy import linkage, leaves_list
        from scipy.spatial.distance import pdist
    except Exception:
        return list(df.columns)

    if df.shape[1] < 3:
        return list(df.columns)

    A = df.to_numpy(dtype=float)
    if A.shape[0] > int(max_cells):
        keep = np.linspace(0, A.shape[0] - 1, int(max_cells)).astype(int)
        A = A[keep, :]

    A = _prep_for_clustering(A)

    # Drop zero-variance columns from the distance calculation. They are appended
    # to the tail in original order, otherwise correlation distances can be NaN.
    var = np.nanvar(A, axis=0)
    good = var > 1e-12
    order = np.arange(A.shape[1])

    if good.sum() >= 3:
        try:
            sub = A[:, good]
            Z = linkage(
                pdist(sub.T, metric=metric),
                method=method,
                optimal_ordering=True,
            )
            sub_order = leaves_list(Z)
            good_idx = np.where(good)[0][sub_order]
            tail = np.where(~good)[0]
            order = np.concatenate([good_idx, tail])
        except Exception:
            pass

    return [df.columns[i] for i in order]


def _cluster_row_index(df, max_cells=2000, metric="correlation", method="average"):
    """
    Cluster rows/cells of a DataFrame and return the clustered index.

    If the group is larger than ``max_cells``, evenly subsample first and then
    cluster only the rendered rows. This keeps PNG rendering and clustering costs
    bounded while preserving the visual structure of large groups.
    """
    try:
        from scipy.cluster.hierarchy import linkage, leaves_list
        from scipy.spatial.distance import pdist
    except Exception:
        return df.index

    if df.shape[0] < 3:
        return df.index

    df_work = df
    if df_work.shape[0] > int(max_cells):
        keep = np.linspace(0, df_work.shape[0] - 1, int(max_cells)).astype(int)
        df_work = df_work.iloc[keep, :]

    A = _prep_for_clustering(df_work.to_numpy(dtype=float))

    row_var = np.nanvar(A, axis=1)
    good = row_var > 1e-12
    order = np.arange(A.shape[0])

    if good.sum() >= 3:
        try:
            Z = linkage(
                pdist(A[good, :], metric=metric),
                method=method,
                optimal_ordering=True,
            )
            good_order = np.where(good)[0][leaves_list(Z)]
            tail = np.where(~good)[0]
            order = np.concatenate([good_order, tail])
        except Exception:
            pass

    return df_work.index[order]


def _cluster_group_order(df, labels, groups, metric="correlation", method="average"):
    """
    Cluster label groups by their median AF profile.

    This makes similar donors/clusters adjacent while keeping each label block
    contiguous, which is the key visual behaviour expected from Tapestri-style
    grouped heatmaps.
    """
    try:
        from scipy.cluster.hierarchy import linkage, leaves_list
        from scipy.spatial.distance import pdist
    except Exception:
        return groups

    med_profiles = []
    kept_groups = []

    for g in groups:
        idx = labels.index[labels == g]
        if len(idx) == 0:
            continue
        med_profiles.append(np.nanmedian(df.loc[idx].to_numpy(dtype=float), axis=0))
        kept_groups.append(g)

    if len(kept_groups) < 3:
        return groups

    A = _prep_for_clustering(np.vstack(med_profiles))
    row_var = np.nanvar(A, axis=1)
    good = row_var > 1e-12

    if good.sum() < 3:
        return groups

    try:
        Z = linkage(
            pdist(A[good, :], metric=metric),
            method=method,
            optimal_ordering=True,
        )
        ordered_good = [kept_groups[i] for i in np.where(good)[0][leaves_list(Z)]]
        remaining = [g for g in groups if g not in ordered_good]
        return ordered_good + remaining
    except Exception:
        return groups


def _natkey(s):
    """Natural sort key: pure-digit labels sort numerically, then the rest A->Z."""
    s = str(s)
    return (0, int(s)) if s.isdigit() else (1, s)


def _missionbio_colors():
    """Return the categorical colour cycle used by Mission Bio plotting.

    Mosaic/Tapestri scatterplots and heatmaps use
    ``missionbio.plotting.constants.COLORS`` for categorical labels. Importing it
    here makes the left label strip use the same colours as, for example::

        sample_qc.dna.scatterplot(attribute="umap", colorby="label")

    A Plotly-like fallback is provided so the helper still works outside a
    Mission Bio environment.
    """
    try:
        from missionbio.plotting.constants import COLORS
        return list(COLORS)
    except Exception:
        try:
            import plotly.express as px
            return list(px.colors.qualitative.Plotly)
        except Exception:
            return list(plt.get_cmap("tab20").colors)


def _group_palette(groups, all_labels=None, source="missionbio"):
    """Create a stable categorical palette for row-label strips.

    The important detail is that colours are assigned from the full sorted label
    universe, not from the plotted/clustered group order. Otherwise, clustering
    groups would change the colour of a label relative to Mosaic's scatterplot.
    """
    if all_labels is None:
        label_universe = sorted([str(g) for g in groups])
    else:
        label_universe = sorted([str(g) for g in pd.unique(pd.Series(all_labels).astype(str))])

    if source == "missionbio":
        base = _missionbio_colors()
    else:
        base = list(plt.get_cmap("tab20").colors)

    return {g: base[i % len(base)] for i, g in enumerate(label_universe)}


def palette_from_plotly_scatter(fig, label_values=None):
    """Extract a ``{label: colour}`` mapping from a Mosaic/Plotly scatterplot.

    This is optional, but useful if your local Mosaic version assigns categorical
    colours in a slightly different order than ``missionbio.plotting.constants``.
    Use it as::

        scat = sample_qc.dna.scatterplot(attribute="umap", colorby="label")
        pal = DNA_Utils.palette_from_plotly_scatter(scat)
        DNA_Utils.fast_af_heatmap(sample_qc, palette=pal)
    """
    palette = {}

    # Plotly categorical traces usually have one trace per label, with the label
    # in trace.name and the marker colour in trace.marker.color.
    try:
        for tr in fig.data:
            name = getattr(tr, "name", None)
            marker = getattr(tr, "marker", None)
            color = getattr(marker, "color", None) if marker is not None else None
            if name is None or color is None:
                continue
            name = str(name)
            if isinstance(color, (list, tuple, np.ndarray, pd.Series)):
                if len(color) == 0:
                    continue
                c0 = color[0]
                # Numeric per-point arrays are not categorical colours.
                if isinstance(c0, (int, float, np.integer, np.floating)):
                    continue
                color = c0
            palette[name] = color
    except Exception:
        pass

    if palette:
        return palette

    # Fallback: reproduce Mission Bio's categorical cycle over supplied labels.
    if label_values is not None:
        return _group_palette(pd.unique(pd.Series(label_values).astype(str)), all_labels=label_values)

    return {}


def _coerce_palette(palette, groups=None, labels=None, palette_source="missionbio"):
    """Return a plain ``{str(label): colour}`` palette dict.

    ``fast_af_heatmap`` accepts either a dict or the Plotly/FigureWidget object
    returned by Mosaic's ``scatterplot``. This helper normalizes those inputs so
    downstream code never calls ``.get`` on a Plotly figure.
    """
    if labels is not None:
        label_universe = pd.unique(pd.Series(labels).astype(str))
    elif groups is not None:
        label_universe = [str(g) for g in groups]
    else:
        label_universe = []

    if palette is None:
        return _group_palette(label_universe, all_labels=label_universe, source=palette_source)

    if isinstance(palette, dict):
        return {str(k): v for k, v in palette.items()}

    # Plotly go.Figure / FigureWidget / dict-like Plotly JSON.
    try:
        extracted = palette_from_plotly_scatter(palette, label_values=label_universe)
        if isinstance(extracted, dict) and len(extracted) > 0:
            return {str(k): v for k, v in extracted.items()}
    except Exception:
        pass

    # Last-resort fallback rather than failing during plotting.
    return _group_palette(label_universe, all_labels=label_universe, source=palette_source)


def _auto_scatterplot_palette(sample_or_assay, label_key="label", labels=None, attribute_order=("umap", "pca", "tsne")):
    """Try to recover the categorical colours used by Mosaic scatterplot.

    This lets ``fast_af_heatmap(sample_qc)`` use the same label colours as::

        sample_qc.dna.scatterplot(attribute="umap", colorby="label")

    without requiring the caller to pass a palette explicitly. If no compatible
    scatterplot is available, an empty dict is returned and the caller can fall
    back to the Mission Bio categorical cycle.
    """
    assay = getattr(sample_or_assay, "dna", sample_or_assay)
    scatterplot = getattr(assay, "scatterplot", None)
    if scatterplot is None:
        return {}

    if isinstance(attribute_order, str):
        attribute_order = (attribute_order,)

    for attr in attribute_order:
        try:
            fig = scatterplot(attribute=attr, colorby=label_key)
            pal = palette_from_plotly_scatter(fig, label_values=labels)
            if isinstance(pal, dict) and pal:
                return {str(k): v for k, v in pal.items()}
        except Exception:
            continue
    return {}


# Default AF colour ramp: white/grey for low AF through yellow/orange to dark red.
AF_HEATMAP_CMAP_COLORS = [
    "#ffffff",
    "#f5f3f4",
    "#d3d3d3",
    "#b1a7a6",
    "#e7d8c9",
    "#f5cb5c",
    "#f6aa1c",
    "#e85d04",
    "#e5383b",
    "#ba181b",
    "#a4161a",
    "#660708",
    "#370617",
]


def _default_af_heatmap_cmap():
    """Default AF heatmap colormap: white/grey low values to red/dark high values."""
    return ListedColormap(AF_HEATMAP_CMAP_COLORS, name="tapestri_af_white_to_red")


def _resolve_cmap(cmap, nan_color="#dcdcdc"):
    """Return a copy of a matplotlib colormap, accepting None, a name, or a Colormap."""
    if cmap is None:
        base_cmap = _default_af_heatmap_cmap()
    else:
        # ``plt.get_cmap`` accepts string names and matplotlib Colormap objects.
        base_cmap = plt.get_cmap(cmap)
    try:
        base_cmap = base_cmap.copy()
    except Exception:
        pass
    try:
        base_cmap.set_bad(nan_color)
    except Exception:
        pass
    return base_cmap


def _auto_heatmap_figsize(
    n_rows,
    n_cols,
    width=None,
    height=None,
    min_width=8.0,
    max_width=18.0,
    min_height=8.0,
    max_height=55.0,
    width_base=8.0,
    height_base=6.0,
    variants_per_inch=800.0,
    cells_per_inch=320.0,
):
    """Return data-aware figure dimensions in inches.

    Tuned so ~11.5k cells x ~1.6k variants resolves to roughly 10 x 42 inches,
    which is the practical size used for the large pre-cleanup AF plots.
    """
    n_rows = max(1, int(n_rows))
    n_cols = max(1, int(n_cols))

    if width is None:
        width = width_base + (n_cols / float(variants_per_inch))
        width = float(np.clip(width, min_width, max_width))
    if height is None:
        height = height_base + (n_rows / float(cells_per_inch))
        height = float(np.clip(height, min_height, max_height))

    return float(width), float(height)


def _af_colour_limits(vals, clip_percentiles, center_af):
    """Colour limits ``(vmin, vmax, vcenter)`` for an AF matrix.

    The scale is 0–100 when the 99th percentile exceeds 1.5, else 0–1. Limits
    are the ``clip_percentiles`` when finite and increasing, otherwise the full
    scale; with ``center_af`` they are widened to include the centre (50 or 0.5).
    """
    with np.errstate(all="ignore"):
        p99 = np.nanpercentile(vals, 99)
    is_percent = np.isfinite(p99) and p99 > 1.5

    # Full AF range is 0-1 or 0-100; clipping improves contrast in noisy panels.
    full_vmin, full_vmax = (0.0, 100.0) if is_percent else (0.0, 1.0)
    vmin, vmax = full_vmin, full_vmax
    if clip_percentiles is not None:
        with np.errstate(all="ignore"):
            lo, hi = np.nanpercentile(vals, list(clip_percentiles))
        if np.isfinite(lo) and np.isfinite(hi) and lo < hi:
            vmin, vmax = float(lo), float(hi)

    # Keep the diverging center meaningful even if percentile clipping would
    # otherwise exclude it.
    vcenter = 50.0 if is_percent else 0.5
    if center_af:
        vmin = min(vmin, vcenter)
        vmax = max(vmax, vcenter)
    return vmin, vmax, vcenter


def _grouped_row_order(
    df, labels, groups, per_group_cap, sort_vals, cluster_cells_within_group,
    row_cluster_max_cells, cluster_metric, cluster_method,
):
    """Rendered row labels and ``(start, end, group, n_available)`` blocks, group by group.

    Within a group rows are sorted by ``sort_vals`` (descending) and thinned,
    or clustered (thinned to ``min(per_group_cap, row_cluster_max_cells)``),
    or only thinned, to at most ``per_group_cap`` rows.
    """
    row_index, group_bounds, cursor = [], [], 0
    for g in groups:
        idx = labels.index[labels == g]
        if len(idx) == 0:
            continue

        n_avail = len(idx)
        if sort_vals is not None:
            idx = sort_vals.reindex(idx).sort_values(ascending=False).index
            if len(idx) > per_group_cap:
                idx = idx[np.linspace(0, len(idx) - 1, per_group_cap).astype(int)]
        elif cluster_cells_within_group:
            df_g = df.loc[idx]
            idx = _cluster_row_index(
                df_g,
                max_cells=min(per_group_cap, row_cluster_max_cells),
                metric=cluster_metric,
                method=cluster_method,
            )
        else:
            if len(idx) > per_group_cap:
                idx = idx[np.linspace(0, len(idx) - 1, per_group_cap).astype(int)]

        row_index.extend(list(idx))
        group_bounds.append((cursor, cursor + len(idx), g, n_avail))
        cursor += len(idx)
    return row_index, group_bounds


def _draw_label_strip(
    ax_lab, group_bounds, n_rows, palette_map, yticks, yticklabels,
    label_fontsize, separator_color, separator_lw,
):
    """Draw the categorical label strip (one colour per group block) with group ticks."""
    gidx = np.full(n_rows, -1, dtype=int)
    group_to_i = {g: i for i, (_s, _e, g, _n) in enumerate(group_bounds)}
    strip_colors = []
    for (_s, _e, g, _n) in group_bounds:
        strip_colors.append(palette_map.get(str(g), "#808080"))
    for (start, end, g, _n) in group_bounds:
        gidx[start:end] = group_to_i[g]

    strip_cmap = ListedColormap(strip_colors if strip_colors else ["#808080"])
    ax_lab.imshow(
        gidx.reshape(-1, 1),
        aspect="auto",
        cmap=strip_cmap,
        interpolation="nearest",
        origin="upper",
        vmin=0,
        vmax=max(0, len(group_bounds) - 1),
    )
    ax_lab.set_xticks([])
    ax_lab.set_yticks(yticks)
    ax_lab.set_yticklabels(yticklabels, fontsize=label_fontsize)
    ax_lab.tick_params(length=0)
    for (start, _end, _g, _n) in group_bounds[1:]:
        ax_lab.axhline(start - 0.5, color=separator_color, lw=separator_lw)


def fast_af_heatmap(
    sample_or_assay,
    label_key="label",
    attribute="AF",
    features=None,
    group_order=None,
    max_cells_per_group=2000,
    max_total_cells=20000,
    cluster_variants=True,
    cluster_groups=True,
    cluster_cells_within_group=True,
    cluster_max_cells=2000,
    row_cluster_max_cells=2000,
    cluster_metric="correlation",
    cluster_method="average",
    sort_within_group_by=None,
    cmap=None,
    nan_color="#dcdcdc",
    clip_percentiles=(1, 99),
    center_af=True,
    separator_color="#4a4a4a",
    separator_lw=0.8,
    show_label_strip=True,
    ytick_proportions=False,
    palette="auto",
    palette_source="missionbio",
    auto_palette_attribute=("umap", "pca", "tsne"),
    title=None,
    width=None,
    height=None,
    auto_size=True,
    min_width=8.0,
    max_width=18.0,
    min_height=8.0,
    max_height=55.0,
    variants_per_inch=800.0,
    cells_per_inch=320.0,
    dpi=150,
    save_path=None,
    show=True,
    return_order=False,
):
    """
    Fast, scalable AF heatmap styled after ``missionbio.plotting.heatmap.Heatmap``.

    Rows are grouped contiguously by ``label_key``. Variant columns, label groups,
    and cells within each label block can all be hierarchically clustered while
    preserving contiguous group blocks. The default colour scale is a custom
    white/grey-to-red AF palette so high AF is red/dark red.

    Parameters
    ----------
    sample_or_assay : mosaic Sample or dna assay
        A Sample (``.dna`` is used) or a dna assay directly.
    label_key : str
        Row attribute used to group rows, e.g. raw cluster IDs pre-demux or donor
        labels post-demux.
    attribute : str
        Assay attribute to display. Usually ``"AF"``.
    features : sequence, optional
        Explicit column/variant order. Takes precedence over ``cluster_variants``.
    group_order : sequence, optional
        Explicit top-to-bottom group order. If omitted, groups are naturally
        sorted and then optionally clustered by median AF profile.
    max_cells_per_group, max_total_cells : int
        Per-group and total caps on rendered cells. Each group is capped at
        ``min(max_cells_per_group, max(50, max_total_cells // n_groups))``
        rows (so the total can exceed ``max_total_cells`` when there are many
        groups). Large groups are evenly subsampled before within-group
        clustering.
    cluster_variants : bool
        Cluster variant columns by AF pattern across cells.
    cluster_groups : bool
        Cluster row-label groups by median AF profile, while keeping each group
        contiguous.
    cluster_cells_within_group : bool
        Cluster rendered cells inside each group. Ignored when
        ``sort_within_group_by`` is provided.
    cluster_max_cells, row_cluster_max_cells : int
        Maximum cells used for variant clustering and maximum rendered cells used
        per group for row clustering.
    cluster_metric, cluster_method : str
        Distance metric and linkage method passed to SciPy clustering.
    sort_within_group_by : array-like, optional
        Per-cell value aligned to assay rows. If provided, rows inside each group
        are sorted descending by this value instead of clustered.
    cmap : str, matplotlib Colormap, or None
        Matplotlib colormap. Default ``None`` uses a custom white/grey-to-red AF
        palette with high AF in red/dark red. Pass a colormap name/object to override.
    nan_color : str
        Colour for missing entries.
    separator_color : str
        Colour for horizontal lines separating label/cluster blocks.
    separator_lw : float
        Width of horizontal separator lines.
    clip_percentiles : tuple(float, float) or None
        Colour limits from percentiles, e.g. ``(1, 99)``. Use ``None`` for fixed
        full AF scale, i.e. 0-1 or 0-100.
    center_af : bool
        Use a diverging normalization centered at 0.5 or 50 when possible.
    show_label_strip : bool
        Draw the left categorical colour strip.
    ytick_proportions : bool
        Append each group's fraction of cells to its tick label.
    palette : "auto", dict, Plotly figure, or None
        Label-strip colours. Default ``"auto"`` first tries to extract the exact
        colours from ``sample_qc.dna.scatterplot(attribute="umap", colorby=label_key)``
        and then falls back to ``missionbio.plotting.constants.COLORS``. You may
        also pass a ``{group: colour}`` dict or an existing Plotly/FigureWidget.
    palette_source : {"missionbio", "matplotlib"}
        Fallback colour source when no scatterplot palette can be extracted.
    auto_palette_attribute : str or sequence[str]
        Embedding attribute(s) to try when ``palette="auto"``. Default tries
        ``"umap"``, then ``"pca"``, then ``"tsne"``.
    title : str, optional
        Figure title.
    width, height : float or None
        Figure size in inches. ``None`` uses data-aware auto sizing.
    auto_size : bool
        If True, choose missing width/height from the number of rendered cells
        and variants. The default is tuned so ~11.5k cells x ~1.6k variants is
        approximately 10 x 42 inches.
    min_width, max_width, min_height, max_height : float
        Bounds used by auto sizing.
    variants_per_inch, cells_per_inch : float
        Auto-size scaling factors. Lower values make the plot larger.
    dpi : int
        DPI for saved PNG.
    save_path : str, optional
        Path to save the figure.
    show : bool
        Whether to display the figure.
    return_order : bool
        If True, return ``(fig, rendered_cell_index, column_order)``.

    Returns
    -------
    matplotlib.figure.Figure
        The rendered figure, optionally with row/column order metadata.
    """
    assay = getattr(sample_or_assay, "dna", sample_or_assay)
    df = _af_dataframe(assay, attribute=attribute)

    labels = pd.Series(
        np.asarray(assay.row_attrs[label_key]).astype(object),
        index=df.index,
        name=label_key,
    ).astype(str)

    # ── Column order ──
    if features is not None:
        cols = [c for c in features if c in df.columns]
        df = df.loc[:, cols] if cols else df
    elif cluster_variants:
        df = df.loc[:, _cluster_columns(
            df,
            max_cells=cluster_max_cells,
            metric=cluster_metric,
            method=cluster_method,
        )]
    column_order = list(df.columns)

    # ── Colour scale ──
    vals = df.to_numpy(dtype=float)
    vmin, vmax, vcenter = _af_colour_limits(vals, clip_percentiles, center_af)

    # ── Group order ──
    if group_order is None:
        groups = sorted(pd.unique(labels), key=_natkey)
    else:
        present = set(labels)
        groups = [str(g) for g in group_order if str(g) in present]

    if cluster_groups:
        groups = _cluster_group_order(
            df,
            labels,
            groups,
            metric=cluster_metric,
            method=cluster_method,
        )

    n_groups = max(1, len(groups))
    per_group_cap = int(min(max_cells_per_group, max(50, max_total_cells // n_groups)))

    sort_vals = None
    if sort_within_group_by is not None:
        sort_vals = pd.Series(np.asarray(sort_within_group_by, dtype=float), index=df.index)

    # ── Row order with per-group downsampling/clustering ──
    row_index, group_bounds = _grouped_row_order(
        df, labels, groups, per_group_cap, sort_vals, cluster_cells_within_group,
        row_cluster_max_cells, cluster_metric, cluster_method,
    )

    rendered_index = pd.Index(row_index)
    pos = df.index.get_indexer(rendered_index)
    if np.any(pos < 0):
        keep = pos >= 0
        rendered_index = rendered_index[keep]
        pos = pos[keep]

    A = np.ma.masked_invalid(vals[pos]) if len(pos) else np.ma.masked_invalid(vals[:0])
    n_rows = A.shape[0]

    base_cmap = _resolve_cmap(cmap, nan_color=nan_color)

    norm = None
    if center_af and np.isfinite(vmin) and np.isfinite(vmax) and vmin < vcenter < vmax:
        norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)

    # ── Label palette ──
    if isinstance(palette, str) and palette.lower() in {"auto", "scatterplot", "mosaic"}:
        extracted_palette = _auto_scatterplot_palette(
            sample_or_assay,
            label_key=label_key,
            labels=labels,
            attribute_order=auto_palette_attribute,
        )
        palette_for_plot = extracted_palette if extracted_palette else None
    else:
        palette_for_plot = palette

    palette_map = _coerce_palette(
        palette_for_plot,
        groups=groups,
        labels=labels,
        palette_source=palette_source,
    )

    # ── Data-aware default figure size ──
    if auto_size:
        width, height = _auto_heatmap_figsize(
            n_rows=A.shape[0],
            n_cols=A.shape[1],
            width=width,
            height=height,
            min_width=min_width,
            max_width=max_width,
            min_height=min_height,
            max_height=max_height,
            variants_per_inch=variants_per_inch,
            cells_per_inch=cells_per_inch,
        )
    else:
        width = 12.0 if width is None else float(width)
        height = 16.0 if height is None else float(height)

    # ── Layout: thin label strip + main data + dedicated colorbar, shared y ──
    fig = plt.figure(figsize=(width, height))
    if show_label_strip:
        gs = fig.add_gridspec(1, 3, width_ratios=[1, 24, 0.5], wspace=0.02)
        ax_lab = fig.add_subplot(gs[0, 0])
        ax = fig.add_subplot(gs[0, 1], sharey=ax_lab)
        cax = fig.add_subplot(gs[0, 2])
    else:
        gs = fig.add_gridspec(1, 2, width_ratios=[24, 0.5], wspace=0.02)
        ax = fig.add_subplot(gs[0, 0])
        cax = fig.add_subplot(gs[0, 1])
        ax_lab = None

    if norm is not None:
        im = ax.imshow(
            A,
            aspect="auto",
            cmap=base_cmap,
            norm=norm,
            interpolation="nearest",
            origin="upper",
        )
    else:
        im = ax.imshow(
            A,
            aspect="auto",
            cmap=base_cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
            origin="upper",
        )

    im.set_rasterized(True)
    ax.set_xticks([])
    ax.set_xlabel(
        f"{A.shape[1]} variants"
        + (" (clustered)" if (cluster_variants and features is None) else "")
    )

    # Group separators on the data panel.
    for (start, _end, _g, _n) in group_bounds[1:]:
        ax.axhline(start - 0.5, color=separator_color, lw=separator_lw)

    # ── Centred group ticks (Mosaic-style) ──
    yticks, yticklabels = [], []
    for (start, end, g, n_avail) in group_bounds:
        yticks.append((start + end) / 2.0 - 0.5)
        if ytick_proportions:
            yticklabels.append(f"{g}  {n_avail / max(1, len(labels)):.1%}")
        else:
            yticklabels.append(str(g))

    label_fontsize = max(4, min(9, int(420 / max(1, n_groups))))

    if show_label_strip:
        _draw_label_strip(
            ax_lab, group_bounds, n_rows, palette_map, yticks, yticklabels,
            label_fontsize, separator_color, separator_lw,
        )
        plt.setp(ax.get_yticklabels(), visible=False)
        ax.tick_params(axis="y", length=0)
    else:
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels, fontsize=label_fontsize)

    total_rendered = sum(e - s for s, e, _g, _n in group_bounds)
    total_available = int(labels.isin(groups).sum())
    subtitle = f"{total_rendered:,} of {total_available:,} cells shown - {len(group_bounds)} groups"
    full_title = (title + "\n" if title else "") + subtitle

    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label(attribute)

    fig.subplots_adjust(left=0.07, right=0.93, top=0.95, bottom=0.04)
    fig.suptitle(full_title, fontsize=11, y=0.985)

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()

    if return_order:
        return fig, rendered_index, column_order
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# CLUSTER MERGING / DEMULTIPLEXING CLEAN-UP
# ═══════════════════════════════════════════════════════════════════════════════


def _donor_label_list(labels, mixed_label, lowq_label):
    """Sorted unique labels excluding ``mixed_label`` and ``lowq_label``."""
    return sorted(x for x in labels.unique() if x not in (mixed_label, lowq_label))


def _mad_or_quantile_threshold(values, k_mad, q):
    """``min(median + k_mad * 1.4826 * MAD, quantile(q))`` of ``values`` (NaN-aware)."""
    med = np.nanmedian(values)
    mad = np.nanmedian(np.abs(values - med)) + _EPS
    return min(med + k_mad * _MAD_TO_SD * mad, np.nanquantile(values, q))


def _binomial_se(p, n):
    """Binomial standard error ``sqrt(max(p * (1 - p) / n, 1e-12))``."""
    return np.sqrt(np.maximum(p * (1.0 - p) / n, _EPS))


def _low_af_fraction(af_vals, informative, low_af_cut):
    """Per-cell fraction of informative loci that are missing or below ``low_af_cut``.

    Returns ``(low_af_frac, n_info)`` where ``n_info`` counts finite informative loci.
    """
    low_af_frac = np.mean(
        (~np.isfinite(af_vals[:, informative])) | (af_vals[:, informative] < low_af_cut),
        axis=1,
    )
    n_info = np.sum(np.isfinite(af_vals[:, informative]), axis=1)
    return low_af_frac, n_info


def _extract_chrom(name):
    """Chromosome token (``"1"``–``"99"``, ``X``, ``Y``, ``M``) from a variant name, or None.

    Matches ``chr<token>`` anywhere first, then a leading ``<token>`` followed
    by ``:``, ``_`` or ``-``.
    """
    s = str(name)
    m = re.search(r"chr\s*([0-9]{1,2}|[XYM])", s, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()
    m = re.match(r"\s*([0-9]{1,2}|[XYM])[:_\-]", s, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()
    return None


def _localized_loh_mask(
    af, af_vals, labels, mixed_label, lowq_label, min_donor_cells, is_percent_scale,
    loh_shift, loh_concentration, loh_max_chromosomes, loh_min_loci_per_chr,
    loh_min_total_loci,
):
    """Boolean mask of cells whose AF shifts look like localized (few-chromosome) LOH.

    Per donor with at least ``min_donor_cells`` cells, informative loci have a
    donor-median AF in [0.20, 0.80] (or [20, 80]) and a parsable chromosome.
    Per cell and chromosome (with at least ``loh_min_loci_per_chr`` such loci),
    hits are loci with ``|AF - donor median| >= loh_shift``. A cell is
    LOH-like when it has ``>= loh_min_total_loci`` hits, at least
    ``loh_concentration`` of them on one chromosome, and at most
    ``loh_max_chromosomes`` chromosomes with ``>= loh_min_loci_per_chr`` hits.
    """
    loh_protected_mask = np.zeros(len(labels), dtype=bool)
    col_names = pd.Index(af.columns).astype(str)

    chrom = np.array([_extract_chrom(c) for c in col_names], dtype=object)
    valid_chrom = pd.notna(chrom)

    if not valid_chrom.any():
        return loh_protected_mask

    uniq_chrom = [c for c in pd.unique(chrom[valid_chrom]) if c is not None]

    for d in _donor_label_list(labels, mixed_label, lowq_label):
        m = (labels == d).values
        if m.sum() < min_donor_cells:
            continue

        donor_med = np.nanmedian(af_vals[m], axis=0)
        donor_mid = (donor_med >= (0.20 if not is_percent_scale else 20.0)) & (
            donor_med <= (0.80 if not is_percent_scale else 80.0)
        )
        informative_loci = donor_mid & valid_chrom
        if informative_loci.sum() < int(loh_min_total_loci):
            continue

        cell_idx = np.where(m)[0]
        if len(cell_idx) == 0:
            continue

        hit_counts = np.zeros((len(cell_idx), len(uniq_chrom)), dtype=float)
        for ci, ch in enumerate(uniq_chrom):
            use = informative_loci & (chrom == ch)
            if use.sum() < int(loh_min_loci_per_chr):
                continue
            D = np.abs(af_vals[cell_idx][:, use] - donor_med[use][None, :])
            hit_counts[:, ci] = np.sum(np.isfinite(D) & (D >= float(loh_shift)), axis=1)

        total_hits = hit_counts.sum(axis=1)
        max_hits = hit_counts.max(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            concentration = max_hits / np.maximum(total_hits, 1.0)
        n_affected = np.sum(hit_counts >= float(loh_min_loci_per_chr), axis=1)

        donor_loh_like = (
            (total_hits >= float(loh_min_total_loci))
            & (concentration >= float(loh_concentration))
            & (n_affected <= int(loh_max_chromosomes))
        )
        loh_protected_mask[cell_idx] = donor_loh_like
    return loh_protected_mask


def apply_low_quality_qc(
    sample_qc,
    DNA_Utils,
    sample_subsets=None,
    IS_MULTIPLEXED: bool = False,
    label_key: str = "label",
    mixed_label: str = "Mixed",
    lowq_label: str = "Low_quality",
    k_mad: float = 3.75,
    q: float = 0.975,
    min_donor_cells: int = 20,
    abs_global_dropout=0.50,
    abs_low_af_info=0.50,
    depth_adaptive: bool = True,
    abs_floor_quantile: float = 0.98,
    use_global_dropout_gate: bool = True,
    use_low_af_gate: bool = False,
    use_combined_gate: bool = True,
    combined_quantile: float = 0.95,
    min_effect_size=2,
    protect_localized_loh: bool = False,
    loh_shift: float = 0.35,
    loh_concentration: float = 0.75,
    loh_max_chromosomes: int = 2,
    loh_min_loci_per_chr: int = 4,
    loh_min_total_loci: int = 10,
    show_global_heatmap: bool = True,
    show_subset_heatmaps: bool = True,
    heatmap_save_dir=None,
):
    """Relabel low-quality cells to ``lowq_label`` with a sequence of gates.

    The gates are OR-ed (a cell flagged by an earlier gate stays flagged):

    1. **Amplicon loss**: ``DNA_Utils.flag_amplicon_loss_cells(df,
       label_col=label_key)`` with that function's own defaults.
    2. **Global dropout** (``use_global_dropout_gate``): per donor, a cell's
       dropout (fraction of loci missing or with AF ``<= 0.01`` / ``1.0`` %)
       above ``min(median + k_mad*1.4826*MAD, quantile(q))`` with a binomial
       effect size ``>= min_effect_size``, or above an absolute floor.
       Donors smaller than ``min_donor_cells`` use the threshold of all
       non-Mixed/non-Low_quality cells pooled.
    3. **Low AF at informative loci** (``use_low_af_gate``): the same rule on
       the fraction of donor-informative loci (donor median AF ``>= 0.10`` /
       10) that are missing or below 0.05 / 5.
    4. **Combined** (``use_combined_gate``): both metrics at or above their
       donor ``combined_quantile`` with effect size ``>= min_effect_size``.

    Optionally (``protect_localized_loh``) cells with localized-LOH-like AF
    shifts are un-flagged at the end. Labels other than ``mixed_label`` are
    then set to ``lowq_label`` for flagged cells, and global / per-subset AF
    heatmaps are shown with :func:`fast_af_heatmap`.

    Parameters
    ----------
    sample_qc : mosaic Sample
        Sample with ``dna.row_attrs[label_key]`` (modified in place).
    DNA_Utils : module
        Module providing ``flag_amplicon_loss_cells`` (normally the
        ``DNA_Utils`` package).
    sample_subsets : dict, optional
        ``{name: dna assay}``; with ``IS_MULTIPLEXED`` and
        ``show_subset_heatmaps`` their labels are updated from ``sample_qc``
        by row id (missing ids get the subset name) and a heatmap is drawn per
        subset.
    IS_MULTIPLEXED : bool
        Whether to process ``sample_subsets``.
    label_key, mixed_label, lowq_label : str
        Label attribute, Mixed label (never relabelled) and low-quality label.
    k_mad, q : float
        MAD multiplier and quantile of gates 2 and 3 (not used by gate 1).
    min_donor_cells : int
        Minimum donor size for per-donor thresholds (gates 2–4, LOH).
    abs_global_dropout, abs_low_af_info : float or None
        Absolute floors of gates 2 and 3; None disables that absolute rule.
    depth_adaptive : bool
        Raise each absolute floor to ``max(floor, donor quantile
        abs_floor_quantile)``.
    abs_floor_quantile : float
        Donor quantile used by ``depth_adaptive``.
    use_global_dropout_gate, use_low_af_gate, use_combined_gate : bool
        Enable gates 2, 3 and 4.
    combined_quantile : float
        Donor quantile of both metrics in gate 4.
    min_effect_size : float
        Minimum ``(value - threshold) / binomial SE``.
    protect_localized_loh : bool
        Rescue localized-LOH-like cells from relabelling.
    loh_shift, loh_concentration, loh_max_chromosomes, loh_min_loci_per_chr, loh_min_total_loci
        Localized-LOH rule parameters (see Notes).
    show_global_heatmap, show_subset_heatmaps : bool
        Draw the heatmaps.
    heatmap_save_dir : str, optional
        Directory for ``AF_lowq_global.png`` and ``AF_lowq_<subset>.png``.

    Returns
    -------
    summary_df : pd.DataFrame
        Per label except ``mixed_label`` (so including ``lowq_label`` if
        present): ``donor, n_cells, n_stable_loci, score_thr, n_lowq,
        pct_lowq`` (``score_thr`` / ``n_stable_loci`` from gate 1).
    fail : np.ndarray of bool
        Final low-quality flag per cell.
    scores : pd.DataFrame
        Gate-1 score table from ``flag_amplicon_loss_cells``.
    sample_subsets : dict or None
        The input subsets (labels possibly updated).

    Notes
    -----
    * Row attributes written to ``sample_qc.dna``: ``label_key``,
      ``lowq_germline_fail``, ``lowq_germline_score``, ``lowq_stable_loci_n``,
      ``loh_protected`` and ``chr7_loh_protected`` (same mask).
    * Prints per-donor counts of cells added by each gate, the LOH rescue
      count, the summary table and the final label counts.
    * The AF scale (0–1 vs 0–100) is detected from the 99th percentile
      (> 1.5 means percent) and the AF cut-offs above are switched
      accordingly, but ``loh_shift`` is compared with AF differences without
      rescaling (0.35 AF-percent on a 0–100 scale).
    * Localized LOH: per donor, informative loci have median AF in [0.20,
      0.80] (or [20, 80]); a cell is protected when it has ``>=
      loh_min_total_loci`` loci shifted by ``>= loh_shift``, at least
      ``loh_concentration`` of them on one chromosome and at most
      ``loh_max_chromosomes`` chromosomes with ``>= loh_min_loci_per_chr``
      shifted loci.
    """
    af = _af_dataframe(sample_qc.dna)

    labels = pd.Series(
        np.asarray(sample_qc.dna.row_attrs[label_key]).astype(str),
        index=af.index,
        name=label_key,
    )
    df = pd.concat([labels, af], axis=1)

    # Gate 1: base amplicon-loss from DNA_Utils (its own defaults, not k_mad/q).
    fail, scores = DNA_Utils.flag_amplicon_loss_cells(df, label_col=label_key, return_scores=True)
    fail = np.asarray(fail, dtype=bool)

    af_vals = af.values.astype(float)
    with np.errstate(all="ignore"):
        p99 = np.nanpercentile(af_vals, 99)
    is_percent_scale = np.isfinite(p99) and p99 > 1.5
    zero_like_cut = 1.0 if is_percent_scale else 0.01
    informative_min = 10.0 if is_percent_scale else 0.10
    low_af_cut = 5.0 if is_percent_scale else 0.05

    global_dropout = np.mean(~np.isfinite(af_vals) | (af_vals <= zero_like_cut), axis=1)
    n_total_loci = max(1, af_vals.shape[1])

    # Pooled threshold, used for donors too small for their own.
    valid_pool = ~labels.isin([mixed_label, lowq_label])
    pool_vals = global_dropout[valid_pool.values]
    pool_thr = _mad_or_quantile_threshold(pool_vals, k_mad, q)

    # Localized-LOH mask (protected from Low_quality relabeling)
    loh_protected_mask = np.zeros(len(labels), dtype=bool)
    if protect_localized_loh:
        loh_protected_mask = _localized_loh_mask(
            af, af_vals, labels, mixed_label, lowq_label, min_donor_cells, is_percent_scale,
            loh_shift, loh_concentration, loh_max_chromosomes, loh_min_loci_per_chr,
            loh_min_total_loci,
        )

    # Gate 2: global dropout per donor
    if use_global_dropout_gate:
        for d in _donor_label_list(labels, mixed_label, lowq_label):
            m = (labels == d).values
            if m.sum() == 0:
                continue

            donor_dropout = global_dropout[m]
            if m.sum() >= min_donor_cells:
                dropout_thr = _mad_or_quantile_threshold(donor_dropout, k_mad, q)
            else:
                dropout_thr = pool_thr

            se_drop = _binomial_se(dropout_thr, n_total_loci)
            effect_drop = (global_dropout - dropout_thr) / se_drop

            # Depth-adaptive absolute floor: float it above the donor's bulk so a
            # globally high-dropout (low-depth) donor isn't wiped out wholesale.
            if abs_global_dropout is None:
                abs_hit = np.zeros_like(global_dropout, dtype=bool)
            else:
                abs_drop_eff = abs_global_dropout
                if depth_adaptive:
                    abs_drop_eff = max(
                        abs_global_dropout,
                        float(np.nanquantile(donor_dropout, abs_floor_quantile)),
                    )
                abs_hit = global_dropout >= abs_drop_eff

            extra = (
                (((global_dropout > dropout_thr) & (effect_drop >= float(min_effect_size)))
                 | abs_hit)
                & m & ~fail
            )
            if extra.any():
                fail |= extra
                print(f"  {d}: +{int(extra.sum())} by global dropout (thr={dropout_thr:.3f})")

    # Gate 3: low AF at donor-informative loci
    if use_low_af_gate:
        for d in _donor_label_list(labels, mixed_label, lowq_label):
            m = (labels == d).values
            if m.sum() < min_donor_cells:
                continue

            donor_med = np.nanmedian(af_vals[m], axis=0)
            informative = donor_med >= informative_min
            if informative.sum() < 10:
                continue

            low_af_frac, n_info = _low_af_fraction(af_vals, informative, low_af_cut)
            donor_low = low_af_frac[m]
            low_thr = _mad_or_quantile_threshold(donor_low, k_mad, q)

            se_low = _binomial_se(low_thr, np.maximum(n_info, 1))
            effect_low = (low_af_frac - low_thr) / se_low

            if abs_low_af_info is None:
                abs_low_hit = np.zeros_like(low_af_frac, dtype=bool)
            else:
                abs_low_eff = abs_low_af_info
                if depth_adaptive:
                    abs_low_eff = max(
                        abs_low_af_info,
                        float(np.nanquantile(donor_low, abs_floor_quantile)),
                    )
                abs_low_hit = low_af_frac >= abs_low_eff

            extra = (
                (((low_af_frac > low_thr) & (effect_low >= float(min_effect_size)))
                 | abs_low_hit)
                & m & ~fail
            )
            if extra.any():
                fail |= extra
                print(f"  {d}: +{int(extra.sum())} by low-AF informative (thr={low_thr:.3f})")

    # Gate 4: moderate risk on both metrics
    if use_combined_gate:
        for d in _donor_label_list(labels, mixed_label, lowq_label):
            m = (labels == d).values
            if m.sum() < min_donor_cells:
                continue

            donor_dropout = global_dropout[m]
            d_q = np.nanquantile(donor_dropout, combined_quantile)

            donor_med = np.nanmedian(af_vals[m], axis=0)
            informative = donor_med >= informative_min
            if informative.sum() < 10:
                continue

            low_af_frac, n_info = _low_af_fraction(af_vals, informative, low_af_cut)
            l_q = np.nanquantile(low_af_frac[m], combined_quantile)

            se_drop = _binomial_se(d_q, n_total_loci)
            se_low = _binomial_se(l_q, np.maximum(n_info, 1))
            eff_drop = (global_dropout - d_q) / se_drop
            eff_low = (low_af_frac - l_q) / se_low
            extra = (
                (global_dropout >= d_q)
                & (low_af_frac >= l_q)
                & (eff_drop >= float(min_effect_size))
                & (eff_low >= float(min_effect_size))
                & m & ~fail
            )
            if extra.any():
                fail |= extra
                print(f"  {d}: +{int(extra.sum())} by combined moderate-risk gate")

    # Protect likely biological localized LOH (e.g. chr7) from low-quality relabeling
    if protect_localized_loh and loh_protected_mask.any():
        rescued = fail & loh_protected_mask
        if rescued.any():
            fail[rescued] = False
            print(f"  rescued {int(rescued.sum())} localized-LOH-like cells from Low_quality")

    # Summary
    donors = sorted(d for d in labels.unique() if d != mixed_label)
    summary = []
    for d in donors:
        m = labels == d
        n = int(m.sum())
        n_fail = int(fail[m.values].sum())
        thr = scores.loc[m, "threshold"].iloc[0] if m.any() else np.nan
        n_stable = int(scores.loc[m, "n_stable_loci"].iloc[0]) if m.any() else 0
        summary.append({
            "donor": d,
            "n_cells": n,
            "n_stable_loci": n_stable,
            "score_thr": float(thr),
            "n_lowq": n_fail,
            "pct_lowq": 100.0 * n_fail / max(n, 1),
        })
    summary_df = pd.DataFrame(summary).sort_values("donor")
    print(summary_df)

    # Write back labels
    new_labels = labels.to_numpy(dtype=object).copy()
    new_labels[fail & (new_labels != mixed_label)] = lowq_label
    sample_qc.dna.row_attrs[label_key] = new_labels
    sample_qc.dna.row_attrs["lowq_germline_fail"] = fail
    sample_qc.dna.row_attrs["lowq_germline_score"] = scores["composite_score"].to_numpy(dtype=float)
    sample_qc.dna.row_attrs["lowq_stable_loci_n"] = scores["n_stable_loci"].to_numpy(dtype=int)
    sample_qc.dna.row_attrs["loh_protected"] = loh_protected_mask
    sample_qc.dna.row_attrs["chr7_loh_protected"] = loh_protected_mask

    print(pd.Series(new_labels).value_counts(dropna=False))

    if show_global_heatmap:
        save_path = (
            os.path.join(heatmap_save_dir, "AF_lowq_global.png")
            if heatmap_save_dir else None
        )
        fast_af_heatmap(
            sample_qc,
            label_key=label_key,
            attribute="AF",
            title="AF: final doublets + germline-stable Low_quality",
            width=14.0,
            height=18.0,
            save_path=save_path,
            show=True,
        )

    if IS_MULTIPLEXED and sample_subsets is not None and show_subset_heatmaps:
        label_series = pd.Series(np.asarray(sample_qc.dna.row_attrs[label_key]).astype(object), index=af.index)
        for subset in sample_subsets:
            sub_af = _af_dataframe(sample_subsets[subset])
            sub_ids = pd.Index(sub_af.index)
            sample_subsets[subset].row_attrs[label_key] = (
                label_series.reindex(sub_ids).fillna(subset).to_numpy(dtype=object)
            )

            save_path = (
                os.path.join(heatmap_save_dir, f"AF_lowq_{subset}.png")
                if heatmap_save_dir else None
            )
            fast_af_heatmap(
                sample_subsets[subset],
                label_key=label_key,
                attribute="AF",
                title=f"{subset}: with Low_quality",
                width=12.0,
                height=16.0,
                save_path=save_path,
                show=True,
            )

    return summary_df, fail, scores, sample_subsets
