"""Cell-level QC flags: amplicon loss, spurious gain, and doublet-like AF shifts."""

import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════


def _corr_complete_obs(x, y, min_shared=40):
    """Correlation with complete observations only."""
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
    df,
    label_col="label",
    af_change_cut=0.30,
    min_variant_af=0.10,
    max_variant_af=0.90,
    z_k=3.5,
    min_informative=30,
    return_scores=True,
    min_gain_frac=0.01,
    min_loss_frac=0.01,
    cross_donor_refs=None,
    donor_name=None,
    disc_af_diff=0.20,
    min_disc_variants=5,
):
    """
    Flag cells with doublet-like AF profiles.

    When cross_donor_refs is provided, computes frac_changed using only
    donor-discriminating variants (where this donor's consensus differs
    from at least one other donor by >= disc_af_diff). This concentrates
    the metric on the variants that actually distinguish donors, preventing
    dilution by shared germline.

    Falls back to all-informative-variant mode when:
      - cross_donor_refs is None (single-donor or not provided)
      - fewer than min_disc_variants discriminating variants found

    Parameters
    ----------
    df : DataFrame
        [label_col] + AF variant columns. One row per cell.
    label_col : str
        Column containing donor labels.
    af_change_cut : float
        Per-variant AF deviation threshold to count as "changed".
    min_variant_af, max_variant_af : float
        AF range for a variant to be considered informative.
    z_k : float
        MAD-based cutoff strength for the legacy score threshold.
    min_informative : int
        Minimum informative variants; if fewer, use all variants.
    min_gain_frac, min_loss_frac : float
        Minimum gain/loss fractions for the bidirectional gate.
    cross_donor_refs : dict, optional
        {donor_name: pd.Series} of median AF profiles for all donors.
    donor_name : str, optional
        Which donor this subset belongs to.
    disc_af_diff : float
        Minimum absolute AF difference between this donor and any other
        to consider a variant discriminating.
    min_disc_variants : int
        Fall back to all-informative mode if fewer found.

    Returns
    -------
    fail : np.ndarray[bool]
    qc : dict (if return_scores=True)
        Keys: frac_changed, frac_gain, frac_loss, frac_changed_bidir,
        af_dilution, change_score, threshold, fail_bidir,
        n_disc_variants, used_discriminating.
    """
    af = df.drop(columns=[label_col]).astype(float)
    med = af.median(axis=0, skipna=True)

    informative = (med >= min_variant_af) & (med <= max_variant_af)
    mutant_like = med >= 0.30

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

    def robust_z(x):
        x = np.asarray(x, dtype=float)
        medx = np.nanmedian(x)
        mad = np.nanmedian(np.abs(x - medx)) + 1e-12
        return 0.6745 * (x - medx) / mad

    score = robust_z(frac_changed.values) + robust_z(af_dilution.values)

    s_med = np.nanmedian(score)
    s_mad = np.nanmedian(np.abs(score - s_med)) + 1e-12
    thr = s_med + z_k * 1.4826 * s_mad
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


def piecewise_elbow_threshold(values, min_frac=0.02, max_frac=0.40, slope_ratio_min=3.0):
    """
    Find elbow on a descending curve by fitting two lines.

    Returns (threshold_value, split_index, info_dict).
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
        x1, y1 = x[:k], y[:k]
        x2, y2 = x[k:], y[k:]
        m1, b1 = np.polyfit(x1, y1, 1)
        m2, b2 = np.polyfit(x2, y2, 1)

        if m1 < 0 and m2 < 0:
            ratio = abs(m1) / (abs(m2) + 1e-12)
            if ratio >= slope_ratio_min:
                sse = np.sum((y1 - (m1 * x1 + b1)) ** 2) + np.sum((y2 - (m2 * x2 + b2)) ** 2)
                if (best is None) or (sse < best["sse"]):
                    best = {"k": k, "thr": y[k], "sse": sse, "m1": m1, "m2": m2, "ratio": ratio}

    if best is None:
        for k in range(kmin, kmax + 1):
            x1, y1 = x[:k], y[:k]
            x2, y2 = x[k:], y[k:]
            m1, b1 = np.polyfit(x1, y1, 1)
            m2, b2 = np.polyfit(x2, y2, 1)
            sse = np.sum((y1 - (m1 * x1 + b1)) ** 2) + np.sum((y2 - (m2 * x2 + b2)) ** 2)
            ratio = abs(m1) / (abs(m2) + 1e-12)
            if (best is None) or (sse < best["sse"]):
                best = {"k": k, "thr": y[k], "sse": sse, "m1": m1, "m2": m2, "ratio": ratio}

    return best["thr"], best["k"], best


def define_doublet_cutoff_per_donor(
    sample_qc,
    label_key="label",
    exclude_labels=("Mixed", "Low_quality"),
    min_frac=0.01,
    max_frac=0.55,
    slope_ratio_min=3.0,
    n_boot=200,
    ci=0.95,
    near_ci_mult=1.0,
    plot=True,
    figsize=(5.0, 2.6),
    plot_tiny_heatmap=False,
    heatmap_max_cells=250,
    heatmap_cmap="viridis",
    use_bidirectional=True,
    min_gain_frac=0.01,
    min_loss_frac=0.01,
    disc_af_diff=0.20,
    min_disc_variants=5,
):
    """
    Per-donor doublet detection with data-driven thresholds.

    Builds cross-donor consensus AF profiles, identifies discriminating
    variants, then computes per-cell doublet metrics restricted to those
    variants. Falls back to all-informative mode when donors share too
    many variants (< min_disc_variants discriminating).

    Parameters
    ----------
    sample_qc : mosaic Sample
        Sample with dna assay and row_attrs[label_key] set.
    label_key : str
        Row attribute containing donor labels.
    exclude_labels : tuple
        Labels to skip (e.g. "Mixed", "Low_quality").
    disc_af_diff : float
        Minimum AF difference between donors for a variant to be
        considered discriminating.
    min_disc_variants : int
        Minimum discriminating variants to use discriminating mode.
    [other params]
        See flag_doublet_change_cells and piecewise_elbow_threshold.

    Returns
    -------
    summary_df : pd.DataFrame
        Per-donor summary with thresholds, flagged counts, etc.
    donor_calls : dict
        Per-donor detailed results for apply_doublet_calls_with_corr_reassign.
    """
    donors = sorted([d for d in set(sample_qc.dna.row_attrs[label_key])
                     if d not in set(exclude_labels)])

    # ── Build cross-donor consensus profiles ──
    all_af = sample_qc.dna.get_attribute("AF", constraint="row+col")
    if not isinstance(all_af, pd.DataFrame):
        all_af = pd.DataFrame(all_af)
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
        from itertools import combinations
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

        df_af = assay.get_attribute("AF", constraint="row+col")
        if not isinstance(df_af, pd.DataFrame):
            df_af = pd.DataFrame(df_af)

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
            y = np.sort(fc)[::-1]
            x = np.arange(len(y))

            disc_label = (f" [{qc['n_disc_variants']} disc. vars]"
                          if qc.get("used_discriminating") else "")

            if plot_tiny_heatmap:
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
                A = A[:, col_order]

                fig = plt.figure(figsize=(max(figsize[0], 7.0), figsize[1]))
                gs = fig.add_gridspec(
                    1, 4, width_ratios=[1.2, 0.12, 0.18, 1.8], wspace=0.0,
                )
                ax0 = fig.add_subplot(gs[0, 0])
                ax1 = fig.add_subplot(gs[0, 2])
                ax2 = fig.add_subplot(gs[0, 3])

                ax0.plot(x, y, lw=1.2)
                ax0.axhline(thr, color="red", ls="--", lw=1, label=f"thr={thr:.3f}")
                if np.isfinite(ci_low) and np.isfinite(ci_high):
                    ax0.axhspan(ci_low, ci_high, color="red", alpha=0.12,
                                label=f"{int(ci * 100)}% CI")
                if k >= 0:
                    ax0.axvline(k, color="red", ls=":", lw=1)
                ax0.set_title(
                    f"{donor} | {metric_name}{disc_label} | "
                    f"flagged {flagged}/{n} ({100 * flagged / max(n, 1):.1f}%)"
                )
                ax0.set_xlabel("cells (sorted high→low)")
                ax0.set_ylabel(metric_name)
                ax0.legend(fontsize=7, loc="best")

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
            else:
                plt.figure(figsize=figsize)
                plt.plot(x, y, lw=1.2)
                plt.axhline(thr, color="red", ls="--", lw=1, label=f"thr={thr:.3f}")
                if np.isfinite(ci_low) and np.isfinite(ci_high):
                    plt.axhspan(ci_low, ci_high, color="red", alpha=0.12,
                                label=f"{int(ci * 100)}% CI")
                if k >= 0:
                    plt.axvline(k, color="red", ls=":", lw=1)
                plt.title(
                    f"{donor} | {metric_name}{disc_label} | "
                    f"flagged {flagged}/{n} ({100 * flagged / max(n, 1):.1f}%)"
                )
                plt.xlabel("cells (sorted high→low)")
                plt.ylabel(metric_name)
                plt.legend(fontsize=8, loc="best")
                plt.tight_layout()
                plt.show()

    summary_df = pd.DataFrame(summary_rows).sort_values("donor").reset_index(drop=True)
    return summary_df, donor_calls


def apply_doublet_calls_with_corr_reassign(
    sample_qc,
    donor_calls,
    donors=None,
    label_key="label",
    band=0.02,
    min_shared=40,
    delta_corr=0.00,
    mixed_label="Mixed",
    write_back=True,
):
    """
    Reassign borderline cells using AF-profile correlation.

    Note: write_back=True writes labels into the returned subset assays,
    NOT into sample_qc directly. The caller must propagate labels back
    into sample_qc manually (e.g. via barcode matching).

    Returns
    -------
    reassign_results : dict[str, pd.DataFrame]
    final_doublet_calls : dict[str, np.ndarray[bool]]
    sample_subsets : dict[str, assay]
    """
    if donors is None:
        donors = sorted(list(donor_calls.keys()))

    all_af = sample_qc.dna.get_attribute("AF", constraint="row+col")
    if not isinstance(all_af, pd.DataFrame):
        all_af = pd.DataFrame(all_af)
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

        df_af = assay.get_attribute("AF", constraint="row+col")
        if not isinstance(df_af, pd.DataFrame):
            df_af = pd.DataFrame(df_af)

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
    df,
    label_col="label",
    informative_range=(0.10, 0.90),
    stability_quantile=0.35,
    min_stable_loci=25,
    fallback_quantile=0.50,
    min_shared_corr=20,
    score_weights=(1.0, 1.0, 0.7),
    k_mad=5.0,
    q=0.995,
    min_cells=20,
    return_scores=True,
):
    """
    Flag low-quality cells by measuring deviation from donor consensus
    at germline-stable loci.

    For each donor, identifies stable loci (informative AF range, lowest
    variance) and computes three per-cell metrics:
      1. Median absolute deviation from consensus at stable loci
      2. 1 − Pearson correlation with consensus profile
      3. NaN fraction across stable loci

    Each metric is robust-Z normalised and combined into a weighted
    composite score. Cells exceeding an adaptive MAD-based threshold
    per donor are flagged. Mixed and Low_quality labels are excluded
    from donor processing.

    Returns
    -------
    fail : np.ndarray[bool]
    scores : pd.DataFrame (if return_scores=True)
        Columns: noise_mad, one_minus_corr, nan_frac, composite_score,
        threshold, n_stable_loci.
    """
    assert label_col in df.columns, f"'{label_col}' not found in columns"

    labels = df[label_col].astype(str)
    af = df.drop(columns=[label_col]).apply(pd.to_numeric, errors="coerce")

    with np.errstate(all="ignore"):
        p99 = np.nanpercentile(af.values, 99)
    if np.isfinite(p99) and p99 > 1.5:
        af = af / 100.0

    exclude = {"Mixed", "Low_quality"}
    donors = sorted(d for d in labels.unique() if d not in exclude)
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

    def _robust_z(x):
        x = np.asarray(x, dtype=float)
        med = np.nanmedian(x)
        mad = np.nanmedian(np.abs(x - med)) + 1e-12
        return 0.6745 * (x - med) / mad

    def _corr(x, y):
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < min_shared_corr:
            return np.nan
        xv, yv = x[m], y[m]
        if np.std(xv) < 1e-12 or np.std(yv) < 1e-12:
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
            s_mad = np.nanmedian(np.abs(s - s_med)) + 1e-12
            thr = max(s_med + k_mad * 1.4826 * s_mad, np.nanquantile(s, q))
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
    df,
    label_col="label",
    silent_max=0.02,
    k_mad=5.0,
    high_floor=0.15,
    frac_spike_threshold=0.30,
    min_silent=10,
    return_scores=True,
):
    """Flag cells with spurious AF spikes in variants expected to be silent."""
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
# EXAMPLE / DEMO
# ═══════════════════════════════════════════════════════════════════════════════


def doublets_example(
    seed=42,
    n_variants=80,
    n_singlet=120,
    n_doublet=30,
    af_change_cut=0.30,
    figsize=(10, 3.5),
    return_data=True,
):
    """Mock example illustrating frac_changed for singlets vs doublets."""
    np.random.seed(seed)

    consensus = np.concatenate([
        np.random.normal(0.05, 0.02, 25),
        np.random.normal(0.50, 0.05, 35),
        np.random.normal(0.95, 0.02, 20),
    ])

    if n_variants != 80:
        x_old = np.linspace(0, 1, len(consensus))
        x_new = np.linspace(0, 1, n_variants)
        consensus = np.interp(x_new, x_old, consensus)

    consensus = np.clip(consensus, 0, 1)

    singlets = consensus + np.random.normal(0, 0.04, (n_singlet, n_variants))
    singlets = np.clip(singlets, 0, 1)

    doublets = consensus + np.random.normal(0, 0.10, (n_doublet, n_variants))
    shift_mask = np.random.rand(n_doublet, n_variants) < 0.35
    doublets[shift_mask] += np.random.choice([-0.35, 0.35], size=shift_mask.sum())
    doublets = np.clip(doublets, 0, 1)

    X = np.vstack([singlets, doublets])
    cell_type = np.array(["Singlet"] * n_singlet + ["Doublet"] * n_doublet)

    df_af = pd.DataFrame(X, columns=[f"V{i + 1}" for i in range(n_variants)])

    med = df_af.median(axis=0)
    informative = (med >= 0.10) & (med <= 0.90)
    delta = (df_af.sub(med, axis=1)).abs()
    changed = delta.loc[:, informative] >= af_change_cut
    frac_changed = changed.sum(axis=1) / max(1, int(informative.sum()))

    out = pd.DataFrame({"cell_type": cell_type, "frac_changed": frac_changed})
    summary = out.groupby("cell_type")["frac_changed"].describe()[["count", "mean", "50%", "std"]]
    print(summary)

    row_order = np.argsort(-frac_changed.values)
    mat = df_af.iloc[row_order, :]
    ann = pd.Series(cell_type[row_order]).map({"Singlet": 0, "Doublet": 1}).values.reshape(-1, 1)

    fig, axes = plt.subplots(1, 3, figsize=figsize, gridspec_kw={"width_ratios": [1.2, 0.15, 6]})

    axes[0].plot(frac_changed.values[row_order], np.arange(len(row_order)), lw=1)
    axes[0].invert_yaxis()
    axes[0].set_title("frac_changed")
    axes[0].set_xlabel("value")
    axes[0].set_ylabel("cells (sorted)")

    sns.heatmap(
        ann, ax=axes[1], cbar=False,
        cmap=sns.color_palette(["#1f77b4", "#d62728"], as_cmap=True),
        xticklabels=False, yticklabels=False,
    )
    axes[1].set_title("type")

    sns.heatmap(mat, ax=axes[2], cmap="viridis", xticklabels=False, yticklabels=False, cbar=True)
    axes[2].set_title("Mock AF matrix (cells x variants)")

    plt.tight_layout()
    plt.show()

    if return_data:
        return {
            "df_af": df_af,
            "summary": summary,
            "out": out,
            "consensus": consensus,
        }

# ═══════════════════════════════════════════════════════════════════════════════
# FAST AF HEATMAP (matplotlib, rasterized) — drop-in for sample.dna.heatmap()
# ═══════════════════════════════════════════════════════════════════════════════


def _af_dataframe(assay, attribute="AF"):
    """Return an assay attribute as a DataFrame regardless of Mosaic's return type."""
    df = assay.get_attribute(attribute, constraint="row+col")
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)
    return df


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
    from matplotlib.colors import ListedColormap
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
        Per-group and total caps on rendered cells. Large groups are evenly
        subsampled before within-group clustering.
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
    from matplotlib.colors import ListedColormap, TwoSlopeNorm

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


def _get_assay_labels(assay, label_key="label"):
    """Return row labels from ``row_attrs[label_key]`` with a safe fallback."""
    if hasattr(assay, "row_attrs") and label_key in assay.row_attrs:
        return np.asarray(assay.row_attrs[label_key]).astype(str)
    try:
        return np.asarray(assay.get_labels()).astype(str)
    except Exception as exc:
        raise KeyError(
            f"Could not find labels in assay.row_attrs[{label_key!r}] or assay.get_labels()."
        ) from exc


def _set_assay_labels(assay, labels, label_key="label"):
    """Set Mosaic labels and row_attrs label_key consistently."""
    labels = np.asarray(labels).astype(object)
    try:
        assay.set_labels(clusters=labels)
    except Exception:
        # Some Mosaic assay objects expose row_attrs but not set_labels.
        pass
    try:
        assay.row_attrs[label_key] = labels
    except Exception:
        pass
    return labels


def _cluster_mean_af(sample_or_assay, label_key="label", attribute="AF"):
    """Return AF dataframe, labels series, and per-label mean AF profiles."""
    assay = getattr(sample_or_assay, "dna", sample_or_assay)
    af = _af_dataframe(assay, attribute=attribute).apply(pd.to_numeric, errors="coerce")
    labels = pd.Series(_get_assay_labels(assay, label_key=label_key), index=af.index, name=label_key)
    grouped_means = af.groupby(labels, sort=False).mean()
    return af, labels, grouped_means


def _safe_profile_corr(grouped_means):
    """Correlation between row profiles, robust to NaNs and zero-variance rows."""
    if grouped_means.shape[0] == 0:
        return pd.DataFrame()
    if grouped_means.shape[0] == 1:
        return pd.DataFrame([[1.0]], index=grouped_means.index, columns=grouped_means.index)

    A = grouped_means.apply(pd.to_numeric, errors="coerce").copy()
    col_med = A.median(axis=0, skipna=True)
    A = A.fillna(col_med).fillna(0.0)

    arr = A.to_numpy(dtype=float)
    corr = np.corrcoef(arr)
    corr = np.asarray(corr, dtype=float)
    corr = np.nan_to_num(corr, nan=-1.0, posinf=1.0, neginf=-1.0)
    corr = np.clip(corr, -1.0, 1.0)
    np.fill_diagonal(corr, 1.0)
    return pd.DataFrame(corr, index=grouped_means.index, columns=grouped_means.index)


def merge_correlated_af_clusters(
    sample_qc,
    IS_MULTIPLEXED=True,
    N_DONORS=None,
    label_key="label",
    attribute="AF",
    corr_threshold=0.90,
    merge_margin=0.02,
    max_iter=10,
    target_extra=2,
    plot_corr=True,
    corr_figsize=(16, 16),
    corr_annot=True,
    corr_save_dir=None,
    final_heatmap=True,
    heatmap_title="AF after cluster merge",
    heatmap_save_path=None,
    show=True,
):
    """
    Iteratively merge highly correlated AF clusters, then draw the fast AF heatmap.

    This replaces the older notebook block that repeatedly computed cluster mean
    AF correlations and ended with ``sample_qc.dna.heatmap(...)``. It keeps the
    same logic, but is safer around missing AF values, writes labels back to both
    ``set_labels`` and ``row_attrs[label_key]``, and uses ``fast_af_heatmap`` for
    the final plot.

    Parameters
    ----------
    sample_qc : mosaic Sample
        Sample containing ``.dna`` assay.
    IS_MULTIPLEXED : bool
        If False, demultiplexing/merging is skipped and only a message is printed.
    N_DONORS : int, optional
        Expected number of donors. If provided, merging stops once the cluster
        count is <= ``N_DONORS + target_extra``.
    label_key, attribute : str
        Label row attribute and DNA assay attribute to use.
    corr_threshold : float
        Merge only while the maximum off-diagonal cluster correlation is at least
        this value.
    merge_margin : float
        On each iteration, merge clusters with correlation at least
        ``max(corr_threshold, max_corr - merge_margin)``. This collapses the most
        similar cluster block without aggressively over-merging.
    max_iter : int
        Maximum merge iterations.
    target_extra : int
        Allowed extra clusters above ``N_DONORS`` before stopping.
    plot_corr : bool
        Draw a correlation clustermap after each merge.
    corr_save_dir : str, optional
        Directory to save per-iteration correlation plots.
    final_heatmap : bool
        Draw a final ``fast_af_heatmap``.
    heatmap_save_path : str, optional
        Path for the final heatmap PNG.
    show : bool
        Whether to display plots.

    Returns
    -------
    history : pandas.DataFrame
        One row per merge iteration with cluster counts and thresholds.
    heatmap_fig : matplotlib.figure.Figure or None
        Final AF heatmap figure if ``final_heatmap=True``.
    """
    if not IS_MULTIPLEXED:
        print("Single donor — demultiplexing skipped")
        return pd.DataFrame(), None

    if N_DONORS is None:
        target_max = None
    else:
        target_max = int(N_DONORS) + int(target_extra)

    assay = sample_qc.dna
    history = []

    for iteration in range(int(max_iter)):
        _af, labels, grouped_means = _cluster_mean_af(
            sample_qc,
            label_key=label_key,
            attribute=attribute,
        )

        n_before = int(grouped_means.shape[0])
        if n_before <= 1:
            print(f"All clusters merged after {iteration} iterations")
            break

        corr_df = _safe_profile_corr(grouped_means)
        corr_vals = corr_df.to_numpy(dtype=float).copy()
        np.fill_diagonal(corr_vals, -1.0)
        max_corr = float(np.nanmax(corr_vals))

        if not np.isfinite(max_corr) or max_corr < float(corr_threshold):
            print(
                f"Converged after {iteration + 1} iterations "
                f"({n_before} clusters, max off-diag corr={max_corr:.3f})"
            )
            break

        # Merge the most correlated block, not every moderately similar cluster.
        merge_corr = max(float(corr_threshold), max_corr - float(merge_margin))
        merge_dist = 1.0 - merge_corr

        try:
            import scipy.cluster.hierarchy as sch
            from scipy.spatial.distance import squareform
        except Exception as exc:
            raise ImportError("merge_correlated_af_clusters requires scipy.") from exc

        dist = 1.0 - corr_df.to_numpy(dtype=float)
        dist = np.nan_to_num(dist, nan=2.0, posinf=2.0, neginf=0.0)
        dist = np.clip(dist, 0.0, 2.0)
        dist = (dist + dist.T) / 2.0
        np.fill_diagonal(dist, 0.0)

        Z = sch.linkage(squareform(dist, checks=False), method="average")
        cluster_ids = sch.fcluster(Z, t=merge_dist, criterion="distance")

        # Stable relabeling: C1, C2, ... in the order returned for grouped_means.
        merge_guide = {
            str(old_label): f"C{int(new_label)}"
            for old_label, new_label in zip(grouped_means.index.astype(str), cluster_ids)
        }
        new_labels = labels.astype(str).map(merge_guide).fillna(labels.astype(str)).to_numpy(dtype=object)
        _set_assay_labels(assay, new_labels, label_key=label_key)

        # Recompute after merge for reporting/plotting.
        _af2, labels2, grouped_means2 = _cluster_mean_af(
            sample_qc,
            label_key=label_key,
            attribute=attribute,
        )
        n_after = int(grouped_means2.shape[0])

        history.append({
            "iteration": int(iteration + 1),
            "n_clusters_before": n_before,
            "n_clusters_after": n_after,
            "max_corr_before": max_corr,
            "merge_corr_threshold": merge_corr,
            "merge_dist_threshold": merge_dist,
        })

        print(
            f"  Iteration {iteration + 1}: {n_before} → {n_after} clusters "
            f"(merged at corr>{merge_corr:.3f}; max={max_corr:.3f})"
        )

        if plot_corr and n_after > 1:
            corr_df_plot = _safe_profile_corr(grouped_means2)
            cg = sns.clustermap(
                corr_df_plot,
                annot=bool(corr_annot),
                cmap="coolwarm",
                vmin=-1,
                vmax=1,
                center=0,
                annot_kws={"size": 6},
                figsize=corr_figsize,
            )
            cg.ax_heatmap.set_title(
                f"Cluster correlation (iteration {iteration + 1}, n={n_after})",
                pad=20,
            )
            if corr_save_dir:
                os.makedirs(corr_save_dir, exist_ok=True)
                cg.fig.savefig(
                    os.path.join(corr_save_dir, f"cluster_corr_iteration_{iteration + 1:02d}.png"),
                    dpi=150,
                    bbox_inches="tight",
                )
            if show:
                plt.show()
            else:
                plt.close(cg.fig)

        if target_max is not None and n_after <= target_max:
            print(f"Reached {n_after} clusters (target ≤{target_max})")
            break

        if N_DONORS is not None and n_after < int(N_DONORS):
            print(f"Warning: over-merged to {n_after} clusters (expected {N_DONORS})")
            break

    heatmap_fig = None
    if final_heatmap:
        heatmap_fig = fast_af_heatmap(
            sample_qc,
            label_key=label_key,
            attribute=attribute,
            title=heatmap_title,
            save_path=heatmap_save_path,
            show=show,
        )

    return pd.DataFrame(history), heatmap_fig

def apply_low_quality_qc(
    sample_qc,
    DNA_Utils,
    sample_subsets=None,
    IS_MULTIPLEXED=False,
    label_key="label",
    mixed_label="Mixed",
    lowq_label="Low_quality",
    k_mad=3.75,
    q=0.975,
    min_donor_cells=20,
    abs_global_dropout=0.50,
    abs_low_af_info=0.50,
    depth_adaptive=True,
    abs_floor_quantile=0.98,
    use_global_dropout_gate=True,
    use_low_af_gate=False,
    use_combined_gate=True,
    combined_quantile=0.95,
    min_effect_size=2,
    protect_localized_loh=False,
    loh_shift=0.35,
    loh_concentration=0.75,
    loh_max_chromosomes=2,
    loh_min_loci_per_chr=4,
    loh_min_total_loci=10,
    show_global_heatmap=True,
    show_subset_heatmaps=True,
    heatmap_save_dir=None,
):
    """
    Apply low-quality QC on top of existing labels and optionally plot heatmaps.

    Depth handling
    --------------
    The relative gates (per-donor MAD/quantile + binomial effect size) adapt to
    sample depth on their own. The *absolute* floors ``abs_global_dropout`` and
    ``abs_low_af_info`` do not — on a low-depth sample the median cell can already
    exceed 0.50 dropout, so a fixed 0.50 floor flags almost everything.

    With ``depth_adaptive=True`` (default) each absolute floor floats up to
    ``max(abs_floor, per_donor_quantile(abs_floor_quantile))`` so it only ever
    catches the donor's extreme tail. On high-quality donors the donor quantile
    sits below the floor, so behaviour is unchanged. Set either ``abs_*`` to
    ``None`` to disable that absolute gate entirely.

    Heatmaps now use ``fast_af_heatmap`` (rasterized matplotlib) instead of the
    Plotly-based ``sample.dna.heatmap``, which scales to large cell/variant counts.
    Pass ``heatmap_save_dir`` to also write PNGs.

    Returns
    -------
    summary_df : pd.DataFrame
    fail : np.ndarray[bool]
    scores : pd.DataFrame
    sample_subsets : dict or None
    """
    af = sample_qc.dna.get_attribute("AF", constraint="row+col")
    if not isinstance(af, pd.DataFrame):
        af = pd.DataFrame(af)

    labels = pd.Series(
        np.asarray(sample_qc.dna.row_attrs[label_key]).astype(str),
        index=af.index,
        name=label_key,
    )
    df = pd.concat([labels, af], axis=1)

    # Gate 1: base amplicon-loss from DNA_Utils
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

    # pooled fallback for small donors
    valid_pool = ~labels.isin([mixed_label, lowq_label])
    pool_vals = global_dropout[valid_pool.values]
    pool_med = np.nanmedian(pool_vals)
    pool_mad = np.nanmedian(np.abs(pool_vals - pool_med)) + 1e-12
    pool_thr = min(pool_med + k_mad * 1.4826 * pool_mad, np.nanquantile(pool_vals, q))

    # Localized-LOH mask (protected from Low_quality relabeling)
    loh_protected_mask = np.zeros(len(labels), dtype=bool)
    if protect_localized_loh:
        col_names = pd.Index(af.columns).astype(str)

        def _extract_chrom(name):
            s = str(name)
            import re
            m = re.search(r"chr\s*([0-9]{1,2}|[XYM])", s, flags=re.IGNORECASE)
            if m:
                return m.group(1).upper()
            m = re.match(r"\s*([0-9]{1,2}|[XYM])[:_\-]", s, flags=re.IGNORECASE)
            if m:
                return m.group(1).upper()
            return None

        chrom = np.array([_extract_chrom(c) for c in col_names], dtype=object)
        valid_chrom = pd.notna(chrom)

        if valid_chrom.any():
            uniq_chrom = [c for c in pd.unique(chrom[valid_chrom]) if c is not None]

            for d in sorted(x for x in labels.unique() if x not in (mixed_label, lowq_label)):
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

    # Gate 2: global dropout per donor
    if use_global_dropout_gate:
        for d in sorted(x for x in labels.unique() if x not in (mixed_label, lowq_label)):
            m = (labels == d).values
            if m.sum() == 0:
                continue

            donor_dropout = global_dropout[m]
            if m.sum() >= min_donor_cells:
                med = np.nanmedian(donor_dropout)
                mad = np.nanmedian(np.abs(donor_dropout - med)) + 1e-12
                dropout_thr = min(med + k_mad * 1.4826 * mad, np.nanquantile(donor_dropout, q))
            else:
                dropout_thr = pool_thr

            se_drop = np.sqrt(np.maximum(dropout_thr * (1.0 - dropout_thr) / n_total_loci, 1e-12))
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
        for d in sorted(x for x in labels.unique() if x not in (mixed_label, lowq_label)):
            m = (labels == d).values
            if m.sum() < min_donor_cells:
                continue

            donor_med = np.nanmedian(af_vals[m], axis=0)
            informative = donor_med >= informative_min
            if informative.sum() < 10:
                continue

            low_af_frac = np.mean(
                (~np.isfinite(af_vals[:, informative])) | (af_vals[:, informative] < low_af_cut),
                axis=1,
            )
            n_info = np.sum(np.isfinite(af_vals[:, informative]), axis=1)
            donor_low = low_af_frac[m]
            med_low = np.nanmedian(donor_low)
            mad_low = np.nanmedian(np.abs(donor_low - med_low)) + 1e-12
            low_thr = min(med_low + k_mad * 1.4826 * mad_low, np.nanquantile(donor_low, q))

            se_low = np.sqrt(np.maximum(low_thr * (1.0 - low_thr) / np.maximum(n_info, 1), 1e-12))
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
        for d in sorted(x for x in labels.unique() if x not in (mixed_label, lowq_label)):
            m = (labels == d).values
            if m.sum() < min_donor_cells:
                continue

            donor_dropout = global_dropout[m]
            d_q = np.nanquantile(donor_dropout, combined_quantile)

            donor_med = np.nanmedian(af_vals[m], axis=0)
            informative = donor_med >= informative_min
            if informative.sum() < 10:
                continue

            low_af_frac = np.mean(
                (~np.isfinite(af_vals[:, informative])) | (af_vals[:, informative] < low_af_cut),
                axis=1,
            )
            n_info = np.sum(np.isfinite(af_vals[:, informative]), axis=1)
            l_q = np.nanquantile(low_af_frac[m], combined_quantile)

            se_drop = np.sqrt(np.maximum(d_q * (1.0 - d_q) / n_total_loci, 1e-12))
            se_low = np.sqrt(np.maximum(l_q * (1.0 - l_q) / np.maximum(n_info, 1), 1e-12))
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

    # Protect likely biological chr7 LOH from low-quality relabeling
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
            sub_af = sample_subsets[subset].get_attribute("AF", constraint="row+col")
            if not isinstance(sub_af, pd.DataFrame):
                sub_af = pd.DataFrame(sub_af)
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

# ═══════════════════════════════════════════════════════════════════════════════
# RESIDUAL MIXED-LIKE CELL RESCUE (CORRELATION-BASED)
# ═══════════════════════════════════════════════════════════════════════════════
#
# Append this block to qc_filters.py / DNA_Utils. It depends on helpers already
# defined in the module:
#   - _corr_complete_obs   (correlation on complete observations)
#   - _group_palette       (Mission Bio categorical colours)
# so do NOT also paste the standalone corr_complete_obs from the second script;
# it is the same function under a different name and would be redundant.


def find_remaining_cells_like_mixed(
    sample_qc,
    sample_subsets,
    reassign_results,
    donor_calls=None,
    donor="Donor_A",
    donors=None,
    label_key="label",
    mixed_label="Mixed",
    lowq_label="Low_quality",
    disc_af_diff=0.20,
    min_disc_variants=5,
    min_shared=40,
    delta_corr_cut=0.02,
    min_corr_mixed=0.75,
    use_discriminating_variants=True,
    plot=True,
):
    """
    Scan remaining donor-labelled cells and ask whether their AF profile is
    more correlated with the donor-specific Mixed cluster than with the donor
    singlet reference.

    For one donor subset, builds a donor reference (high-confidence singlets)
    and a Mixed reference, optionally restricts to donor-discriminating
    variants, then correlates every still-donor-labelled cell against both.
    Cells whose Mixed correlation exceeds the donor correlation by at least
    ``delta_corr_cut`` (and whose Mixed correlation is at least
    ``min_corr_mixed``) are flagged Mixed-like.

    Parameters
    ----------
    sample_qc : mosaic Sample
        Full sample, used to build cross-donor references.
    sample_subsets : dict
        Per-donor assay subsets from apply_doublet_calls_with_corr_reassign.
    reassign_results : dict
        Per-donor reassignment tables; used to pick high-confidence donor cells.
    donor_calls : dict, optional
        Per-donor doublet results; merged in for context if provided.
    donor : str
        Donor subset to scan.
    donors : sequence, optional
        Donor universe for discriminating-variant detection. Defaults to all
        non-Mixed/non-Low_quality labels.
    disc_af_diff, min_disc_variants : float, int
        Discriminating-variant criteria (see flag_doublet_change_cells).
    min_shared : int
        Minimum shared finite loci for a correlation to be computed.
    delta_corr_cut : float
        Minimum (corr_mixed - corr_donor) to call a cell Mixed-like.
    min_corr_mixed : float
        Minimum absolute correlation to the Mixed reference.
    use_discriminating_variants : bool
        Restrict correlations to donor-discriminating variants when enough exist.
    plot : bool
        Draw the per-donor scatter and delta-correlation histogram.

    Returns
    -------
    out : pd.DataFrame
        One row per remaining donor-labelled cell, sorted by Mixed-correlation
        excess, with a ``mixed_like_by_corr`` boolean flag.
    """
    # ── Donor-specific subset after doublet reassignment ──
    sub = sample_subsets[donor]

    af_sub = sub.get_attribute("AF", constraint="row+col")
    if not isinstance(af_sub, pd.DataFrame):
        af_sub = pd.DataFrame(af_sub)

    labels_sub = pd.Series(
        np.asarray(sub.row_attrs[label_key]).astype(str),
        index=af_sub.index,
        name=label_key,
    )

    scan_ids = labels_sub.index[labels_sub == donor]
    mixed_ids = labels_sub.index[labels_sub == mixed_label]

    if len(scan_ids) == 0:
        raise ValueError(f"No remaining cells labelled {donor!r} in sample_subsets[{donor!r}].")

    if len(mixed_ids) < 3:
        raise ValueError(
            f"Too few {mixed_label!r} cells in sample_subsets[{donor!r}] "
            f"to build a stable Mixed reference: n={len(mixed_ids)}"
        )

    # ── High-confidence donor cells as the donor reference ──
    rr = reassign_results.get(donor)

    if rr is not None:
        donor_ref_ids = rr.index[
            (~rr["fail_final"].astype(bool))
            & rr["group"].isin(["LowChange", "Borderline→Donor"])
        ]
        donor_ref_ids = pd.Index(donor_ref_ids).intersection(af_sub.index)
        donor_ref_ids = donor_ref_ids[labels_sub.reindex(donor_ref_ids) == donor]
    else:
        donor_ref_ids = scan_ids

    if len(donor_ref_ids) < 10:
        print(
            f"Warning: only {len(donor_ref_ids)} high-confidence donor reference cells. "
            "Falling back to all currently donor-labelled cells."
        )
        donor_ref_ids = scan_ids

    donor_ref = af_sub.loc[donor_ref_ids].median(axis=0, skipna=True)
    mixed_ref = af_sub.loc[mixed_ids].median(axis=0, skipna=True)

    # ── Optional: restrict to donor-discriminating variants ──
    use_cols = list(af_sub.columns)

    if use_discriminating_variants:
        all_af = sample_qc.dna.get_attribute("AF", constraint="row+col")
        if not isinstance(all_af, pd.DataFrame):
            all_af = pd.DataFrame(all_af)

        all_labels = pd.Series(
            np.asarray(sample_qc.dna.row_attrs[label_key]).astype(str),
            index=all_af.index,
        )

        if donors is None:
            donors = sorted(
                d for d in pd.unique(all_labels)
                if d not in {mixed_label, lowq_label}
            )

        donor_refs = {}
        for d in donors:
            m = all_labels == d
            if m.sum() >= 10:
                donor_refs[d] = all_af.loc[m].median(axis=0, skipna=True)

        disc_cols = set()
        if donor in donor_refs:
            this_ref = donor_refs[donor]
            for other_d, other_ref in donor_refs.items():
                if other_d == donor:
                    continue
                common = (
                    this_ref.index
                    .intersection(other_ref.index)
                    .intersection(af_sub.columns)
                )
                diff = (this_ref.reindex(common) - other_ref.reindex(common)).abs()
                disc_cols.update(diff[diff >= float(disc_af_diff)].index)

        disc_cols = [c for c in af_sub.columns if c in disc_cols]

        if len(disc_cols) >= int(min_disc_variants):
            use_cols = disc_cols
            print(f"{donor}: using {len(use_cols)} donor-discriminating variants for correlation.")
        else:
            print(
                f"{donor}: only {len(disc_cols)} discriminating variants found; "
                "using all variants."
            )

    # ── Correlate each remaining donor-labelled cell to donor and Mixed refs ──
    donor_ref_use = donor_ref.reindex(use_cols).to_numpy(dtype=float)
    mixed_ref_use = mixed_ref.reindex(use_cols).to_numpy(dtype=float)

    rows = []
    for cell in scan_ids:
        x = af_sub.loc[cell, use_cols].to_numpy(dtype=float)
        corr_donor = _corr_complete_obs(x, donor_ref_use, min_shared=min_shared)
        corr_mixed = _corr_complete_obs(x, mixed_ref_use, min_shared=min_shared)
        rows.append({
            "cell": cell,
            "corr_donor": corr_donor,
            "corr_mixed": corr_mixed,
            "delta_corr_mixed_minus_donor": corr_mixed - corr_donor,
        })

    out = pd.DataFrame(rows).set_index("cell")

    # Add original doublet metrics, if available
    if donor_calls is not None and donor in donor_calls:
        dc = donor_calls[donor]
        metric_df = pd.DataFrame(
            {
                "doublet_metric": dc.get("metric_values", dc.get("frac_changed")),
                "frac_gain": dc.get("frac_gain", np.full(len(dc["cell_ids"]), np.nan)),
                "frac_loss": dc.get("frac_loss", np.full(len(dc["cell_ids"]), np.nan)),
                "original_fail": dc.get("fail", np.full(len(dc["cell_ids"]), False)),
            },
            index=pd.Index(dc["cell_ids"]),
        )
        out = out.join(metric_df, how="left")

    out["mixed_like_by_corr"] = (
        (out["delta_corr_mixed_minus_donor"] >= float(delta_corr_cut))
        & (out["corr_mixed"] >= float(min_corr_mixed))
    )

    out = out.sort_values("delta_corr_mixed_minus_donor", ascending=False)

    print(
        f"{donor}: {int(out['mixed_like_by_corr'].sum())} / {len(out)} "
        f"remaining donor-labelled cells are Mixed-like by correlation "
        f"(delta_corr ≥ {delta_corr_cut}, corr_mixed ≥ {min_corr_mixed})."
    )

    if plot:
        plt.figure(figsize=(4, 4))
        plt.scatter(out["corr_donor"], out["corr_mixed"], s=10, alpha=0.6)
        lim_min = np.nanmin(out[["corr_donor", "corr_mixed"]].to_numpy())
        lim_max = np.nanmax(out[["corr_donor", "corr_mixed"]].to_numpy())
        plt.plot([lim_min, lim_max], [lim_min, lim_max], linestyle="--", linewidth=1)
        plt.xlabel("corr to donor reference")
        plt.ylabel("corr to Mixed reference")
        plt.title(f"{donor}: remaining cells vs Mixed")
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(5, 2.5))
        plt.hist(out["delta_corr_mixed_minus_donor"].dropna(), bins=50)
        plt.axvline(delta_corr_cut, linestyle="--", linewidth=1)
        plt.xlabel("corr_mixed - corr_donor")
        plt.ylabel("cells")
        plt.title(f"{donor}: Mixed-correlation excess")
        plt.tight_layout()
        plt.show()

    return out


def visualise_mixed_like_rescue(
    sample_qc,
    sample_subsets,
    reassign_results,
    summary_df,
    donor_calls=None,
    label_key="label",
    mixed_label="Mixed",
    lowq_label="Low_quality",
    delta_corr_cut=-0.03,
    min_corr_mixed=0.80,
    disc_af_diff=0.25,
    min_disc_variants=10,
    use_discriminating_variants=True,
    min_shared=40,
    plot_per_donor=False,
    plot_summary=True,
    figsize=(13, 6),
    palette="auto",
    save_path=None,
    show=True,
):
    """
    Scan every donor subset for residual Mixed-like cells and summarise the
    result across donors in a single overview figure.

    Wraps find_remaining_cells_like_mixed over all donors present in both
    ``summary_df`` and ``sample_subsets``, concatenates the per-cell scans, and
    builds a per-donor rescue summary. The overview figure has three panels:
      - corr-to-donor vs corr-to-Mixed for every remaining cell, coloured by
        donor, with Mixed-like cells outlined in black;
      - per-donor counts of remaining vs Mixed-like cells;
      - the per-donor distribution of (corr_mixed - corr_donor) against the cut.

    Defaults reproduce the stringent clean-up settings (delta_corr_cut=-0.03,
    min_corr_mixed=0.80, disc_af_diff=0.25, min_disc_variants=10).

    Parameters
    ----------
    sample_qc, sample_subsets, reassign_results, donor_calls
        Passed straight through to find_remaining_cells_like_mixed.
    summary_df : pd.DataFrame
        Per-donor table with a ``donor`` column; defines the donors to scan.
    plot_per_donor : bool
        Draw the individual scatter/histogram for each donor as well.
    plot_summary : bool
        Draw the combined cross-donor overview figure.
    palette : "auto", dict, or None
        Donor colours. "auto" uses the Mission Bio categorical cycle so colours
        match the rest of the pipeline; pass a {donor: colour} dict to override.
    [other params]
        See find_remaining_cells_like_mixed.

    Returns
    -------
    all_donor_corr_scan : pd.DataFrame
        One row per scanned cell across all donors.
    summary_corr_rescue : pd.DataFrame
        Per-donor rescue summary.
    fig : matplotlib.figure.Figure or None
        Overview figure if plot_summary=True and any cells were scanned.
    """
    donors = [d for d in summary_df["donor"].tolist() if d in sample_subsets]

    donor_corr_scans = {}
    for donor in donors:
        print(f"\nScanning {donor} for remaining Mixed-like cells...")
        donor_corr_scans[donor] = find_remaining_cells_like_mixed(
            sample_qc=sample_qc,
            sample_subsets=sample_subsets,
            reassign_results=reassign_results,
            donor_calls=donor_calls,
            donor=donor,
            donors=donors,
            label_key=label_key,
            mixed_label=mixed_label,
            lowq_label=lowq_label,
            delta_corr_cut=delta_corr_cut,
            min_corr_mixed=min_corr_mixed,
            disc_af_diff=disc_af_diff,
            min_disc_variants=min_disc_variants,
            use_discriminating_variants=use_discriminating_variants,
            min_shared=min_shared,
            plot=plot_per_donor,
        )

    all_donor_corr_scan = (
        pd.concat(donor_corr_scans, names=["donor", "cell"])
        .reset_index()
        .sort_values(
            ["donor", "mixed_like_by_corr", "delta_corr_mixed_minus_donor"],
            ascending=[True, False, False],
        )
    )

    summary_corr_rescue = (
        all_donor_corr_scan
        .groupby("donor")
        .agg(
            n_remaining_cells=("cell", "size"),
            n_corr_mixed_like=("mixed_like_by_corr", "sum"),
            max_delta_corr=("delta_corr_mixed_minus_donor", "max"),
            median_delta_corr=("delta_corr_mixed_minus_donor", "median"),
            median_corr_mixed=("corr_mixed", "median"),
            median_corr_donor=("corr_donor", "median"),
        )
        .reset_index()
    )

    fig = None
    if plot_summary and len(all_donor_corr_scan):
        if isinstance(palette, dict):
            palette_map = {str(k): v for k, v in palette.items()}
        else:
            palette_map = _group_palette(donors, all_labels=donors)

        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(
            2, 2, width_ratios=[1.5, 1.0], height_ratios=[1, 1],
            wspace=0.28, hspace=0.45,
        )
        ax_sc = fig.add_subplot(gs[:, 0])
        ax_bar = fig.add_subplot(gs[0, 1])
        ax_box = fig.add_subplot(gs[1, 1])

        # Panel 1: corr-to-donor vs corr-to-Mixed, coloured by donor
        for donor in donors:
            sub = all_donor_corr_scan[all_donor_corr_scan["donor"] == donor]
            colour = palette_map.get(str(donor), "#808080")
            flagged = sub["mixed_like_by_corr"].astype(bool)
            ax_sc.scatter(
                sub.loc[~flagged, "corr_donor"], sub.loc[~flagged, "corr_mixed"],
                s=10, alpha=0.5, color=colour, linewidths=0, label=str(donor),
            )
            ax_sc.scatter(
                sub.loc[flagged, "corr_donor"], sub.loc[flagged, "corr_mixed"],
                s=26, alpha=0.9, color=colour, edgecolors="black", linewidths=0.6,
            )

        finite = all_donor_corr_scan[["corr_donor", "corr_mixed"]].to_numpy(dtype=float)
        finite = finite[np.isfinite(finite).all(axis=1)]
        if len(finite):
            lo, hi = float(finite.min()), float(finite.max())
            ax_sc.plot([lo, hi], [lo, hi], ls="--", lw=1, color="#4a4a4a")
        ax_sc.axhline(min_corr_mixed, ls=":", lw=1, color="red",
                      label=f"min corr_mixed={min_corr_mixed}")
        ax_sc.set_xlabel("corr to donor reference")
        ax_sc.set_ylabel("corr to Mixed reference")
        ax_sc.set_title("Remaining donor cells vs Mixed (outlined = Mixed-like)")
        ax_sc.legend(fontsize=7, loc="best", framealpha=0.6)

        # Panel 2: per-donor remaining vs Mixed-like counts
        order = summary_corr_rescue["donor"].tolist()
        y = np.arange(len(order))
        rem = summary_corr_rescue["n_remaining_cells"].to_numpy(dtype=float)
        mix = summary_corr_rescue["n_corr_mixed_like"].to_numpy(dtype=float)
        bar_colours = [palette_map.get(str(d), "#808080") for d in order]
        ax_bar.barh(y, rem, color="#d3d3d3", label="remaining")
        ax_bar.barh(y, mix, color=bar_colours, label="Mixed-like")
        ax_bar.set_yticks(y)
        ax_bar.set_yticklabels(order, fontsize=8)
        ax_bar.invert_yaxis()
        ax_bar.set_xlabel("cells")
        ax_bar.set_title("Mixed-like vs remaining per donor")
        for yi, (m_, r_) in enumerate(zip(mix, rem)):
            ax_bar.text(r_, yi, f" {int(m_)}/{int(r_)}", va="center", fontsize=7)

        # Panel 3: delta_corr distribution by donor (drop donors with no finite delta)
        box_order, box_data = [], []
        for d in order:
            arr = all_donor_corr_scan.loc[
                all_donor_corr_scan["donor"] == d, "delta_corr_mixed_minus_donor"
            ].dropna().to_numpy()
            if len(arr):
                box_order.append(d)
                box_data.append(arr)

        if box_data:
            bp = ax_box.boxplot(
                box_data, vert=False, patch_artist=True, widths=0.6,
                flierprops=dict(marker=".", markersize=3, alpha=0.4),
            )
            for patch, d in zip(bp["boxes"], box_order):
                patch.set_facecolor(palette_map.get(str(d), "#808080"))
                patch.set_alpha(0.7)
            for med in bp["medians"]:
                med.set_color("black")
            ax_box.set_yticks(np.arange(1, len(box_order) + 1))
            ax_box.set_yticklabels(box_order, fontsize=8)
            ax_box.invert_yaxis()
        ax_box.axvline(delta_corr_cut, ls="--", lw=1, color="red",
                       label=f"cut={delta_corr_cut}")
        ax_box.set_xlabel("corr_mixed - corr_donor")
        ax_box.set_title("Mixed-correlation excess per donor")
        ax_box.legend(fontsize=7, loc="best")

        fig.suptitle("Residual Mixed-like cell rescue", fontsize=12, y=0.99)
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        if show:
            plt.show()

    return all_donor_corr_scan, summary_corr_rescue, fig