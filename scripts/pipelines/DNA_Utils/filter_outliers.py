"""Somatic variant outlier filtering restricted to clone–variant carrier cells.

For each somatic variant, AF outlier filtering is performed only on cells
belonging to clone labels that carry that exact variant (Het or Hom by
default) in ``clone_variant_calls``. Cells from WT, missing, ADO, FP, small,
or unrelated clones are excluded from that variant's AF distribution. Variant
IDs are matched on their canonical chr:pos:ref/alt key, so a leading gene
prefix (e.g. ``TP53:chr17:7577539:G/A``) matches a bare ``chr17:7577539:G/A``.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_filename(s: str) -> str:
    """Sanitise a string for use in a filename."""
    return (
        str(s)
        .replace("/", "_")
        .replace(":", "_")
        .replace("\\", "_")
        .replace(" ", "_")
    )


def _canonical_variant_key(v) -> str:
    """Strip any leading gene prefix so 'TP53:chr17:7577539:G/A' matches
    'chr17:7577539:G/A'. Returns the substring from the first chr token."""
    s = str(v).strip()
    parts = s.split(":")
    for i, tok in enumerate(parts):
        if tok.lower().startswith("chr"):
            return ":".join(parts[i:])
    return s


def _is_het_clone_call(value) -> bool:
    """Return True only for heterozygous clone-level genotype calls."""
    if pd.isna(value):
        return False

    value = str(value).strip().lower()

    return (
        value.startswith("het")
        or value.startswith("heterozygous")
        or value in {"0/1", "1/0"}
    )


def _clone_call_matches(value, accepted=("het", "hom")) -> bool:
    """True if a clone-level call matches any accepted genotype class.

    ``accepted`` may contain 'het' and/or 'hom'. Values look like
    'Het (50.5%)', 'Hom (99.5%)', 'WT (0.3%)', or genotype strings such as
    '0/1' / '1/1'.
    """
    if pd.isna(value):
        return False

    v = str(value).strip().lower()
    accepted = {a.lower() for a in accepted}

    match = False
    if "het" in accepted:
        match = (
            match
            or v.startswith("het")
            or v.startswith("heterozygous")
            or v in {"0/1", "1/0"}
        )
    if "hom" in accepted:
        match = (
            match
            or v.startswith("hom")
            or v.startswith("homozygous")
            or v.startswith("hemi")
            or v == "1/1"
        )
    return match


def _prepare_clone_variant_calls(
    clone_variant_calls: pd.DataFrame,
    variant_column: str,
) -> pd.DataFrame:
    """Return clone-level genotype calls indexed by canonical variant key.

    The variant column is located by whitespace-normalised name, so a wrapped
    header like 'Variant\\nname' still matches variant_column='Variant name'.
    """
    import re

    if not isinstance(clone_variant_calls, pd.DataFrame):
        raise TypeError("clone_variant_calls must be a pandas DataFrame.")

    calls = clone_variant_calls.copy()

    def _norm(c):
        return re.sub(r"\s+", " ", str(c)).strip().lower()

    target = _norm(variant_column)
    col_match = next((c for c in calls.columns if _norm(c) == target), None)

    if col_match is not None:
        calls[col_match] = calls[col_match].astype(str)
        calls = calls.set_index(col_match, drop=False)
    else:
        calls.index = calls.index.astype(str)

    # Match on canonical chr:pos:ref/alt, ignoring any leading gene prefix.
    calls.index = pd.Index(
        [_canonical_variant_key(x) for x in calls.index],
        name=calls.index.name,
    )

    return calls


def _ensure_barcode_index(
    df_AF: pd.DataFrame,
    df_DP: pd.DataFrame,
    Sample,
):
    """Use Sample.dna barcodes when extracted matrices have positional indices."""
    dna_barcodes = pd.Index(
        np.asarray(Sample.dna.row_attrs["barcode"]).astype(str)
    )

    if len(df_AF) == 0:
        raise ValueError("No cells remain after aligning df_AF and df_DP.")

    af_index_as_str = pd.Index(df_AF.index.astype(str))
    n_matching_barcodes = af_index_as_str.isin(dna_barcodes).sum()

    index_is_integer = pd.api.types.is_integer_dtype(df_AF.index)

    # Typical get_attribute output: positional / integer row index.
    if (
        len(df_AF) == len(dna_barcodes)
        and (index_is_integer or n_matching_barcodes == 0)
    ):
        df_AF = df_AF.copy()
        df_DP = df_DP.copy()

        df_AF.index = dna_barcodes
        df_DP.index = dna_barcodes

    # Positional index in a subset: map positions back to DNA barcodes.
    elif index_is_integer:
        barcode_map = {
            int(position): barcode
            for position, barcode in enumerate(dna_barcodes)
        }

        df_AF = df_AF.copy()
        df_DP = df_DP.copy()

        df_AF.index = pd.Index(
            [barcode_map.get(int(i), str(i)) for i in df_AF.index]
        )

        df_DP.index = pd.Index(
            [barcode_map.get(int(i), str(i)) for i in df_DP.index]
        )

    else:
        df_AF = df_AF.copy()
        df_DP = df_DP.copy()

        df_AF.index = df_AF.index.astype(str)
        df_DP.index = df_DP.index.astype(str)

    return df_AF, df_DP


def _jitter_plot_by_label(
    df: pd.DataFrame,
    y_col: str,
    outdir: str,
    label_col: str = "label",
    jitter: float = 0.18,
    alpha: float = 0.6,
    s: float = 18,
    rect_width: float = 0.35,
    rect_height: float = 1.2,
    cap_width: float = 0.18,
    title: str = None,
    dpi: int = 300,
    y_limits: tuple = (-1, 101),
    filename_suffix: str = "",
    show_plot: bool = True,
):
    """Jitter plot of AF by label with mean ± SD whiskers."""
    d = df[[label_col, y_col]].copy()
    d[y_col] = pd.to_numeric(d[y_col], errors="coerce")
    d = d.dropna(subset=[label_col, y_col])

    if d.empty:
        return

    labels = pd.Index(pd.unique(d[label_col]))
    x_map = {lab: i for i, lab in enumerate(labels)}

    d["x"] = d[label_col].map(x_map)

    xj = d["x"].to_numpy(dtype=float) + np.random.uniform(
        -jitter,
        jitter,
        size=len(d),
    )

    y = d[y_col].to_numpy(dtype=float)

    stats = d.groupby(label_col)[y_col].agg(["mean", "std"]).reset_index()

    plt.figure(figsize=(max(6, 1.2 * len(labels)), 4.5))
    ax = plt.gca()

    ax.scatter(xj, y, s=s, alpha=alpha, zorder=2)

    for _, row in stats.iterrows():
        lab = row[label_col]
        mean_val = row["mean"]
        sd_val = row["std"]
        xc = x_map[lab]

        rect = Rectangle(
            (xc - rect_width / 2, mean_val - rect_height / 2),
            rect_width,
            rect_height,
            linewidth=1,
            edgecolor="black",
            facecolor="white",
            zorder=4,
        )

        ax.add_patch(rect)

        if not np.isnan(sd_val):
            y_low = mean_val - sd_val
            y_high = mean_val + sd_val

            ax.plot(
                [xc, xc],
                [y_low, y_high],
                color="black",
                linewidth=1.5,
                zorder=3,
            )

            ax.plot(
                [xc - cap_width / 2, xc + cap_width / 2],
                [y_low, y_low],
                color="black",
                linewidth=1.5,
                zorder=3,
            )

            ax.plot(
                [xc - cap_width / 2, xc + cap_width / 2],
                [y_high, y_high],
                color="black",
                linewidth=1.5,
                zorder=3,
            )

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")

    ax.set_ylabel("Allele frequency (AF)")
    ax.set_xlabel("")
    ax.set_title(title if title else y_col)
    ax.set_ylim(*y_limits)

    plt.tight_layout()

    fname = f"{_safe_filename(y_col)}{filename_suffix}.png"
    outpath = os.path.join(outdir, fname)

    plt.savefig(outpath, dpi=dpi, bbox_inches="tight")

    plt.savefig(
        outpath.replace(".png", ".pdf"),
        dpi=dpi,
        bbox_inches="tight",
    )

    if show_plot:
        plt.show()

    plt.close()


def _sd_filter_per_label(
    df: pd.DataFrame,
    y_col: str,
    label_col: str = "label",
    k: float = 1.5,
    min_n: int = 3,
):
    """Flag outlier cells per label using mean ± k × SD."""
    d = df[[label_col, y_col]].copy()
    d[y_col] = pd.to_numeric(d[y_col], errors="coerce")

    d = d.dropna(subset=[label_col, y_col]).copy()

    if d.empty:
        return d, pd.DataFrame()

    keep = pd.Series(True, index=d.index)
    audit_parts = []

    for lab, sub in d.groupby(label_col):
        idx = sub.index
        x = sub[y_col].to_numpy(dtype=float)

        if len(x) < min_n:
            mu = float(np.mean(x)) if len(x) else np.nan
            sd = float(np.std(x, ddof=1)) if len(x) > 1 else np.nan

            lo, hi = -np.inf, np.inf
            lab_keep = pd.Series(True, index=idx)

        else:
            mu = float(np.mean(x))
            sd = float(np.std(x, ddof=1))

            if not np.isfinite(sd) or sd == 0.0:
                lo, hi = -np.inf, np.inf
                lab_keep = pd.Series(True, index=idx)

            else:
                lo = mu - k * sd
                hi = mu + k * sd

                lab_keep = (
                    (sub[y_col] >= lo)
                    & (sub[y_col] <= hi)
                )

        keep.loc[idx] &= lab_keep

        audit_part = sub.copy()

        audit_part["mu"] = mu
        audit_part["sd"] = sd
        audit_part["lo"] = lo
        audit_part["hi"] = hi
        audit_part["k"] = k

        audit_parts.append(audit_part)

    audit = pd.concat(audit_parts, axis=0)

    audit["status"] = np.where(
        keep.loc[audit.index],
        "kept",
        "removed",
    )

    return d.loc[keep].copy(), audit


# ═══════════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════════

def filter_somatic_variant_outliers(
    Sample,
    df_AF: pd.DataFrame,
    differential_variants,
    outdir: str,
    clone_variant_calls: pd.DataFrame = None,
    variant_column: str = "Variant name",
    eligible_genotypes=("het", "hom"),
    label_col: str = "label",
    dp_threshold: int = 5,
    sd_k: float = 1.25,
    sd_min_n: int = 3,
    jitter: float = 0.20,
    alpha: float = 0.65,
    point_size: float = 20,
    rect_width: float = 0.4,
    rect_height: float = 1.5,
    cap_width: float = 0.2,
    y_limits: tuple = (-1, 101),
    dpi: int = 300,
    save_audit: bool = True,
    save_kept_barcodes: bool = True,
    show_plots: bool = True,
    verbose: bool = True,
    random_seed: int = 0,
):
    """Filter somatic-variant AF outliers within carrier clone cells.

    For every variant, this function first identifies clone labels that carry
    the variant in ``clone_variant_calls`` (Het or Hom by default, controlled
    by ``eligible_genotypes``). It then restricts the DP filter, AF jitter
    plots, per-clone SD filter, audits, and kept barcodes to cells from those
    carrier clones only.

    Calls such as ``WT (...)`` are excluded. Columns in ``clone_variant_calls``
    that are not active labels in ``Sample.dna`` are ignored automatically.
    Variant IDs are matched on their canonical chr:pos:ref/alt key, so a gene
    prefix on either side does not break matching.

    Parameters
    ----------
    Sample : MosaicSample
        Sample object containing the DNA assay.

    df_AF : pd.DataFrame
        Allele-frequency matrix (cells × variants). It must either contain
        ``label_col`` or have rows ordered identically to ``Sample.dna``.

    differential_variants : array-like
        Variant IDs to process.

    outdir : str
        Output directory for plots, audits, and barcode files.

    clone_variant_calls : pd.DataFrame, required
        Clone-level genotype table. Rows must correspond to variant IDs and
        columns must include clone labels, with values such as
        ``WT (0.26%)``, ``Het (50.48%)``, and ``Hom (99.44%)``.

    variant_column : str
        Column in ``clone_variant_calls`` containing variant IDs. If absent,
        the dataframe index is used.

    eligible_genotypes : tuple
        Clone genotype classes that count as carrying the variant. Default
        ``("het", "hom")``. Pass ``("het",)`` for HET-only behaviour.

    label_col : str
        Cell clone-label column in ``df_AF`` (default ``"label"``).

    dp_threshold : int
        A cell passes only if DP is strictly greater than this value.

    sd_k : float
        Number of standard deviations for the per-carrier-clone AF outlier
        filter.

    sd_min_n : int
        Minimum cells per clone to apply the SD filter.

    Returns
    -------
    kept_barcodes : dict[str, pd.Index]
        Barcodes retained after DP and SD filtering for each variant.

    all_audits : pd.DataFrame
        Combined per-cell audit table for all processed variants.

    removed_barcodes_by_variant : dict[str, pd.Index]
        Eligible (carrier-clone) cells that did NOT survive for each variant,
        i.e. DP failures plus SD-flagged AF outliers. WT and non-carrier cells
        never appear here.
    """
    np.random.seed(random_seed)

    def vprint(message):
        if verbose:
            print(message)

    if clone_variant_calls is None:
        raise ValueError(
            "clone_variant_calls is required. Supply the clone-by-variant "
            "genotype table used to identify carrier clone cells per mutation."
        )

    os.makedirs(outdir, exist_ok=True)

    vprint("\n" + "=" * 60)
    vprint("FILTERING SOMATIC VARIANT OUTLIERS")
    vprint("Scope: carrier clone cells (" + "/".join(eligible_genotypes) + "), per variant")
    vprint("=" * 60)

    # ── Build DP matrix ───────────────────────────────────────────────────
    df_DP = Sample.dna.get_attribute(
        "DP",
        constraint="row+col",
    ).copy()

    dp_columns = list(df_DP.columns)

    df_DP.loc[:, label_col] = Sample.dna.get_labels()
    df_DP = df_DP.loc[:, [label_col] + dp_columns]

    # ── Ensure label column in AF matrix ───────────────────────────────────
    df_AF = df_AF.copy()

    if label_col not in df_AF.columns:
        af_columns = list(df_AF.columns)

        df_AF.loc[:, label_col] = Sample.dna.get_labels()
        df_AF = df_AF.loc[:, [label_col] + af_columns]

    # ── Align AF and DP matrices ───────────────────────────────────────────
    common_idx = df_AF.index.intersection(df_DP.index)

    if len(common_idx) == 0:
        raise ValueError("df_AF and df_DP do not share any cell rows.")

    df_AF_aligned = df_AF.loc[common_idx].copy()
    df_DP_aligned = df_DP.loc[common_idx].copy()

    df_AF_aligned, df_DP_aligned = _ensure_barcode_index(
        df_AF=df_AF_aligned,
        df_DP=df_DP_aligned,
        Sample=Sample,
    )

    vprint(f"  Aligned cells: {len(df_AF_aligned)}")

    # ── Restrict to variants available in both AF and DP ───────────────────
    diff_cols = [
        variant
        for variant in differential_variants
        if (
            variant in df_AF_aligned.columns
            and variant in df_DP_aligned.columns
        )
    ]

    if len(diff_cols) == 0:
        raise ValueError(
            "None of the differential_variants are present in both "
            "df_AF and df_DP."
        )

    (
        het_barcodes_by_variant,
        het_clones_by_variant,
        het_clone_audit,
    ) = _get_het_clone_barcodes_by_variant(
        Sample=Sample,
        differential_variants=diff_cols,
        clone_variant_calls=clone_variant_calls,
        variant_column=variant_column,
        eligible_genotypes=eligible_genotypes,
    )

    if save_audit:
        het_clone_audit_path = os.path.join(
            outdir,
            "carrier_clone_eligibility_by_variant.csv",
        )

        het_clone_audit.to_csv(
            het_clone_audit_path,
            index=False,
        )

        vprint(
            f"  Carrier clone eligibility audit: {het_clone_audit_path}"
        )

    vprint(f"  Variants to process: {len(diff_cols)}")
    vprint(f"  DP threshold: >{dp_threshold}")
    vprint(f"  SD filter: ±{sd_k} SD (min_n={sd_min_n})")

    df_AF_filt = df_AF_aligned.loc[
        :,
        [label_col] + diff_cols,
    ].copy()

    # ── Process each variant ──────────────────────────────────────────────
    kept_barcodes = {}
    removed_barcodes_by_variant = {}
    all_audit_list = []
    eligible_barcode_rows = []
    summary_records = []

    for i, var in enumerate(diff_cols, 1):
        var_id = str(var)

        het_clones = het_clones_by_variant.get(var_id, [])
        eligible_barcode_set = het_barcodes_by_variant.get(var_id, set())

        vprint(f"\n  [{i}/{len(diff_cols)}] {var_id}")

        vprint(
            "    Carrier clones (" + "/".join(eligible_genotypes) + "): "
            + ("; ".join(het_clones) if het_clones else "none")
        )

        eligible_mask = df_AF_filt.index.astype(str).isin(
            eligible_barcode_set
        )

        df_variant_scope = df_AF_filt.loc[
            eligible_mask,
            [label_col, var],
        ].copy()

        n_eligible = len(df_variant_scope)

        # Export exact carrier-clone cells considered before DP filtering.
        if n_eligible > 0:
            eligible_export = df_variant_scope[[label_col]].copy()

            eligible_export.insert(
                0,
                "barcode",
                eligible_export.index.astype(str),
            )

            eligible_export.insert(
                0,
                "variant",
                var_id,
            )

            eligible_barcode_rows.append(
                eligible_export.reset_index(drop=True)
            )

        if n_eligible == 0:
            vprint("    Skipped: no barcodes from carrier clones.")

            kept_barcodes[var] = pd.Index([], dtype=object)
            removed_barcodes_by_variant[var] = pd.Index([], dtype=object)

            summary_records.append(
                {
                    "variant": var_id,
                    "status": "skipped_no_carrier_clone_cells",
                    "het_clones": ";".join(het_clones),
                    "n_het_clones": len(het_clones),
                    "n_eligible_cells": 0,
                    "n_dp_pass": 0,
                    "n_dp_fail": 0,
                    "n_sd_kept": 0,
                    "n_sd_removed": 0,
                }
            )

            continue

        # DP filter only among carrier-clone cells for this variant.
        dp_series = pd.to_numeric(
            df_DP_aligned.loc[df_variant_scope.index, var],
            errors="coerce",
        )

        dp_pass_mask = dp_series > dp_threshold

        df_to_plot = df_variant_scope.loc[
            dp_pass_mask,
            [label_col, var],
        ].copy()

        n_dp_pass = int(dp_pass_mask.sum())
        n_dp_fail = int((~dp_pass_mask).sum())

        vprint(
            f"    Carrier-clone cells: {n_eligible}; "
            f"DP filter: {n_dp_pass} pass, {n_dp_fail} fail"
        )

        # Plot after carrier restriction + DP filter, before SD filtering.
        _jitter_plot_by_label(
            df=df_to_plot,
            y_col=var,
            outdir=outdir,
            label_col=label_col,
            jitter=jitter,
            rect_width=rect_width,
            rect_height=rect_height,
            cap_width=cap_width,
            alpha=alpha,
            s=point_size,
            title=(
                f"AF in carrier clones "
                f"(DP>{dp_threshold}, unfiltered): {var_id}"
            ),
            dpi=dpi,
            y_limits=y_limits,
            filename_suffix="__CARRIER_CLONES__UNFILTERED",
            show_plot=show_plots,
        )

        # Per-carrier-clone SD filter.
        df_kept, audit = _sd_filter_per_label(
            df=df_to_plot,
            y_col=var,
            label_col=label_col,
            k=sd_k,
            min_n=sd_min_n,
        )

        n_kept = len(df_kept)
        n_removed = len(df_to_plot) - n_kept

        vprint(f"    SD filter: {n_kept} kept, {n_removed} removed")

        kept_barcodes[var] = pd.Index(df_kept.index.astype(str))

        # Eligible cells that did not survive = DP failures + SD outliers.
        removed_barcodes_by_variant[var] = pd.Index(
            sorted(
                set(df_variant_scope.index.astype(str))
                - set(kept_barcodes[var].astype(str))
            ),
            dtype=object,
        )

        # Save retained barcodes.
        if save_kept_barcodes:
            kept_path = os.path.join(
                outdir,
                (
                    f"{_safe_filename(var_id)}"
                    f"__carrier_clones__kept_barcodes__SD{sd_k}.txt"
                ),
            )

            with open(kept_path, "w") as handle:
                for barcode in kept_barcodes[var]:
                    handle.write(f"{barcode}\n")

        # Save per-variant cell-level audit.
        if not audit.empty:
            audit = audit.copy()

            audit.insert(0, "variant", var_id)
            audit.insert(1, "barcode", audit.index.astype(str))

            audit["carrier_clones"] = ";".join(het_clones)
            audit["filter_scope"] = "carrier_clone_cells_only"

            audit["DP"] = pd.to_numeric(
                df_DP_aligned.loc[audit.index, var],
                errors="coerce",
            ).to_numpy()

            audit["DP_threshold"] = dp_threshold
            audit["DP_status"] = "passed"

            if save_audit:
                audit_path = os.path.join(
                    outdir,
                    (
                        f"{_safe_filename(var_id)}"
                        f"__carrier_clones__sd_filter_audit__k{sd_k}.csv"
                    ),
                )

                audit.to_csv(
                    audit_path,
                    index=False,
                )

            all_audit_list.append(audit)

        # Plot after per-carrier-clone SD filtering.
        _jitter_plot_by_label(
            df=df_kept,
            y_col=var,
            outdir=outdir,
            label_col=label_col,
            jitter=jitter,
            rect_width=rect_width,
            rect_height=rect_height,
            cap_width=cap_width,
            alpha=alpha,
            s=point_size,
            title=(
                f"AF in carrier clones "
                f"(DP>{dp_threshold}, ±{sd_k} SD): {var_id}"
            ),
            dpi=dpi,
            y_limits=y_limits,
            filename_suffix=f"__CARRIER_CLONES__SD{sd_k}",
            show_plot=show_plots,
        )

        summary_records.append(
            {
                "variant": var_id,
                "status": "processed",
                "het_clones": ";".join(het_clones),
                "n_het_clones": len(het_clones),
                "n_eligible_cells": n_eligible,
                "n_dp_pass": n_dp_pass,
                "n_dp_fail": n_dp_fail,
                "n_sd_kept": n_kept,
                "n_sd_removed": n_removed,
            }
        )

    # ── Save carrier-clone barcode scope ───────────────────────────────────
    if eligible_barcode_rows and save_audit:
        eligible_barcodes_df = pd.concat(
            eligible_barcode_rows,
            axis=0,
            ignore_index=True,
        )

        eligible_barcodes_path = os.path.join(
            outdir,
            "carrier_clone_eligible_barcodes_by_variant.csv",
        )

        eligible_barcodes_df.to_csv(
            eligible_barcodes_path,
            index=False,
        )

        vprint(
            f"\n  Eligible carrier-clone barcodes: {eligible_barcodes_path}"
        )

    # ── Combined audit and summary ─────────────────────────────────────────
    if all_audit_list:
        all_audits = pd.concat(
            all_audit_list,
            axis=0,
            ignore_index=True,
        )
    else:
        all_audits = pd.DataFrame(
            columns=[
                "variant",
                "barcode",
                label_col,
                "status",
                "carrier_clones",
                "filter_scope",
                "DP",
                "DP_threshold",
                "DP_status",
            ]
        )

    summary_df = pd.DataFrame(summary_records)

    if save_audit:
        combined_audit_path = os.path.join(
            outdir,
            f"sd_filter_audit_ALL__carrier_clones__k{sd_k}.csv",
        )

        all_audits.to_csv(
            combined_audit_path,
            index=False,
        )

        summary_path = os.path.join(
            outdir,
            f"somatic_variant_filter_summary__carrier_clones__k{sd_k}.csv",
        )

        summary_df.to_csv(
            summary_path,
            index=False,
        )

        vprint(f"  Combined audit saved: {combined_audit_path}")
        vprint(f"  Filter summary saved: {summary_path}")

    # ── Summary ───────────────────────────────────────────────────────────
    vprint("\n" + "=" * 60)
    vprint("SUMMARY")
    vprint("=" * 60)

    for _, row in summary_df.iterrows():
        vprint(
            f"  {row['variant']}: {row['n_sd_kept']} cells kept "
            f"from {row['n_eligible_cells']} carrier-clone cells "
            f"({row['status']})"
        )

    vprint(f"\n  Output directory: {outdir}")
    vprint("=" * 60 + "\n")

    return kept_barcodes, all_audits, removed_barcodes_by_variant

def _get_het_clone_barcodes_by_variant(
    Sample,
    differential_variants,
    clone_variant_calls: pd.DataFrame,
    variant_column: str = "Variant name",
    eligible_genotypes=("het", "hom"),
):
    """Identify clone labels and barcodes eligible for each variant.

    A clone is eligible for a variant if its call matches one of
    ``eligible_genotypes`` (Het and Hom by default). Only columns in
    ``clone_variant_calls`` that are active labels in ``Sample.dna`` are
    considered clone columns, so table-only summary columns such as
    ``Missing GT clones (3)`` and ``Small subclones (7)`` are excluded
    automatically. Variants are matched on their canonical chr:pos:ref/alt key.
    """
    calls = _prepare_clone_variant_calls(
        clone_variant_calls=clone_variant_calls,
        variant_column=variant_column,
    )

    sample_clone_labels = {
        str(label)
        for label in Sample.dna.get_labels()
        if pd.notna(label) and str(label).strip() != ""
    }

    clone_columns = [
        column
        for column in calls.columns
        if str(column) in sample_clone_labels
    ]

    if len(clone_columns) == 0:
        raise ValueError(
            "No clone columns in clone_variant_calls match labels in "
            "Sample.dna.get_labels(). Ensure clone-table column names match "
            "the labels stored in Sample.dna."
        )

    clone_barcodes = {}

    for column in clone_columns:
        clone_label = str(column)

        clone_barcodes[clone_label] = {
            str(barcode)
            for barcode in Sample.dna.barcodes({clone_label})
        }

    eligible_barcodes_by_variant = {}
    het_clones_by_variant = {}
    audit_records = []

    for variant in map(str, differential_variants):
        key = _canonical_variant_key(variant)

        if key not in calls.index:
            eligible_barcodes_by_variant[variant] = set()
            het_clones_by_variant[variant] = []

            audit_records.append(
                {
                    "variant": variant,
                    "status": "missing_from_clone_variant_calls",
                    "het_clones": "",
                    "n_het_clones": 0,
                    "n_het_clone_barcodes": 0,
                }
            )

            continue

        variant_calls = calls.loc[key]

        # Handle duplicate variant rows safely:
        # a clone is eligible if any matching row carries the variant.
        if isinstance(variant_calls, pd.DataFrame):
            is_het_by_clone = {
                column: variant_calls[column].map(
                    lambda x: _clone_call_matches(x, eligible_genotypes)
                ).any()
                for column in clone_columns
            }
        else:
            is_het_by_clone = {
                column: _clone_call_matches(variant_calls[column], eligible_genotypes)
                for column in clone_columns
            }

        het_clones = [
            str(column)
            for column in clone_columns
            if is_het_by_clone[column]
        ]

        eligible_barcodes = set()

        for clone_label in het_clones:
            eligible_barcodes.update(clone_barcodes[clone_label])

        eligible_barcodes_by_variant[variant] = eligible_barcodes
        het_clones_by_variant[variant] = het_clones

        audit_records.append(
            {
                "variant": variant,
                "status": (
                    "ok"
                    if len(eligible_barcodes) > 0
                    else "no_carrier_clone_barcodes"
                ),
                "het_clones": ";".join(het_clones),
                "n_het_clones": len(het_clones),
                "n_het_clone_barcodes": len(eligible_barcodes),
            }
        )

    return (
        eligible_barcodes_by_variant,
        het_clones_by_variant,
        pd.DataFrame(audit_records),
    )