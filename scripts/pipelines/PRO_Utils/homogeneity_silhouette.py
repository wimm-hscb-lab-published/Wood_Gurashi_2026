"""Per-label silhouette QC for protein clusters / cell-type labels.

Main entry point
----------------
plot_protein_homogeneity_silhouette
    Per-cell silhouette scores on (optionally z-scored) protein features,
    summarised per label, with a bar plot and a box + strip plot.
"""

from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import homogeneity_score, silhouette_samples
from sklearn.preprocessing import LabelEncoder, StandardScaler


def plot_protein_homogeneity_silhouette(
    df_PRO: pd.DataFrame,
    *,
    label_col: str = "label",
    scale: bool = True,
    figsize: tuple = (7, 4),
    drop_duplicate_columns_after_str_cast: bool = True,
    save_csv: bool = False,
    csv_path: str = None,
    show_plot: bool = True,
    verbose: bool = True,
) -> tuple[pd.DataFrame, float]:
    """Compute per-label silhouette scores on protein data and plot them.

    Two matplotlib figures are drawn: (1) a "homogeneity-like" bar per label
    (mean silhouette clipped at 0) and (2) the per-cell silhouette
    distribution per label (box plot without fliers plus a strip plot).

    Parameters
    ----------
    df_PRO : pd.DataFrame
        Cells × features table. Must contain ``label_col``; every other column
        is treated as a feature.
    label_col : str
        Column of cell-type / cluster labels (default ``'label'``). Labels are
        cast to ``str``.
    scale : bool
        Z-score features with ``StandardScaler`` before computing silhouette
        (default True).
    figsize : tuple
        Size of each figure (default ``(7, 4)``).
    drop_duplicate_columns_after_str_cast : bool
        After casting feature names to ``str``, keep only the first of any
        duplicated names (default True).
    save_csv : bool
        Export ``label, mean, std, count`` per label to ``csv_path``
        (default False).
    csv_path : str, optional
        CSV destination; required when ``save_csv`` is True. Its parent
        directory is created.
    show_plot : bool
        Call ``plt.show()`` after each figure (default True). Figures are not
        closed when False.
    verbose : bool
        Print progress and the per-label table (default True).

    Returns
    -------
    per_label : pd.DataFrame
        Columns ``label, mean, median, std, count, homogeneity_like``, sorted
        by ``mean`` descending.
    global_homogeneity : float
        ``homogeneity_score(y, y)`` — always 1.0; kept for interface
        compatibility.

    Raises
    ------
    KeyError
        If ``label_col`` is not a column of ``df_PRO``.
    ValueError
        If there are fewer than 2 unique labels or 2 rows, or if
        ``save_csv=True`` without ``csv_path``.

    Notes
    -----
    Features are coerced with ``pd.to_numeric(errors='coerce')`` and NaNs
    filled with 0 before scaling. Silhouette uses sklearn's default Euclidean
    metric. The CSV is written after both figures have been drawn, and
    ``os.makedirs(os.path.dirname(csv_path))`` fails for a bare filename
    (empty dirname).

    Examples
    --------
    >>> per_label, gh = plot_protein_homogeneity_silhouette(
    ...     df_PRO, label_col="label", scale=True
    ... )
    """
    def vprint(msg):
        if verbose:
            print(msg)
            sys.stdout.flush()

    # ── Validate inputs ──
    if label_col not in df_PRO.columns:
        raise KeyError(f"df_PRO must contain '{label_col}'")

    # ── Prepare feature matrix ──
    X_df = df_PRO.drop(columns=[label_col]).copy()

    # sklearn requires uniform feature-name dtype; force to str
    X_df.columns = X_df.columns.map(str)

    if drop_duplicate_columns_after_str_cast:
        X_df = X_df.loc[:, ~X_df.columns.duplicated(keep="first")]

    X_df = X_df.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    labels = df_PRO[label_col].astype(str)

    if labels.nunique() < 2:
        raise ValueError("Silhouette requires at least 2 unique labels.")
    if len(labels) < 2:
        raise ValueError("Silhouette requires at least 2 rows.")

    if scale:
        X = StandardScaler().fit_transform(X_df.to_numpy())
    else:
        X = X_df.to_numpy()

    le = LabelEncoder()
    y = le.fit_transform(labels)

    # ── Scores ──
    global_homogeneity = homogeneity_score(y, y)

    vprint(f"  Computing silhouette for {len(labels)} cells, "
           f"{labels.nunique()} labels …")

    sil_vals = silhouette_samples(X, y)

    sil_df = pd.DataFrame({
        "label": labels.values,
        "silhouette": sil_vals,
    })

    # ── Per-label summaries ──
    per_label = (
        sil_df
        .groupby("label")["silhouette"]
        .agg(["mean", "median", "std", "count"])
        .reset_index()
        .sort_values("mean", ascending=False)
    )
    per_label["homogeneity_like"] = per_label["mean"].clip(lower=0)

    vprint("  Per-label silhouette stats:")
    vprint(per_label[["label", "mean", "std", "count"]].to_string(index=False))

    # ── Figure 1 – Homogeneity per label ──
    fig1, ax1 = plt.subplots(figsize=figsize)
    sns.barplot(data=per_label, x="label", y="homogeneity_like",
                color="steelblue", ax=ax1)
    ax1.axhline(0, color="black", lw=1)
    ax1.set_ylim(0, 1)
    ax1.set_title("Protein Homogeneity per Label")
    ax1.set_ylabel("Homogeneity (mean silhouette ≥ 0)")
    ax1.set_xlabel("Label")
    plt.setp(ax1.get_xticklabels(), rotation=45, ha="right")
    fig1.tight_layout()
    if show_plot:
        plt.show()

    # ── Figure 2 – Silhouette per label ──
    fig2, ax2 = plt.subplots(figsize=figsize)
    sns.boxplot(data=sil_df, x="label", y="silhouette",
                showfliers=False, ax=ax2)
    sns.stripplot(data=sil_df, x="label", y="silhouette",
                  color="black", alpha=0.3, size=2, ax=ax2)
    ax2.axhline(0, color="red", linestyle="--", lw=1)
    ax2.set_title("Protein Silhouette per Label")
    ax2.set_ylabel("Silhouette")
    ax2.set_xlabel("Label")
    plt.setp(ax2.get_xticklabels(), rotation=45, ha="right")
    fig2.tight_layout()
    if show_plot:
        plt.show()

    # ── Optional CSV export ──
    if save_csv:
        if csv_path is None:
            raise ValueError("csv_path must be provided when save_csv=True")
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        export_df = per_label[["label", "mean", "std", "count"]]
        export_df.to_csv(csv_path, index=False)
        vprint(f"  ✓ Saved CSV: {csv_path}")

    return per_label, global_homogeneity