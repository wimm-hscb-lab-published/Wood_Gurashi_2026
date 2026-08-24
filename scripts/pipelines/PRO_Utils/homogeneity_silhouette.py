import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import homogeneity_score, silhouette_samples


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
):
    """
    Compute per-label silhouette scores on protein data and plot two figures:

      1) Homogeneity-like score per label (mean silhouette clipped at >= 0)
      2) Silhouette per label (box + strip)

    Parameters
    ----------
    df_PRO : pd.DataFrame
        Protein expression dataframe with cells as rows. Must contain a
        column named *label_col*; all other columns are treated as features.
    label_col : str
        Column containing cell-type / cluster labels (default: ``'label'``).
    scale : bool
        Whether to z-score features before computing silhouette (default: True).
    figsize : tuple
        Size of each matplotlib figure (default: ``(7, 4)``).
    drop_duplicate_columns_after_str_cast : bool
        Safety flag – after casting column names to ``str``, drop any
        duplicate columns that may arise (default: True).
    save_csv : bool
        If True, export per-label stats to *csv_path* (default: False).
    csv_path : str, optional
        Destination path for the exported CSV.  Required when *save_csv* is
        True.
    show_plot : bool
        Whether to call ``plt.show()`` (default: True).  Set to False when
        running non-interactively.
    verbose : bool
        Print progress messages (default: True).

    Returns
    -------
    per_label : pd.DataFrame
        Label-level silhouette summary (mean, median, std, count,
        homogeneity_like).
    global_homogeneity : float
        ``homogeneity_score(y, y)`` – always 1.0; included for interface
        completeness.

    Raises
    ------
    KeyError
        If *label_col* is not present in *df_PRO*.
    ValueError
        If fewer than 2 labels or 2 rows are present.

    Example
    -------
    >>> from PRO_pipeline import plot_protein_homogeneity_silhouette
    >>> per_label, gh = plot_protein_homogeneity_silhouette(
    ...     df_PRO, label_col="label", scale=True
    ... )
    """
    import sys

    def vprint(msg):
        if verbose:
            print(msg)
            sys.stdout.flush()

    # ------------------------------------------------------------------
    # Validate inputs
    # ------------------------------------------------------------------
    if label_col not in df_PRO.columns:
        raise KeyError(f"df_PRO must contain '{label_col}'")

    # ------------------------------------------------------------------
    # Prepare feature matrix
    # ------------------------------------------------------------------
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

    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(labels)

    # ------------------------------------------------------------------
    # Scores
    # ------------------------------------------------------------------
    global_homogeneity = homogeneity_score(y, y)

    vprint(f"  Computing silhouette for {len(labels)} cells, "
           f"{labels.nunique()} labels …")

    sil_vals = silhouette_samples(X, y)

    sil_df = pd.DataFrame({
        "label": labels.values,
        "silhouette": sil_vals,
    })

    # ------------------------------------------------------------------
    # Per-label summaries
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Figure 1 – Homogeneity per label
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Figure 2 – Silhouette per label
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Optional CSV export
    # ------------------------------------------------------------------
    if save_csv:
        if csv_path is None:
            raise ValueError("csv_path must be provided when save_csv=True")
        import os
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        export_df = per_label[["label", "mean", "std", "count"]]
        export_df.to_csv(csv_path, index=False)
        vprint(f"  ✓ Saved CSV: {csv_path}")

    return per_label, global_homogeneity