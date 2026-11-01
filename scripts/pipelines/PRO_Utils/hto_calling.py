"""PRO_utilities – single-hashtag HTO present/absent calling.

Scores each hashtag in a protein assay *independently* as a present/absent QC
call: log1p raw counts -> 2-component Gaussian mixture (background vs stained)
-> per-cell posterior assignment, with a guard that refuses to threshold tags
that aren't genuinely bimodal.

This is NOT demultiplexing. A single tag carries no cross-tag information, so
doublets cannot be identified from one hashtag, and multi-tag input is scored
per-tag rather than by cross-tag argmax. For donor demultiplexing from
competing tags use a dedicated demux method instead.

Main entry points
-----------------
call_hto_present_absent
    Score a list of hashtags on an assay; returns AnnData, summary, results.
call_single_hashtag
    Fit and call one hashtag (a ``pd.Series`` of raw counts).
plot_hashtag_distribution
    Histogram of log1p counts with the fitted mixture and threshold.
HTOResult
    Dataclass holding one tag's calls and fit.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import norm
from sklearn.mixture import GaussianMixture

__all__ = [
    "HTOResult",
    "call_single_hashtag",
    "call_hto_present_absent",
    "plot_hashtag_distribution",
]


@dataclass
class HTOResult:
    """Per-hashtag calling result.

    Attributes
    ----------
    hashtag : str
        Column name of the scored hashtag.
    call : np.ndarray
        Per-cell label, one of "Positive" / "Negative" / "Uncertain".
    p_pos : np.ndarray
        Per-cell posterior probability of the signal (higher-mean) component.
    threshold : float | None
        Raw count at which the posterior crosses the cutoff (reporting only);
        ``None`` when the tag is unimodal.
    bimodal : bool
        Whether a genuine positive/negative split was found.
    means : np.ndarray
        Sorted component means in log space ``[negative, positive]``.
    note : str
        Free-text diagnostic (e.g. unimodal warning).
    gmm : GaussianMixture
        The fitted 2-component model (kept for plotting; excluded from repr).
    """

    hashtag: str
    call: np.ndarray
    p_pos: np.ndarray
    threshold: float | None
    bimodal: bool
    means: np.ndarray
    note: str = ""
    gmm: object = field(default=None, repr=False)


def _extract_hto_counts(assay, hashtags: list[str]) -> pd.DataFrame:
    """Return raw counts for ``hashtags`` as a (cells × tags) DataFrame copy.

    Tries ``assay.get_attribute('read_counts', constraint='row+col')``; on any
    exception falls back to ``assay.to_df()`` or, failing that attribute, an
    AnnData-style ``X`` / ``obs_names`` / ``var_names`` frame. Raises
    ``KeyError`` listing missing hashtags and the available columns.
    """
    try:
        df = assay.get_attribute("read_counts", constraint="row+col")
    except Exception:
        if hasattr(assay, "to_df"):
            df = assay.to_df()
        else:
            df = pd.DataFrame(assay.X, index=assay.obs_names, columns=assay.var_names)

    missing = [h for h in hashtags if h not in df.columns]
    if missing:
        raise KeyError(f"Hashtags not found in assay: {missing}. Available: {list(df.columns)}")
    return df[hashtags].copy()


def call_single_hashtag(
    counts: pd.Series,
    posterior_cutoff: float = 0.5,
    uncertain_band: tuple[float, float] | None = (0.05, 0.95),
    min_mean_separation: float = 1.0,
    n_init: int = 10,
    random_state: int = 42,
) -> HTOResult:
    """Call one hashtag present/absent via a 2-component GMM on log1p counts.

    Cells are classified by their own posterior rather than by collapsing to a
    single count cutoff: with unequal component variances the posterior boundary
    need not be a clean upper interval, so a count threshold can disagree with
    the model. The fit uses all cells (zeros included — the cleanest negatives).

    If the tag is not genuinely bimodal (BIC favours a single component, or the
    two component means are closer than ``min_mean_separation`` log units), no
    split is trusted: every cell is called "Negative" and ``bimodal`` is False.

    Parameters
    ----------
    counts : pd.Series
        Raw counts for one hashtag, indexed by barcode (``.name`` is the tag).
    posterior_cutoff : float
        Posterior at/above which a cell is Positive.
    uncertain_band : tuple[float, float] | None
        Posteriors strictly inside this band are relabelled "Uncertain".
        Pass ``None`` to force a hard Positive/Negative split.
    min_mean_separation : float
        Minimum gap (log units) between component means to accept a split.
    n_init : int
        GMM restarts (the default of 1 is fragile on this problem).
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    HTOResult
        ``call`` is an object array of "Positive" / "Negative" / "Uncertain";
        ``p_pos`` is all zeros when the tag is unimodal.

    Notes
    -----
    Bimodality requires both ``BIC(2 components) < BIC(1 component)`` and a
    gap of at least ``min_mean_separation`` between the sorted component
    means. Cells are Positive when ``p_pos >= posterior_cutoff``; cells with
    ``lo < p_pos < hi`` (``uncertain_band``) are then overwritten as
    "Uncertain". The reported ``threshold`` is ``expm1`` of the first point on
    a 2000-step log1p grid where the posterior goes from below to at/above the
    cutoff, or None if it never crosses upward.
    """
    name = counts.name
    vals = counts.to_numpy(dtype=float)
    logv = np.log1p(vals).reshape(-1, 1)
    n = len(vals)

    g1 = GaussianMixture(1, random_state=random_state).fit(logv)
    g2 = GaussianMixture(2, n_init=n_init, random_state=random_state).fit(logv)

    means_sorted = np.sort(g2.means_.flatten())
    bimodal = (g2.bic(logv) < g1.bic(logv)) and (
        (means_sorted[1] - means_sorted[0]) >= min_mean_separation
    )

    if not bimodal:
        return HTOResult(
            hashtag=name,
            call=np.full(n, "Negative", dtype=object),
            p_pos=np.zeros(n),
            threshold=None,
            bimodal=False,
            means=means_sorted,
            note="unimodal: no positive/negative split found; all cells called Negative",
            gmm=g2,
        )

    sig = int(np.argmax(g2.means_.flatten()))
    p_pos = g2.predict_proba(logv)[:, sig]

    call = np.where(p_pos >= posterior_cutoff, "Positive", "Negative").astype(object)
    if uncertain_band is not None:
        lo, hi = uncertain_band
        call[(p_pos > lo) & (p_pos < hi)] = "Uncertain"

    # Reporting-only threshold: raw count where the posterior first crosses the cutoff.
    grid = np.linspace(logv.min(), logv.max(), 2000)
    pg = g2.predict_proba(grid.reshape(-1, 1))[:, sig]
    crossings = np.where(np.diff((pg >= posterior_cutoff).astype(int)) == 1)[0]
    threshold = float(np.expm1(grid[crossings[0]])) if len(crossings) else None

    return HTOResult(name, call, p_pos, threshold, True, means_sorted, gmm=g2)


def call_hto_present_absent(
    assay,
    hashtags: list[str],
    plot: bool = True,
    verbose: bool = True,
    **call_kwargs,
) -> tuple[sc.AnnData, pd.DataFrame, dict[str, HTOResult]]:
    """Score every hashtag in ``hashtags`` independently.

    Each tag is called with :func:`call_single_hashtag`; no cross-tag
    information is used, so this is a present/absent QC, not a demultiplexer.

    Parameters
    ----------
    assay
        Protein assay exposing ``get_attribute('read_counts', constraint='row+col')``
        (Tapestri-style) or an AnnData-like object.
    hashtags : list[str]
        Hashtag column names to score.
    plot : bool
        Show a per-tag log1p histogram with the fitted mixture and threshold.
    verbose : bool
        Print a one-line summary per tag.
    **call_kwargs
        Forwarded to :func:`call_single_hashtag`.

    Returns
    -------
    adata : sc.AnnData
        HTO counts with per-tag calls in ``.obs`` (``f"{tag}_p_pos"``,
        ``f"{tag}_call"``) plus a convenience ``"n_positive"`` column (count of
        tags called Positive per cell — NOT a demux assignment).
    summary : pd.DataFrame
        One row per hashtag with positive fraction, threshold, means, flags.
    results : dict[str, HTOResult]
        Raw per-tag results.

    Raises
    ------
    KeyError
        If any hashtag is not a column of the assay's counts.

    Notes
    -----
    ``adata.X`` is the raw counts as float32; raw counts per tag are also
    copied into ``adata.obs[tag]``. The verbose line includes the threshold
    only when it is truthy (a threshold of exactly 0.0 is not printed).
    """
    df_htos = _extract_hto_counts(assay, hashtags)

    adata = sc.AnnData(df_htos.astype(np.float32))
    adata.obs_names = df_htos.index.astype(str)
    adata.var_names = df_htos.columns.astype(str)
    for h in hashtags:
        adata.obs[h] = df_htos[h].to_numpy()

    results: dict[str, HTOResult] = {}
    summary_rows = []

    for h in hashtags:
        res = call_single_hashtag(df_htos[h], **call_kwargs)
        results[h] = res

        adata.obs[f"{h}_p_pos"] = res.p_pos
        adata.obs[f"{h}_call"] = pd.Categorical(
            res.call, categories=["Negative", "Uncertain", "Positive"]
        )

        n_pos = int((res.call == "Positive").sum())
        n_unc = int((res.call == "Uncertain").sum())
        summary_rows.append(
            {
                "hashtag": h,
                "bimodal": res.bimodal,
                "n_positive": n_pos,
                "pct_positive": round(100 * n_pos / adata.n_obs, 2),
                "n_uncertain": n_unc,
                "threshold_count": None if res.threshold is None else round(res.threshold, 1),
                "mean_neg_log": round(float(res.means[0]), 3),
                "mean_pos_log": round(float(res.means[-1]), 3),
                "note": res.note,
            }
        )

        if verbose:
            msg = (
                f"[{h}] bimodal={res.bimodal}  positive={n_pos} "
                f"({100 * n_pos / adata.n_obs:.2f}%)  uncertain={n_unc}"
            )
            if res.threshold:
                msg += f"  threshold~{res.threshold:.0f}"
            if res.note:
                msg += f"  -- {res.note}"
            print(msg)

        if plot:
            plot_hashtag_distribution(df_htos[h], res)

    call_cols = [f"{h}_call" for h in hashtags]
    adata.obs["n_positive"] = (adata.obs[call_cols] == "Positive").sum(axis=1).astype(int)

    summary = pd.DataFrame(summary_rows).set_index("hashtag")
    return adata, summary, results


def plot_hashtag_distribution(counts: pd.Series, result: HTOResult, ax=None):
    """Plot one tag's log1p count histogram with the fitted mixture and threshold.

    Parameters
    ----------
    counts : pd.Series
        Raw counts for the tag.
    result : HTOResult
        Result from :func:`call_single_hashtag` for the same tag.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; a new 8 × 5 figure is created when None.

    Returns
    -------
    matplotlib.axes.Axes
        Histogram (100 bins, density), mixture density and dotted weighted
        components when ``result.gmm`` is set, and a dashed line at
        ``log1p(threshold)`` when a threshold exists. Unimodal tags get
        "[UNIMODAL — not thresholded]" in the title. ``plt.show`` is not
        called.
    """
    logv = np.log1p(counts.to_numpy(dtype=float))
    xs = np.linspace(logv.min(), logv.max(), 500)

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    ax.hist(logv, bins=100, density=True, alpha=0.5)

    if result.gmm is not None:
        gmm = result.gmm
        ax.plot(xs, np.exp(gmm.score_samples(xs.reshape(-1, 1))), linewidth=2, label="mixture")
        for w, m, c in zip(gmm.weights_, gmm.means_.flatten(), gmm.covariances_.flatten()):
            ax.plot(xs, w * norm.pdf(xs, m, np.sqrt(c)), linewidth=1.5, linestyle=":")

    if result.threshold is not None:
        ax.axvline(
            np.log1p(result.threshold),
            linestyle="--",
            linewidth=2,
            label=f"threshold ~{result.threshold:.0f}",
        )

    title = result.hashtag + ("" if result.bimodal else "  [UNIMODAL — not thresholded]")
    ax.set_title(title, fontsize=15)
    ax.set_xlabel("log1p(count)", fontsize=13)
    ax.set_ylabel("Density", fontsize=13)
    ax.legend()
    return ax
