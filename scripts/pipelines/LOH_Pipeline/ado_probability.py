"""
ADO-combination probability scoring for LOH clones.

This module asks whether an observed LOH-like subclone could plausibly arise
from independent amplicon-level ADO events within the original/base clone.

Main entry point: ``score_ado_combination``. Helpers:
``normalise_amplicon_ado_rates``, ``poisson_binomial_tail_at_least``,
``binomial_tail_at_least``.

The default pipeline no longer runs this ADO-combination QC; the
``ado_probability_*`` fields of ``LOHConfig`` are retained only for
configuration compatibility.

Example
-------
If a LOH block spans 3 independent amplicons, each with baseline ADO rate 5%,
the single-cell probability of matching all 3 by chance is 0.05**3 = 0.000125.
For an original clone of size X, the probability of seeing >= k matching cells is
Binomial.sf(k-1, X, 0.000125).
"""
from __future__ import annotations

from math import comb, prod
from typing import Any, Iterable

import numpy as np
import pandas as pd

try:
    from scipy.stats import binom
except Exception:  # pragma: no cover
    binom = None


def _variant_pos(v: str) -> int:
    """Genomic position from ``chrN:POS[-END]:...``; 10**18 if unparsable (currently unused)."""
    try:
        return int(str(v).split(":")[1].split("-")[0])
    except Exception:
        return 10**18


def normalise_amplicon_ado_rates(
    amplicon_ado_rates: Any,
    *,
    rate_col: str = "smooth_ado_any_rate",
    amplicon_col: str = "amplicon_id",
    default_rate: float | None = None,
) -> dict[str, float]:
    """
    Convert an amplicon ADO table/mapping to {amplicon_id: probability}.

    Accepted inputs:
      - dict {amplicon_id: rate}
      - pandas Series indexed by amplicon_id
      - pandas DataFrame with amplicon_col and rate_col

    Parameters
    ----------
    amplicon_ado_rates : dict, pd.Series, pd.DataFrame or None
        Rates per amplicon. None returns ``{}``.
    rate_col : str, default "smooth_ado_any_rate"
        DataFrame rate column. If missing, the first present of
        ``smooth_ado_any_rate``, ``raw_ado_any_rate``, ``smooth_ado_low_rate``,
        ``smooth_ado_high_rate``, ``ado_rate`` is used.
    amplicon_col : str, default "amplicon_id"
        DataFrame amplicon column. If missing and the index is named, the
        index is reset and its first column is used.
    default_rate : float or None
        Parsed but not used in the result (see Notes).

    Returns
    -------
    dict[str, float]
        Amplicon id → probability. Non-numeric and non-finite rates are
        skipped; values > 1 are divided by 100; results are clipped to [0, 1].

    Raises
    ------
    ValueError
        If a DataFrame has no usable rate column, or no amplicon column and an
        unnamed index.
    TypeError
        If the input type is not supported.

    Notes
    -----
    *default_rate* is normalised locally but never applied to the returned
    mapping; callers fill missing amplicons themselves.
    """
    if amplicon_ado_rates is None:
        return {}

    rates: dict[str, float] = {}

    if isinstance(amplicon_ado_rates, dict):
        items = amplicon_ado_rates.items()
    elif isinstance(amplicon_ado_rates, pd.Series):
        items = amplicon_ado_rates.items()
    elif isinstance(amplicon_ado_rates, pd.DataFrame):
        df = amplicon_ado_rates.copy()
        if rate_col not in df.columns:
            candidates = [
                "smooth_ado_any_rate", "raw_ado_any_rate",
                "smooth_ado_low_rate", "smooth_ado_high_rate", "ado_rate",
            ]
            found = [c for c in candidates if c in df.columns]
            if not found:
                raise ValueError(
                    f"ADO rate table is missing rate_col={rate_col!r}; columns={list(df.columns)}"
                )
            rate_col = found[0]
        if amplicon_col not in df.columns:
            if df.index.name is not None:
                df = df.reset_index()
                amplicon_col = df.columns[0]
            else:
                raise ValueError(
                    f"ADO rate table is missing amplicon_col={amplicon_col!r}; columns={list(df.columns)}"
                )
        items = zip(df[amplicon_col].astype(str), df[rate_col])
    else:
        raise TypeError(
            "amplicon_ado_rates must be dict, Series, DataFrame, or None."
        )

    for amp, val in items:
        try:
            p = float(val)
        except Exception:
            continue
        if not np.isfinite(p):
            continue
        # Accept percentages accidentally supplied as 0-100.
        if p > 1.0:
            p = p / 100.0
        p = min(max(p, 0.0), 1.0)
        rates[str(amp)] = p

    if default_rate is not None:
        try:
            default_rate = float(default_rate)
            if default_rate > 1.0:
                default_rate /= 100.0
        except Exception:
            default_rate = None

    return rates


def poisson_binomial_tail_at_least(probs: Iterable[float], k: int) -> float:
    """P(sum of independent Bernoulli(probs) >= k), by dynamic programming.

    Parameters
    ----------
    probs : iterable of float
        Success probabilities; non-finite values are dropped, the rest are
        clipped to [0, 1].
    k : int
        Minimum number of successes.

    Returns
    -------
    float
        NaN if no probabilities remain, 1.0 if ``k <= 0``, 0.0 if ``k`` exceeds
        the number of trials.
    """
    probs = [min(max(float(p), 0.0), 1.0) for p in probs if np.isfinite(float(p))]
    m = len(probs)
    if m == 0:
        return np.nan
    k = int(k)
    if k <= 0:
        return 1.0
    if k > m:
        return 0.0

    # dp[j] = probability of exactly j successes so far.
    dp = np.zeros(m + 1, dtype=float)
    dp[0] = 1.0
    for p in probs:
        nxt = dp.copy() * (1.0 - p)
        nxt[1:] += dp[:-1] * p
        dp = nxt
    return float(dp[k:].sum())


def binomial_tail_at_least(n: int, k: int, p: float) -> float:
    """P(Binomial(n, p) >= k).

    Parameters
    ----------
    n : int
        Trials.
    k : int
        Minimum successes.
    p : float
        Success probability (clipped to [0, 1]).

    Returns
    -------
    float
        1.0 if ``k <= 0``; 0.0 if ``n <= 0`` or ``k > n``; otherwise
        ``scipy.stats.binom.sf(k - 1, n, p)``, or an exact sum when SciPy is
        unavailable.
    """
    n = int(n)
    k = int(k)
    p = min(max(float(p), 0.0), 1.0)
    if k <= 0:
        return 1.0
    if n <= 0 or k > n:
        return 0.0
    if binom is not None:
        return float(binom.sf(k - 1, n, p))

    # Exact-sum fallback when SciPy is unavailable; fine for modest n.
    return float(sum(comb(n, i) * (p ** i) * ((1.0 - p) ** (n - i)) for i in range(k, n + 1)))


def score_ado_combination(
    *,
    block_amplicons: Iterable[str],
    amplicon_ado_rates: Any,
    original_clone_size: int,
    observed_subclone_size: int,
    required_amplicons: int | None = None,
    rate_col: str = "smooth_ado_any_rate",
    amplicon_col: str = "amplicon_id",
    default_amplicon_ado_rate: float = 0.01,
    pvalue_threshold: float = 0.05,
) -> dict[str, Any]:
    """
    Score whether a LOH block is expected by chance from amplicon ADO.

    Parameters
    ----------
    block_amplicons : iterable of str
        Amplicons in the LOH block; de-duplicated in order, and ``""``,
        ``"nan"``, ``"None"`` are dropped.
    amplicon_ado_rates : dict, pd.Series, pd.DataFrame or None
        Per-amplicon ADO rates (see ``normalise_amplicon_ado_rates``).
    original_clone_size : int
        Cells in the original/base clone (binomial trials).
    observed_subclone_size : int
        Observed LOH-like cells.
    required_amplicons : int or None
        Amplicons that must drop out together in one cell; None = all.
    rate_col, amplicon_col : str
        Forwarded to ``normalise_amplicon_ado_rates``.
    default_amplicon_ado_rate : float, default 0.01
        Rate used for amplicons missing from the table.
    pvalue_threshold : float, default 0.05
        ``p_at_least_observed >= threshold`` → ``ADO_COMPATIBLE``.

    Returns
    -------
    dict
        ``n_block_amplicons``, ``n_required_amplicons``,
        ``single_cell_ado_probability`` (product of rates when all amplicons
        are required, else a Poisson-binomial tail), ``expected_ado_cells``
        (``original_clone_size * p``), ``p_at_least_observed``
        (P(>= observed cells)), ``ado_enrichment`` (observed / expected, inf if
        expected is 0/NaN), ``classification`` (``ADO_NOT_SCORED``,
        ``ADO_COMPATIBLE`` or ``ADO_UNLIKELY``), ``block_amplicons`` and
        ``missing_amplicon_rates``.
    """
    amps = [str(a) for a in dict.fromkeys(block_amplicons) if str(a) not in {"", "nan", "None"}]
    rates = normalise_amplicon_ado_rates(
        amplicon_ado_rates,
        rate_col=rate_col,
        amplicon_col=amplicon_col,
        default_rate=default_amplicon_ado_rate,
    )

    missing = [a for a in amps if a not in rates]
    probs = [rates.get(a, default_amplicon_ado_rate) for a in amps]
    probs = [min(max(float(p), 0.0), 1.0) for p in probs]

    m = len(probs)
    if required_amplicons is None:
        required_amplicons = m
    required_amplicons = int(required_amplicons)

    if m == 0:
        p_cell = np.nan
    elif required_amplicons >= m:
        p_cell = float(prod(probs))
    else:
        p_cell = poisson_binomial_tail_at_least(probs, required_amplicons)

    n0 = int(original_clone_size)
    kobs = int(observed_subclone_size)
    expected = float(n0 * p_cell) if np.isfinite(p_cell) else np.nan
    p_tail = binomial_tail_at_least(n0, kobs, p_cell) if np.isfinite(p_cell) else np.nan
    enrichment = (float(kobs) / expected) if expected and expected > 0 else np.inf

    if not np.isfinite(p_tail):
        classification = "ADO_NOT_SCORED"
    elif p_tail >= pvalue_threshold:
        classification = "ADO_COMPATIBLE"
    else:
        classification = "ADO_UNLIKELY"

    return {
        "n_block_amplicons": m,
        "n_required_amplicons": required_amplicons,
        "single_cell_ado_probability": p_cell,
        "expected_ado_cells": expected,
        "p_at_least_observed": p_tail,
        "ado_enrichment": enrichment,
        "classification": classification,
        "block_amplicons": amps,
        "missing_amplicon_rates": missing,
    }


__all__ = [
    "score_ado_combination",
    "normalise_amplicon_ado_rates",
]
