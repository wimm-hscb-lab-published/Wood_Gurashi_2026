"""
ADO-combination probability scoring for LOH clones.

This module asks whether an observed LOH-like subclone could plausibly arise
from independent amplicon-level ADO events within the original/base clone.

Example
-------
If a LOH block spans 3 independent amplicons, each with baseline ADO rate 5%,
the single-cell probability of matching all 3 by chance is 0.05**3 = 0.000125.
For an original clone of size X, the probability of seeing >= k matching cells is
Binomial.sf(k-1, X, 0.000125).
"""
from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Any, Iterable

import numpy as np
import pandas as pd

try:
    from scipy.stats import binom
except Exception:  # pragma: no cover
    binom = None


@dataclass
class ADOCombinationResult:
    clone_label: str
    base_clone: str
    loh_arms: str
    n_observed_cells: int
    n_original_clone_cells: int
    n_block_variants: int
    n_block_amplicons: int
    n_required_amplicons: int
    single_cell_ado_probability: float
    expected_ado_cells: float
    p_at_least_observed: float
    ado_enrichment: float
    classification: str
    block_amplicons: str
    block_variants: str
    missing_amplicon_rates: str


def _variant_pos(v: str) -> int:
    try:
        return int(str(v).split(":")[1].split("-")[0])
    except Exception:
        return 10**18


def _build_variant_to_amplicon_from_state(state) -> dict[str, str]:
    """Return variant_id -> amplicon_id from Sample.dna.col_attrs, falling back to variant IDs."""
    out: dict[str, str] = {}
    try:
        col_attrs = state.Sample.dna.col_attrs
        ids = list(col_attrs.get("id", []))
        amps = list(col_attrs.get("amplicon", []))
        if len(ids) == len(amps) and len(ids) > 0:
            out.update({str(i): str(a) for i, a in zip(ids, amps)})
    except Exception:
        pass

    for v in getattr(state, "variant_cols", []):
        out.setdefault(str(v), str(v))
    return out


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
    """P(sum Bernoulli(probs) >= k) via dynamic programming."""
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
    """P(Binomial(n, p) >= k)."""
    n = int(n)
    k = int(k)
    p = min(max(float(p), 0.0), 1.0)
    if k <= 0:
        return 1.0
    if n <= 0 or k > n:
        return 0.0
    if binom is not None:
        return float(binom.sf(k - 1, n, p))

    # Fallback; fine for modest n.
    from math import comb
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

    Returns a dictionary with:
      - single_cell_ado_probability: probability one cell matches the block
      - expected_ado_cells: original_clone_size * p
      - p_at_least_observed: P(>= observed_subclone_size matching cells)
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


def infer_loh_block_variants_for_clone(state, clone_label: str) -> list[str]:
    """
    Infer the current best LOH block variants for a clone.

    Uses state.clone_block_variants when present; otherwise reconstructs from
    each LOH arm by taking contiguous extreme runs in the clone mean AF profile.
    """
    from .loh_config import LOH_TAG, arm_variants, extract_chr_arms, is_loh_label
    from scipy.ndimage import label as ndimage_label

    block_map = getattr(state, "clone_block_variants", {}) or {}
    if clone_label in block_map:
        return [v for v in block_map[clone_label] if v in state.df_numeric.columns]

    cfg = state.config
    if not is_loh_label(clone_label):
        return []
    cells = [c for c in state.clone_to_cells.get(clone_label, []) if c in state.df_numeric.index]
    if not cells:
        return []

    all_block_vars: list[str] = []
    for arm in extract_chr_arms(clone_label):
        vars_arm = [v for v in arm_variants(arm, state) if v in state.df_numeric.columns]
        vars_sorted = sorted(vars_arm, key=_variant_pos)
        if not vars_sorted:
            continue
        vals = state.df_numeric.loc[cells, vars_sorted].values.astype(float)
        mu = np.nanmean(vals, axis=0)
        extreme = ((mu <= cfg.low_allele_threshold) | (mu >= cfg.high_allele_threshold)).astype(int)
        lab, nruns = ndimage_label(extreme)
        best_idx: list[int] = []
        best_ext = -1.0
        for rid in range(1, nruns + 1):
            idx = np.where(lab == rid)[0]
            if len(idx) == 0:
                continue
            blk = vals[:, idx]
            fin = np.isfinite(blk)
            if not fin.any():
                continue
            ext = float(((blk[fin] <= cfg.low_allele_threshold) | (blk[fin] >= cfg.high_allele_threshold)).mean())
            if len(idx) > len(best_idx) or (len(idx) == len(best_idx) and ext > best_ext):
                best_idx = list(map(int, idx))
                best_ext = ext
        all_block_vars.extend([vars_sorted[i] for i in best_idx])

    return list(dict.fromkeys(all_block_vars))


def original_clone_size_for_label(state, clone_label: str) -> int:
    """Current total cells sharing the same base clone, including LOH subclones."""
    from .loh_config import get_base_clone

    base = get_base_clone(clone_label)
    n = 0
    for lbl, cells in state.clone_to_cells.items():
        if get_base_clone(str(lbl)) == base:
            n += len(cells)
    return int(n)


def score_loh_clone_ado_probability(
    state,
    clone_label: str,
    amplicon_ado_rates: Any,
    *,
    required_amplicons: int | None = None,
    rate_col: str = "smooth_ado_any_rate",
    amplicon_col: str = "amplicon_id",
    default_amplicon_ado_rate: float = 0.01,
    pvalue_threshold: float = 0.05,
) -> dict[str, Any]:
    from .loh_config import extract_chr_arms, get_base_clone

    v2a = _build_variant_to_amplicon_from_state(state)
    block_vars = infer_loh_block_variants_for_clone(state, clone_label)
    amps = [v2a.get(v, v) for v in block_vars]
    amps = list(dict.fromkeys([str(a) for a in amps if str(a) not in {"", "nan", "None"}]))

    n_obs = len(state.clone_to_cells.get(clone_label, []))
    n_orig = original_clone_size_for_label(state, clone_label)

    scored = score_ado_combination(
        block_amplicons=amps,
        amplicon_ado_rates=amplicon_ado_rates,
        original_clone_size=n_orig,
        observed_subclone_size=n_obs,
        required_amplicons=required_amplicons,
        rate_col=rate_col,
        amplicon_col=amplicon_col,
        default_amplicon_ado_rate=default_amplicon_ado_rate,
        pvalue_threshold=pvalue_threshold,
    )

    return {
        "clone_label": clone_label,
        "base_clone": get_base_clone(clone_label),
        "loh_arms": ", ".join(extract_chr_arms(clone_label)),
        "n_observed_cells": int(n_obs),
        "n_original_clone_cells": int(n_orig),
        "n_block_variants": int(len(block_vars)),
        "n_block_amplicons": int(scored["n_block_amplicons"]),
        "n_required_amplicons": int(scored["n_required_amplicons"]),
        "single_cell_ado_probability": float(scored["single_cell_ado_probability"]) if np.isfinite(scored["single_cell_ado_probability"]) else np.nan,
        "expected_ado_cells": float(scored["expected_ado_cells"]) if np.isfinite(scored["expected_ado_cells"]) else np.nan,
        "p_at_least_observed": float(scored["p_at_least_observed"]) if np.isfinite(scored["p_at_least_observed"]) else np.nan,
        "ado_enrichment": float(scored["ado_enrichment"]) if np.isfinite(scored["ado_enrichment"]) else np.inf,
        "classification": scored["classification"],
        "block_amplicons": ",".join(scored["block_amplicons"]),
        "block_variants": ",".join(block_vars),
        "missing_amplicon_rates": ",".join(scored["missing_amplicon_rates"]),
    }


def run_ado_probability_qc(state, amplicon_ado_rates: Any | None = None) -> pd.DataFrame:
    """Score all current LOH clones for ADO-combination probability."""
    from .loh_config import LOH_TAG, vprint, is_loh_label

    cfg = state.config
    enabled = bool(getattr(cfg, "ado_probability_qc_enabled", False))
    if not enabled:
        state.ado_probability = pd.DataFrame()
        return state.ado_probability

    if amplicon_ado_rates is None:
        amplicon_ado_rates = getattr(cfg, "amplicon_ado_rates", None)
    if amplicon_ado_rates is None:
        msg = "ADO-combination probability QC requires cfg.amplicon_ado_rates, but none were provided."
        if bool(getattr(cfg, "ado_probability_require_rates", False)):
            raise ValueError(msg)
        vprint("\n  ADO-combination probability QC skipped: no amplicon_ado_rates provided.", cfg)
        state.ado_probability = pd.DataFrame()
        return state.ado_probability

    rate_col = str(getattr(cfg, "ado_probability_rate_col", "smooth_ado_any_rate"))
    amplicon_col = str(getattr(cfg, "ado_probability_amplicon_col", "amplicon_id"))
    default_rate = float(getattr(cfg, "ado_probability_default_amplicon_rate", 0.01))
    pthr = float(getattr(cfg, "ado_probability_pvalue_threshold", 0.05))
    required = getattr(cfg, "ado_probability_required_amplicons", None)
    if required is not None:
        required = int(required)

    rows = []
    state.refresh()
    for lbl, cells in sorted(state.clone_to_cells.items(), key=lambda kv: -len(kv[1])):
        if not is_loh_label(lbl) or len(cells) == 0:
            continue
        rows.append(
            score_loh_clone_ado_probability(
                state,
                str(lbl),
                amplicon_ado_rates,
                required_amplicons=required,
                rate_col=rate_col,
                amplicon_col=amplicon_col,
                default_amplicon_ado_rate=default_rate,
                pvalue_threshold=pthr,
            )
        )

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(
            ["p_at_least_observed", "expected_ado_cells"],
            ascending=[True, True],
            na_position="last",
        ).reset_index(drop=True)

    state.ado_probability = out
    state.results["ado_probability_qc"] = out

    vprint("\n" + "-" * 60, cfg)
    vprint("ADO-combination probability QC", cfg)
    vprint("-" * 60, cfg)
    if out.empty:
        vprint("  No LOH clones available for ADO probability scoring.", cfg)
    else:
        vprint(
            "  Interprets p_at_least_observed as: P(>= observed LOH-like cells "
            "within original/base clone under amplicon-specific ADO).",
            cfg,
        )
        for _, r in out.iterrows():
            vprint(
                f"  • {r['clone_label']}  n={int(r['n_observed_cells'])}/{int(r['n_original_clone_cells'])}  "
                f"amps={int(r['n_block_amplicons'])}  "
                f"p_cell={r['single_cell_ado_probability']:.3g}  "
                f"E={r['expected_ado_cells']:.3g}  "
                f"P≥obs={r['p_at_least_observed']:.3g}  "
                f"{r['classification']}",
                cfg,
            )

    return out


__all__ = [
    "score_ado_combination",
    "score_loh_clone_ado_probability",
    "run_ado_probability_qc",
    "normalise_amplicon_ado_rates",
]
