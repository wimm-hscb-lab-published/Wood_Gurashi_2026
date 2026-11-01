"""Tests for LOH_Pipeline/ado_probability.py."""
import math

import numpy as np
import pandas as pd
import pytest
from scipy.stats import binom

from LOH_Pipeline import ado_probability as ap
from LOH_Pipeline.ado_probability import (
    _variant_pos,
    binomial_tail_at_least,
    normalise_amplicon_ado_rates,
    poisson_binomial_tail_at_least,
    score_ado_combination,
)
from LOH_Pipeline.loh_config import LOHConfig, PipelineState


# ─────────────────────────────────────────────────────────────────────
# Pure helpers
# ─────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "v,expected",
    [("chr1:123:A/G", 123), ("chr1:100-200:A/G", 100), ("bad", 10**18), ("chr1:x:A", 10**18), (None, 10**18)],
)
def test_variant_pos(v, expected):
    assert _variant_pos(v) == expected


def _brute_poisson_binomial(probs, k):
    total = 0.0
    m = len(probs)
    for mask in range(1 << m):
        bits = [(mask >> i) & 1 for i in range(m)]
        if sum(bits) >= k:
            total += math.prod(p if b else 1 - p for p, b in zip(probs, bits))
    return total


@pytest.mark.parametrize(
    "probs",
    [[0.1], [0.1, 0.2], [0.05, 0.5, 0.9], [0.3, 0.3, 0.3, 0.3], [0.0, 1.0, 0.5, 0.25, 0.01]],
)
def test_poisson_binomial_matches_brute_force(probs):
    for k in range(0, len(probs) + 2):
        assert poisson_binomial_tail_at_least(probs, k) == pytest.approx(
            _brute_poisson_binomial(probs, k), abs=1e-12
        )


@pytest.mark.parametrize("n,p", [(5, 0.2), (10, 0.37), (3, 0.9)])
def test_poisson_binomial_equal_probs_equals_binomial(n, p):
    for k in range(n + 1):
        assert poisson_binomial_tail_at_least([p] * n, k) == pytest.approx(binom.sf(k - 1, n, p))


def test_poisson_binomial_edge_cases():
    assert np.isnan(poisson_binomial_tail_at_least([], 1))
    assert np.isnan(poisson_binomial_tail_at_least([np.nan, np.inf], 1))
    assert poisson_binomial_tail_at_least([0.3, 0.4], 0) == 1.0
    assert poisson_binomial_tail_at_least([0.3, 0.4], -2) == 1.0
    assert poisson_binomial_tail_at_least([0.3, 0.4], 3) == 0.0
    # non-finite entries are dropped, shrinking m
    assert poisson_binomial_tail_at_least([np.nan, 0.5], 2) == 0.0
    # out-of-range probabilities are clamped
    assert poisson_binomial_tail_at_least([1.7, -0.2], 1) == pytest.approx(1.0)
    assert poisson_binomial_tail_at_least([1.7, -0.2], 2) == pytest.approx(0.0)
    # all-k equals the product
    assert poisson_binomial_tail_at_least([0.1, 0.2, 0.3], 3) == pytest.approx(0.006)


@pytest.mark.parametrize(
    "n,k,p",
    [(10, 1, 0.1), (10, 3, 0.1), (100, 5, 0.01), (50, 50, 0.9), (20, 0, 0.5), (7, 7, 0.0), (7, 1, 1.0)],
)
def test_binomial_tail_matches_scipy(n, k, p):
    expected = 1.0 if k <= 0 else binom.sf(k - 1, n, p)
    assert binomial_tail_at_least(n, k, p) == pytest.approx(expected)


@pytest.mark.parametrize(
    "n,k,p,expected",
    [
        (10, 0, 0.5, 1.0),
        (10, -1, 0.5, 1.0),
        (0, 0, 0.5, 1.0),
        (0, 1, 0.5, 0.0),
        (-5, 1, 0.5, 0.0),
        (3, 4, 0.5, 0.0),
        (4, 2, 1.5, 1.0),  # p clamped to 1
        (4, 1, -1.0, 0.0),  # p clamped to 0
    ],
)
def test_binomial_tail_edge_cases(n, k, p, expected):
    assert binomial_tail_at_least(n, k, p) == pytest.approx(expected)


def test_binomial_tail_fallback_without_scipy(monkeypatch):
    monkeypatch.setattr(ap, "binom", None)
    for n, k, p in [(10, 3, 0.1), (6, 6, 0.5), (12, 1, 0.02)]:
        assert binomial_tail_at_least(n, k, p) == pytest.approx(binom.sf(k - 1, n, p))


# ─────────────────────────────────────────────────────────────────────
# normalise_amplicon_ado_rates
# ─────────────────────────────────────────────────────────────────────
def test_normalise_none_returns_empty():
    assert normalise_amplicon_ado_rates(None) == {}


def test_normalise_dict_percent_clamp_and_bad_values():
    rates = normalise_amplicon_ado_rates(
        {"a": 0.05, "b": 5, "c": 150, "d": -0.1, "e": np.nan, "f": "x", "g": 1.0, 7: "0.2"}
    )
    assert rates == {"a": 0.05, "b": 0.05, "c": 1.0, "d": 0.0, "g": 1.0, "7": 0.2}


def test_normalise_series():
    s = pd.Series({"amp1": 0.1, "amp2": 20.0})
    assert normalise_amplicon_ado_rates(s) == {"amp1": 0.1, "amp2": pytest.approx(0.2)}


def test_normalise_dataframe_named_columns():
    df = pd.DataFrame({"amplicon_id": ["A", "B"], "smooth_ado_any_rate": [0.1, 0.2], "other": [9, 9]})
    assert normalise_amplicon_ado_rates(df) == {"A": 0.1, "B": 0.2}


def test_normalise_dataframe_custom_columns():
    df = pd.DataFrame({"amp": ["A"], "r": [0.3]})
    assert normalise_amplicon_ado_rates(df, rate_col="r", amplicon_col="amp") == {"A": 0.3}


def test_normalise_dataframe_rate_col_fallback_order():
    df = pd.DataFrame({"amplicon_id": ["A"], "ado_rate": [0.9], "raw_ado_any_rate": [0.4]})
    # raw_ado_any_rate precedes ado_rate in the candidate list
    assert normalise_amplicon_ado_rates(df) == {"A": 0.4}


def test_normalise_dataframe_missing_rate_col_raises():
    with pytest.raises(ValueError, match="rate_col"):
        normalise_amplicon_ado_rates(pd.DataFrame({"amplicon_id": ["A"], "x": [1]}))


def test_normalise_dataframe_amplicon_from_named_index():
    df = pd.DataFrame({"smooth_ado_any_rate": [0.1, 0.2]}, index=pd.Index(["A", "B"], name="amp"))
    assert normalise_amplicon_ado_rates(df) == {"A": 0.1, "B": 0.2}


def test_normalise_dataframe_missing_amplicon_col_unnamed_index_raises():
    df = pd.DataFrame({"smooth_ado_any_rate": [0.1]})
    with pytest.raises(ValueError, match="amplicon_col"):
        normalise_amplicon_ado_rates(df)


@pytest.mark.parametrize("bad", [[0.1, 0.2], 0.1, "A"])
def test_normalise_bad_type_raises(bad):
    with pytest.raises(TypeError):
        normalise_amplicon_ado_rates(bad)


# ─────────────────────────────────────────────────────────────────────
# score_ado_combination
# ─────────────────────────────────────────────────────────────────────
def test_score_docstring_example():
    res = score_ado_combination(
        block_amplicons=["a", "b", "c"],
        amplicon_ado_rates={"a": 0.05, "b": 0.05, "c": 0.05},
        original_clone_size=1000,
        observed_subclone_size=5,
    )
    p = 0.05 ** 3
    assert res["single_cell_ado_probability"] == pytest.approx(p)
    assert res["expected_ado_cells"] == pytest.approx(1000 * p)
    assert res["p_at_least_observed"] == pytest.approx(binom.sf(4, 1000, p))
    assert res["ado_enrichment"] == pytest.approx(5 / (1000 * p))
    assert res["classification"] == "ADO_UNLIKELY"
    assert res["n_block_amplicons"] == 3
    assert res["n_required_amplicons"] == 3
    assert res["missing_amplicon_rates"] == []


def test_score_dedupes_and_drops_blank_amplicons_and_uses_default():
    res = score_ado_combination(
        block_amplicons=["a", "a", "", "nan", "None", "b"],
        amplicon_ado_rates={"a": 0.5},
        original_clone_size=10,
        observed_subclone_size=1,
        default_amplicon_ado_rate=0.2,
    )
    assert res["block_amplicons"] == ["a", "b"]
    assert res["missing_amplicon_rates"] == ["b"]
    assert res["single_cell_ado_probability"] == pytest.approx(0.1)


def test_score_required_amplicons_subset_uses_poisson_binomial():
    probs = {"a": 0.1, "b": 0.2, "c": 0.3}
    res = score_ado_combination(
        block_amplicons=list(probs),
        amplicon_ado_rates=probs,
        original_clone_size=50,
        observed_subclone_size=2,
        required_amplicons=2,
    )
    assert res["single_cell_ado_probability"] == pytest.approx(
        _brute_poisson_binomial(list(probs.values()), 2)
    )
    assert res["n_required_amplicons"] == 2


def test_score_required_more_than_available_falls_back_to_product():
    res = score_ado_combination(
        block_amplicons=["a", "b"],
        amplicon_ado_rates={"a": 0.1, "b": 0.2},
        original_clone_size=10,
        observed_subclone_size=1,
        required_amplicons=5,
    )
    assert res["single_cell_ado_probability"] == pytest.approx(0.02)
    assert res["n_required_amplicons"] == 5


def test_score_no_amplicons_not_scored():
    res = score_ado_combination(
        block_amplicons=[], amplicon_ado_rates=None, original_clone_size=10, observed_subclone_size=3
    )
    assert np.isnan(res["single_cell_ado_probability"])
    assert np.isnan(res["expected_ado_cells"])
    assert np.isnan(res["p_at_least_observed"])
    assert res["ado_enrichment"] == np.inf
    assert res["classification"] == "ADO_NOT_SCORED"


def test_score_zero_rate_gives_inf_enrichment_and_unlikely():
    res = score_ado_combination(
        block_amplicons=["a"], amplicon_ado_rates={"a": 0.0}, original_clone_size=100, observed_subclone_size=1
    )
    assert res["expected_ado_cells"] == 0.0
    assert res["ado_enrichment"] == np.inf
    assert res["p_at_least_observed"] == 0.0
    assert res["classification"] == "ADO_UNLIKELY"


def test_score_compatible_and_threshold_boundary():
    res = score_ado_combination(
        block_amplicons=["a"], amplicon_ado_rates={"a": 0.5}, original_clone_size=1, observed_subclone_size=1,
        pvalue_threshold=0.5,
    )
    assert res["p_at_least_observed"] == pytest.approx(0.5)
    assert res["classification"] == "ADO_COMPATIBLE"  # p == threshold counts as compatible


def test_score_zero_observed_is_compatible():
    res = score_ado_combination(
        block_amplicons=["a"], amplicon_ado_rates={"a": 0.1}, original_clone_size=10, observed_subclone_size=0
    )
    assert res["p_at_least_observed"] == 1.0
    assert res["ado_enrichment"] == 0.0
    assert res["classification"] == "ADO_COMPATIBLE"


@pytest.mark.xfail(
    reason="ado_probability.py:137-143 normalise_amplicon_ado_rates normalises default_rate "
    "(percent -> fraction) but discards it; score_ado_combination (line 216) then uses the raw "
    "default_amplicon_ado_rate, so a percentage default like 5 is clamped to 1.0 instead of "
    "becoming 0.05 as table rates do.",
    strict=True,
)
def test_score_percentage_default_rate_is_normalised_like_table_rates():
    res = score_ado_combination(
        block_amplicons=["missing"], amplicon_ado_rates={}, original_clone_size=10,
        observed_subclone_size=1, default_amplicon_ado_rate=5,
    )
    assert res["single_cell_ado_probability"] == pytest.approx(0.05)


# ─────────────────────────────────────────────────────────────────────
# State-based functions (small synthetic PipelineState)
# ─────────────────────────────────────────────────────────────────────
class _FakeDNA:
    def __init__(self, barcodes, clones, col_attrs=None):
        self.row_attrs = {"barcode": np.array(barcodes, dtype=object), "Clone": np.array(clones, dtype=object)}
        self.col_attrs = col_attrs or {}

    def get_attribute(self, *a, **k):
        raise KeyError("DP")


class _FakeSample:
    def __init__(self, barcodes, clones, col_attrs=None):
        self.dna = _FakeDNA(barcodes, clones, col_attrs)


# 6 variants on chr1p spread over 3 amplicons (2 variants each), positions out of order
VARS = ["chr1:600:A/G", "chr1:100:A/G", "chr1:200:A/G", "chr1:300:A/G", "chr1:400:A/G", "chr1:500:A/G"]
AMPS = {"chr1:100:A/G": "amp1", "chr1:200:A/G": "amp1", "chr1:300:A/G": "amp2",
        "chr1:400:A/G": "amp2", "chr1:500:A/G": "amp3", "chr1:600:A/G": "amp3"}


def _make_state(cfg=None, col_attrs="default"):
    n_loh, n_base = 5, 15
    barcodes = [f"BC{i:02d}" for i in range(n_loh + n_base + 3)]
    clones = (["A LOH-C0 chr1p"] * n_loh) + (["A"] * n_base) + (["B"] * 3)
    # position-sorted AFs: LOH cells extreme on 100..400 (4 variants), het on 500, 600
    sorted_vars = sorted(VARS, key=_variant_pos)
    rows = []
    for i in range(len(barcodes)):
        if i < n_loh:
            rows.append([1.0, 2.0, 98.0, 99.0, 50.0, 45.0])
        else:
            rows.append([50.0, 48.0, 52.0, 47.0, 51.0, 49.0])
    df = pd.DataFrame(rows, index=barcodes, columns=sorted_vars)[VARS]
    if col_attrs == "default":
        col_attrs = {"id": list(AMPS), "amplicon": list(AMPS.values())}
    sample = _FakeSample(barcodes, clones, col_attrs)
    return PipelineState(
        Sample=sample, df=df, grouped_by_chr_arm={"chr1p": VARS}, chr_arms=["chr1p"],
        config=cfg or LOHConfig(verbose=False),
    )
