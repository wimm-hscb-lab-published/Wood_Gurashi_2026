"""Unit tests for pure helpers in LOH_Pipeline/phase1.py."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from LOH_Pipeline import phase1 as p1
from LOH_Pipeline.loh_config import LOHConfig, PipelineState


# ── fakes ─────────────────────────────────────────────────────────────

def _fake_sample(barcodes, clones, col_ids=None, amplicons=None):
    col_attrs = {}
    if col_ids is not None:
        col_attrs["id"] = np.asarray(col_ids, dtype=object)
    if amplicons is not None:
        col_attrs["amplicon"] = np.asarray(amplicons, dtype=object)
    dna = SimpleNamespace(
        row_attrs={
            "barcode": np.asarray(barcodes, dtype=object),
            "Clone": np.asarray(clones, dtype=object),
        },
        col_attrs=col_attrs,
    )
    return SimpleNamespace(dna=dna)


def _make_state(af: pd.DataFrame, clones, arm, cfg=None, amplicons=None):
    cfg = cfg or LOHConfig(verbose=False)
    variants = list(af.columns)
    sample = _fake_sample(
        list(af.index), clones,
        col_ids=variants,
        amplicons=amplicons if amplicons is not None else None,
    )
    return PipelineState(sample, af, {arm: variants}, [arm], cfg)


# ── _variant_pos ──────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "v, expected",
    [
        ("chr5:12345:A/G", 12345),
        ("chr17:1:C/T", 1),
        ("chrX:999", 999),
        ("garbage", 10**18),
        ("chr1:abc:A/T", 10**18),
        ("", 10**18),
        (None, 10**18),
    ],
)
def test_variant_pos(v, expected):
    assert p1._variant_pos(v) == expected


def test_variant_pos_range_style_is_unparseable_in_phase1():
    # phase1 does not split on '-', unlike phase4/phase6 helpers.
    assert p1._variant_pos("chr1:100-200:A/T") == 10**18


def test_variant_pos_sort_puts_unparseable_last():
    vs = ["chr1:300:A/T", "bad", "chr1:100:A/T", "chr1:200:A/T"]
    assert sorted(vs, key=p1._variant_pos) == [
        "chr1:100:A/T", "chr1:200:A/T", "chr1:300:A/T", "bad",
    ]


# ── _build_variant_to_amplicon ────────────────────────────────────────

def test_build_variant_to_amplicon_basic():
    state = SimpleNamespace(Sample=_fake_sample(
        [], [], col_ids=["v1", "v2"], amplicons=["A1", "A2"]))
    assert p1._build_variant_to_amplicon(state) == {"v1": "A1", "v2": "A2"}


def test_build_variant_to_amplicon_length_mismatch_returns_empty():
    state = SimpleNamespace(Sample=_fake_sample(
        [], [], col_ids=["v1", "v2"], amplicons=["A1"]))
    assert p1._build_variant_to_amplicon(state) == {}


def test_build_variant_to_amplicon_missing_attrs_returns_empty():
    assert p1._build_variant_to_amplicon(SimpleNamespace()) == {}
    state = SimpleNamespace(Sample=_fake_sample([], []))
    assert p1._build_variant_to_amplicon(state) == {}


# ── _informative_variant_set ─────────────────────────────────────────

@pytest.mark.parametrize(
    "vals, expected",
    [
        (None, None),
        ([], None),
        (["a", "b"], {"a", "b"}),
        ([1, 2], {"1", "2"}),
        (5, None),  # not iterable
    ],
)
def test_informative_variant_set(vals, expected):
    cfg = SimpleNamespace(informative_variants=vals)
    assert p1._informative_variant_set(cfg) == expected


def test_informative_variant_set_missing_attr():
    assert p1._informative_variant_set(SimpleNamespace()) is None


# ── _block_somatic_flanking_ok ───────────────────────────────────────

def _flank_cfg(info, **kw):
    base = dict(
        informative_variants=info,
        enforce_somatic_block_flanking=True,
        min_informative_variants_flanking_somatic=2,
        require_informative_on_both_sides_of_somatic=True,
    )
    base.update(kw)
    return SimpleNamespace(**base)


def test_flanking_disabled_always_passes():
    cfg = _flank_cfg(["x"], enforce_somatic_block_flanking=False)
    assert p1._block_somatic_flanking_ok(["a", "b"], [0, 1], cfg) == (True, "")


def test_flanking_no_informative_set_passes():
    cfg = _flank_cfg(None)
    assert p1._block_somatic_flanking_ok(["a"], [0], cfg) == (True, "")


def test_flanking_empty_block_fails():
    cfg = _flank_cfg(["a"])
    assert p1._block_somatic_flanking_ok(["a"], [], cfg) == (False, "empty_block")


def test_flanking_all_informative_passes():
    cfg = _flank_cfg(["a", "b", "c"])
    assert p1._block_somatic_flanking_ok(["a", "b", "c"], [0, 1, 2], cfg) == (True, "")


def test_flanking_too_few_informative():
    cfg = _flank_cfg(["a"])
    ok, reason = p1._block_somatic_flanking_ok(["a", "s", "b"], [0, 1, 2], cfg)
    assert not ok
    assert reason == "somatic_block:n_informative<2"


def test_flanking_somatic_flanked_on_both_sides_passes():
    cfg = _flank_cfg(["a", "b"])
    assert p1._block_somatic_flanking_ok(["a", "s", "b"], [0, 1, 2], cfg) == (True, "")


def test_flanking_somatic_at_edge_fails():
    cfg = _flank_cfg(["a", "b"])
    ok, reason = p1._block_somatic_flanking_ok(["a", "b", "s"], [0, 1, 2], cfg)
    assert not ok
    assert reason == "somatic_block:not_flanked_by_informative_variants"


def test_flanking_somatic_at_edge_ok_when_both_sides_not_required():
    cfg = _flank_cfg(["a", "b"], require_informative_on_both_sides_of_somatic=False)
    assert p1._block_somatic_flanking_ok(["a", "b", "s"], [0, 1, 2], cfg) == (True, "")


def test_flanking_uses_only_block_indices():
    cfg = _flank_cfg(["a", "b"])
    vars_sorted = ["s0", "a", "s", "b", "s4"]
    assert p1._block_somatic_flanking_ok(vars_sorted, [1, 2, 3], cfg) == (True, "")


# ── _informative_amplicons_in_block ──────────────────────────────────

def test_informative_amplicons_no_filter():
    cfg = SimpleNamespace(informative_variants=None)
    v2a = {"a": "A1", "b": "A1", "c": "A2", "d": "nan", "e": ""}
    out = p1._informative_amplicons_in_block(["a", "b", "c", "d", "e", "missing"], v2a, cfg)
    assert out == {"A1", "A2"}


def test_informative_amplicons_filters_somatic():
    cfg = SimpleNamespace(informative_variants=["a"])
    v2a = {"a": "A1", "c": "A2"}
    assert p1._informative_amplicons_in_block(["a", "c"], v2a, cfg) == {"A1"}


def test_informative_amplicons_empty():
    cfg = SimpleNamespace(informative_variants=None)
    assert p1._informative_amplicons_in_block([], {}, cfg) == set()


# ── _strip_specific_loh_token_from_label ─────────────────────────────

@pytest.mark.parametrize(
    "label, arm, clone, expected",
    [
        ("TP53_Hem LOH-C1 chr17p", "chr17p", "TP53_Hem LOH-C1 chr17p", "TP53_Hem"),
        ("TP53_Hem LOH-C1 chr17p:p13.1-p13.3", "chr17p", "LOH-C1", "TP53_Hem"),
        # Other arms/tokens are preserved
        ("A LOH-C0 chr5q LOH-C1 chr17p", "chr17p", "LOH-C1", "A LOH-C0 chr5q"),
        ("A LOH-C0 chr5q LOH-C1 chr17p", "chr5q", "LOH-C0", "A LOH-C1 chr17p"),
        # Different clone id -> no change
        ("A LOH-C1 chr17p", "chr17p", "LOH-C2", "A LOH-C1 chr17p"),
        # LOH-C1 must not match LOH-C12
        ("A LOH-C12 chr17p", "chr17p", "LOH-C1", "A LOH-C12 chr17p"),
        # chr1p must not match chr17p
        ("A LOH-C1 chr17p", "chr1p", "LOH-C1", "A LOH-C1 chr17p"),
        # Clone label without LOH token -> unchanged
        ("A LOH-C1 chr17p", "chr17p", "no token", "A LOH-C1 chr17p"),
    ],
)
def test_strip_specific_loh_token(label, arm, clone, expected):
    assert p1._strip_specific_loh_token_from_label(label, arm, clone) == expected


def test_strip_specific_collapses_whitespace():
    out = p1._strip_specific_loh_token_from_label("A  LOH-C1 chr17p  B", "chr17p", "LOH-C1")
    assert out == "A B"


# ── _ngt_no_call_mask ────────────────────────────────────────────────

def test_ngt_no_call_mask_defaults():
    cfg = LOHConfig(verbose=False)
    vals = pd.DataFrame(
        {"v1": [0, 1, 3, np.nan], "v2": ["NGT", " ngt ", "./.", "2"]},
        index=["c1", "c2", "c3", "c4"],
        dtype=object,
    )
    out = p1._ngt_no_call_mask(vals, cfg)
    assert out.index.tolist() == vals.index.tolist()
    assert out.columns.tolist() == vals.columns.tolist()
    assert out["v1"].tolist() == [False, False, True, True]
    assert out["v2"].tolist() == [True, True, True, False]


def test_ngt_no_call_mask_numeric_string_3_counts():
    cfg = LOHConfig(verbose=False)
    vals = pd.DataFrame({"v": ["3", "3.0", "0"]})
    assert p1._ngt_no_call_mask(vals, cfg)["v"].tolist() == [True, True, False]


def test_ngt_no_call_mask_no_rules_all_false():
    cfg = SimpleNamespace(
        phase1_ngt_no_call_values=(),
        phase1_ngt_numeric_no_call_values=(),
        phase1_ngt_treat_nan_as_ngt=False,
    )
    vals = pd.DataFrame({"v": [np.nan, 3, "NGT"]}, dtype=object)
    assert not p1._ngt_no_call_mask(vals, cfg).values.any()


def test_ngt_no_call_mask_empty_frame():
    cfg = LOHConfig(verbose=False)
    out = p1._ngt_no_call_mask(pd.DataFrame(), cfg)
    assert out.empty


@pytest.mark.xfail(
    reason=(
        "phase1._ngt_no_call_mask: phase1_ngt_treat_nan_as_ngt=False is ineffective "
        "with the default phase1_ngt_no_call_values, because NaN is stringified to "
        "'nan' -> 'NAN', which is in the default string no-call set."
    ),
    strict=True,
)
def test_ngt_no_call_mask_treat_nan_false_is_respected():
    cfg = LOHConfig(verbose=False, phase1_ngt_treat_nan_as_ngt=False)
    vals = pd.DataFrame({"v": [np.nan, 0.0]})
    assert p1._ngt_no_call_mask(vals, cfg)["v"].tolist() == [False, False]


# ── _extreme_block_dispersion_stats ──────────────────────────────────

def test_dispersion_stats_empty_block():
    med, p90, n = p1._extreme_block_dispersion_stats(np.zeros((3, 2)), [], 10, 90)
    assert np.isnan(med) and np.isnan(p90) and n == 0


def test_dispersion_stats_wrong_ndim():
    med, p90, n = p1._extreme_block_dispersion_stats(np.zeros(3), [0], 10, 90)
    assert np.isnan(med) and n == 0


def test_dispersion_stats_folds_low_and_high_symmetrically():
    # column 0: LOW values 0, 4 -> folded 0,4 -> sd = 2.828
    # column 1: HIGH values 100, 96 -> folded 0,4 -> same sd
    af = np.array([[0.0, 100.0], [4.0, 96.0]])
    med, p90, n = p1._extreme_block_dispersion_stats(af, [0, 1], 10, 90)
    expected = float(np.std([0, 4], ddof=1))
    assert n == 2
    assert med == pytest.approx(expected)
    assert p90 == pytest.approx(expected)


def test_dispersion_stats_skips_non_extreme_and_nan():
    af = np.array([
        [0.0, 50.0, np.nan],
        [2.0, 50.0, 1.0],
        [50.0, 50.0, np.nan],
    ])
    med, p90, n = p1._extreme_block_dispersion_stats(af, [0, 1, 2], 10, 90)
    # only column 0 contributes (2 extreme values); col 1 none extreme; col 2 one finite
    assert n == 1
    assert med == pytest.approx(float(np.std([0.0, 2.0], ddof=1)))


# ── _amplicon_validate_clone (with a real PipelineState on fakes) ────

ARM = "chr17p"


def _validation_state(af_value_rows, amplicons, clone="A LOH-C0 chr17p", **cfg_kw):
    variants = [f"chr17:{100 * (i + 1)}:A/T" for i in range(len(af_value_rows[0]))]
    af = pd.DataFrame(af_value_rows, columns=variants,
                      index=[f"bc{i}" for i in range(len(af_value_rows))])
    cfg = LOHConfig(verbose=False, **cfg_kw)
    state = _make_state(af, [clone] * len(af), ARM, cfg=cfg, amplicons=amplicons)
    return state, cfg, p1._build_variant_to_amplicon(state)


def test_amplicon_validate_pass_with_enough_amplicons():
    rows = [[0, 0, 0, 0, 50]] * 3
    state, cfg, v2a = _validation_state(rows, ["a1", "a2", "a3", "a3", "a4"])
    assert p1._amplicon_validate_clone(state, ARM, "A LOH-C0 chr17p", cfg, v2a) is True


def test_amplicon_validate_fails_with_too_few_amplicons():
    rows = [[0, 0, 0, 0, 50]] * 3
    state, cfg, v2a = _validation_state(rows, ["a1", "a1", "a2", "a2", "a3"])
    assert p1._amplicon_validate_clone(state, ARM, "A LOH-C0 chr17p", cfg, v2a) is False


def test_amplicon_validate_fails_when_no_extreme_block():
    rows = [[50, 50, 50, 50, 50]] * 3
    state, cfg, v2a = _validation_state(rows, ["a1", "a2", "a3", "a4", "a5"])
    assert p1._amplicon_validate_clone(state, ARM, "A LOH-C0 chr17p", cfg, v2a) is False


def test_amplicon_validate_passes_by_default_with_too_few_variants():
    rows = [[0, 0, 0]] * 3
    state, cfg, v2a = _validation_state(rows, ["a1", "a2", "a3"])
    assert p1._amplicon_validate_clone(state, ARM, "A LOH-C0 chr17p", cfg, v2a) is True


def test_amplicon_validate_passes_for_unknown_clone():
    rows = [[50, 50, 50, 50, 50]] * 3
    state, cfg, v2a = _validation_state(rows, ["a1", "a2", "a3", "a4", "a5"])
    assert p1._amplicon_validate_clone(state, ARM, "not a clone", cfg, v2a) is True
