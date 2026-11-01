"""Tests for LOH_Pipeline/loh_config.py (labels, helpers, LOHConfig, PipelineState)."""
import dataclasses

import numpy as np
import pandas as pd
import pytest
from scipy.stats import pearsonr

from LOH_Pipeline import loh_config as lc
from LOH_Pipeline.loh_config import (
    LOHConfig,
    PipelineState,
    build_cytoband_maps,
    check_directional_consistency,
    clone_category,
    edge_tolerant_extreme_runs,
    extract_chr_arms,
    fmt_float,
    fp_base_name,
    get_base_clone,
    is_fp_label,
    is_hem_like,
    is_loh_label,
    is_small_label,
    is_sub_label,
    is_uncertain_loh_label,
    is_wt_clone,
    make_fp_label,
    make_sub_label,
    mean_profile,
    pearson_safe,
    sort_chromosome_arms,
    vprint,
    arm_variants,
)


# ─────────────────────────────────────────────────────────────────────
# Minimal fake mosaic Sample
# ─────────────────────────────────────────────────────────────────────
class _FakeDNA:
    def __init__(self, barcodes, clones, dp=None):
        self.row_attrs = {
            "barcode": np.array(barcodes, dtype=object),
            "Clone": np.array(clones, dtype=object),
        }
        self.col_attrs = {}
        self._dp = dp

    def get_attribute(self, name, constraint=None):
        if self._dp is None:
            raise KeyError(name)
        return self._dp


class _FakeSample:
    def __init__(self, barcodes, clones, dp=None):
        self.dna = _FakeDNA(barcodes, clones, dp)


VARS = ["chr1:100:A/G", "chr1:200:C/T", "chr1:300:G/A"]


def _make_state(clones=None, df=None, df_dp=None, sample_dp=None, index=None):
    barcodes = ["AAA", "CCC", "GGG", "TTT"]
    if clones is None:
        clones = ["A", "A LOH-C0 chr1p", "FP: A C1 chr1p", "B"]
    if df is None:
        df = pd.DataFrame(
            np.arange(12, dtype=float).reshape(4, 3) * 10,
            index=index if index is not None else barcodes,
            columns=VARS,
        )
    sample = _FakeSample(barcodes, clones, dp=sample_dp)
    return PipelineState(
        Sample=sample,
        df=df,
        grouped_by_chr_arm={"chr1p": VARS + ["chr1:999:missing"]},
        chr_arms=["chr1p"],
        config=LOHConfig(verbose=False),
        df_dp=df_dp,
    )


# ─────────────────────────────────────────────────────────────────────
# Label helpers
# ─────────────────────────────────────────────────────────────────────
LABEL_CASES = [
    # label, small, fp, sub, loh, category
    ("A LOH-C0 chr1p", False, False, False, True, "LOH"),
    ("  A LOH-C0 chr1p  ", False, False, False, True, "LOH"),
    ("FP: A C0 chr1p", False, True, False, False, "possible_LOH"),
    ("FP: A LOH-C0 chr1p", False, True, False, False, "possible_LOH"),
    ("SUB: A C0 chr1p", False, False, True, False, "subclonal_LOH"),
    ("small", True, False, False, False, "small"),
    ("small: A LOH-C0 chr1p", True, False, False, False, "small"),
    ("small: FP: A C0 chr1p", True, False, False, False, "small"),
    (" small ", True, False, False, False, "small"),
    ("smaller LOH-C0", False, False, False, True, "LOH"),  # not the 'small' token
    ("A", False, False, False, False, "non_LOH"),
    ("WT", False, False, False, False, "non_LOH"),
    ("", False, False, False, False, "non_LOH"),
    (None, False, False, False, False, "non_LOH"),
    ("loh lowercase", False, False, False, False, "non_LOH"),  # case-sensitive
]


@pytest.mark.parametrize("label,small,fp,sub,loh,cat", LABEL_CASES)
def test_label_predicates_and_category(label, small, fp, sub, loh, cat):
    assert is_small_label(label) is small
    assert is_fp_label(label) is fp
    assert is_sub_label(label) is sub
    assert is_uncertain_loh_label(label) is (fp or sub)
    assert is_loh_label(label) is loh
    assert clone_category(label) == cat


@pytest.mark.parametrize(
    "inp,expected",
    [
        ("A LOH-C0 chr7p", "FP: A C0 chr7p"),
        ("TP53_R248W_Hem LOH-C0 chr7p:p22.1-p14.1", "FP: TP53_R248W_Hem C0 chr7p:p22.1-p14.1"),
        ("A LOH-C3 chr1p LOH-C12 chr2q", "FP: A C3 chr1p C12 chr2q"),
        ("  A  ", "FP: A"),
        ("FP: already", "FP: already"),  # idempotent
        ("SUB: A C0 chr1p", "FP: SUB: A C0 chr1p"),
    ],
)
def test_make_fp_label(inp, expected):
    out = make_fp_label(inp)
    assert out == expected
    assert not is_loh_label(out)
    assert is_fp_label(out)


def test_make_fp_label_is_idempotent():
    once = make_fp_label("A LOH-C0 chr1p")
    assert make_fp_label(once) == once


@pytest.mark.parametrize(
    "inp,expected",
    [
        ("A LOH-C0 chr7p", "SUB: A C0 chr7p"),
        ("SUB: A C0 chr7p", "SUB: A C0 chr7p"),
        ("A", "SUB: A"),
    ],
)
def test_make_sub_label(inp, expected):
    out = make_sub_label(inp)
    assert out == expected
    assert is_sub_label(out)
    assert clone_category(out) == "subclonal_LOH"


@pytest.mark.parametrize(
    "inp,expected",
    [
        ("FP: TP53_R248W_Hem C0 chr7p:p22.1-p14.1", "TP53_R248W_Hem"),
        ("SUB: A C1 chr1p C2 chr2q", "A"),
        ("small: FP: A C0 chr1p", "A"),
        ("FP:A C0 chr1p", "A"),
        ("DNMT3A_C123X_Het", "DNMT3A_C123X_Het"),  # '_C123' not stripped (no word boundary)
        ("A   B", "A B"),
        ("", ""),
    ],
)
def test_fp_base_name(inp, expected):
    assert fp_base_name(inp) == expected


def test_fp_base_name_roundtrip_from_make_fp_label():
    label = "TP53_R248W_Hem LOH-C4 chr17p"
    assert fp_base_name(make_fp_label(label)) == get_base_clone(label) == "TP53_R248W_Hem"


# ─────────────────────────────────────────────────────────────────────
# Chromosome / clone string helpers
# ─────────────────────────────────────────────────────────────────────
def test_sort_chromosome_arms_natural_order():
    arms = ["chrXq", "chr10p", "chr2q", "chr2p", "foo", "chrXp", "chr1q", "chrYp", "chr1p"]
    assert sort_chromosome_arms(arms) == [
        "chr1p", "chr1q", "chr2p", "chr2q", "chr10p", "chrXp", "chrXq", "chrYp", "foo",
    ]


def test_sort_chromosome_arms_empty_and_unmatched_alpha():
    assert sort_chromosome_arms([]) == []
    assert sort_chromosome_arms(["zz", "aa"]) == ["aa", "zz"]


@pytest.mark.parametrize(
    "label,expected",
    [
        ("A", "A"),
        ("A LOH-C0 chr1p", "A"),
        ("A LOH-C0 chr1p LOH-C3 chr2q", "A"),
        ("A LOH-C0 chr1p+chr1q", "A"),
        ("A_Hem LOH-C12 chr17p:p13.1", "A_Hem"),
        ("FP: A C0 chr1p", "FP: A C0 chr1p"),
        ("A LOH-C0", "A LOH-C0"),  # no arm token -> unchanged
    ],
)
def test_get_base_clone(label, expected):
    assert get_base_clone(label) == expected


@pytest.mark.parametrize(
    "label,expected",
    [
        ("A LOH-C0 chr1p", ["chr1p"]),
        ("A LOH-C0 chr1p+chr1q", ["chr1p", "chr1q"]),
        ("A LOH-C0 chr17p:p13.1 LOH-C2 chrXq", ["chr17p", "chrXq"]),
        ("A", []),
        ("chr1 chr2", []),
        (None, []),
    ],
)
def test_extract_chr_arms(label, expected):
    assert extract_chr_arms(label) == expected


def test_is_wt_clone_default_and_custom_prefixes():
    cfg = LOHConfig()
    assert is_wt_clone("WT", cfg)
    assert is_wt_clone("WT_1", cfg)
    assert not is_wt_clone("wt", cfg)
    assert not is_wt_clone("Normal", cfg)
    cfg.wt_labels = ["Normal", "Ref"]
    assert is_wt_clone("Normal_2", cfg)
    assert is_wt_clone("Ref", cfg)
    assert not is_wt_clone("A", cfg)


@pytest.mark.parametrize(
    "label,expected",
    [
        ("TP53_Hem", True),
        ("TP53 HEM LOH-C0", True),
        ("hem", True),
        ("x-hem.y", True),
        ("Hemizygous", False),
        ("TP53_Hom", False),
        ("ChemX", False),
    ],
)
def test_is_hem_like(label, expected):
    assert is_hem_like(label) is expected


# ─────────────────────────────────────────────────────────────────────
# edge_tolerant_extreme_runs
# ─────────────────────────────────────────────────────────────────────
def _spans(runs):
    return [(r["core_indices"], r["span_indices"], r["n_edge_padded"]) for r in runs]


@pytest.mark.parametrize("mask", [[], [0, 0, 0], np.zeros(0)])
def test_edge_runs_empty(mask):
    assert edge_tolerant_extreme_runs(mask, 1) == []


def test_edge_runs_no_tolerance_splits_cores():
    runs = edge_tolerant_extreme_runs([1, 1, 0, 1, 1], 0)
    assert _spans(runs) == [([0, 1], [0, 1], 0), ([3, 4], [3, 4], 0)]


def test_edge_runs_internal_gap_never_bridged():
    runs = edge_tolerant_extreme_runs([1, 1, 0, 1, 1], 1)
    # the contested middle index is dropped from both blocks
    assert _spans(runs) == [([0, 1], [0, 1], 0), ([3, 4], [3, 4], 0)]


@pytest.mark.parametrize(
    "mask,tol,expected",
    [
        ([1, 1, 0], 1, [([0, 1], [0, 1, 2], 1)]),
        ([0, 1, 1], 1, [([1, 2], [0, 1, 2], 1)]),
        ([0, 0, 1, 1, 0, 0], 1, [([2, 3], [1, 2, 3, 4], 2)]),
        ([0, 0, 1, 1, 0, 0], 2, [([2, 3], [0, 1, 2, 3, 4, 5], 4)]),
        ([0, 0, 1, 1, 0, 0], 5, [([2, 3], [0, 1, 2, 3, 4, 5], 4)]),
        ([1, 1, 1], 3, [([0, 1, 2], [0, 1, 2], 0)]),
        ([1], 1, [([0], [0], 0)]),
        # gap of two between cores, tol=1: each core takes its own neighbour
        ([1, 0, 0, 1], 1, [([0], [0, 1], 1), ([3], [2, 3], 1)]),
        # gap of two, tol=2: both indices contested -> no padding
        ([1, 0, 0, 1], 2, [([0], [0], 0), ([3], [3], 0)]),
    ],
)
def test_edge_runs_padding(mask, tol, expected):
    assert _spans(edge_tolerant_extreme_runs(mask, tol)) == expected


@pytest.mark.parametrize("tol", [None, -3, 0])
def test_edge_runs_nonpositive_tolerance_means_none(tol):
    runs = edge_tolerant_extreme_runs([0, 1, 1, 0], tol)
    assert _spans(runs) == [([1, 2], [1, 2], 0)]


def test_edge_runs_accepts_2d_and_bool_like():
    runs = edge_tolerant_extreme_runs(np.array([[True, False, True]]), 0)
    assert [r["core_indices"] for r in runs] == [[0], [2]]


# ─────────────────────────────────────────────────────────────────────
# pearson_safe
# ─────────────────────────────────────────────────────────────────────
def test_pearson_safe_matches_scipy():
    rng = np.random.default_rng(0)
    x = rng.normal(size=20)
    y = x + rng.normal(scale=0.5, size=20)
    assert pearson_safe(x, y) == pytest.approx(pearsonr(x, y)[0])


def test_pearson_safe_masks_nonfinite():
    x = [1, 2, 3, 4, 5, np.nan, np.inf]
    y = [2, 4, 6, 8, 10, 1, 1]
    assert pearson_safe(x, y) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "x,y,min_n",
    [
        (None, [1, 2, 3], 2),
        ([1, 2, 3], None, 2),
        ([1, 2, 3, 4], [1, 2, 3, 4], 5),  # too few
        ([1, 1, 1, 1, 1], [1, 2, 3, 4, 5], 5),  # constant x
        ([1, 2, 3, 4, 5], [7, 7, 7, 7, 7], 5),  # constant y
        ([np.nan] * 6, [1, 2, 3, 4, 5, 6], 1),
    ],
)
def test_pearson_safe_degenerate_returns_none(x, y, min_n):
    assert pearson_safe(x, y, min_n=min_n) is None


def test_pearson_safe_min_n_boundary():
    assert pearson_safe([1, 2, 4], [1, 3, 2], min_n=3) is not None
    assert pearson_safe([1, 2, 4], [1, 3, 2], min_n=4) is None


def test_pearson_safe_returns_python_float():
    r = pearson_safe([1, 2, 3, 4, 5], [5, 4, 3, 2, 1])
    assert type(r) is float and r == pytest.approx(-1.0)


# ─────────────────────────────────────────────────────────────────────
# check_directional_consistency
# ─────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("block", [np.zeros((0, 0)), np.zeros(5), np.zeros((3, 0))])
def test_directional_empty_or_1d(block):
    res = check_directional_consistency(block, 10, 90)
    assert res["passes"] is False
    assert res["n_variants_evaluated"] == 0
    assert np.isnan(res["block_pass_frac"])
    assert res["dominant_direction"] == "NA"


def test_directional_all_low_passes():
    block = np.full((5, 3), 2.0)
    res = check_directional_consistency(block, 10, 90, 0.9, 1.0, 3)
    assert res["passes"] is True
    assert res["n_variants_evaluated"] == 3
    assert res["n_variants_passing"] == 3
    assert res["block_pass_frac"] == 1.0
    assert res["per_variant_directions"] == ["LOW"] * 3
    assert res["per_variant_agreements"] == [1.0] * 3
    assert res["dominant_direction"] == "LOW"


def test_directional_mixed_columns_dominant_mixed():
    block = np.column_stack([np.full(5, 1.0), np.full(5, 99.0)])
    res = check_directional_consistency(block, 10, 90, 0.9, 1.0, 3)
    assert res["passes"] is True
    assert res["per_variant_directions"] == ["LOW", "HIGH"]
    assert res["dominant_direction"] == "MIXED"


def test_directional_thresholds_inclusive_and_midrange_ignored():
    col = np.array([10.0, 90.0, 50.0, np.nan, 10.0])
    res = check_directional_consistency(col[:, None], 10, 90, 0.5, 1.0, 3)
    # 2 LOW (<=10) and 1 HIGH (>=90); mid/NaN do not vote
    assert res["per_variant_agreements"][0] == pytest.approx(2 / 3)
    assert res["per_variant_directions"] == ["LOW"]
    assert res["passes"] is True


def test_directional_discordant_variant_fails_block():
    good = np.full(10, 1.0)
    bad = np.array([1.0] * 6 + [99.0] * 4)  # agreement 0.6
    res = check_directional_consistency(np.column_stack([good, bad]), 10, 90, 0.85, 1.0, 3)
    assert res["n_variants_evaluated"] == 2
    assert res["n_variants_passing"] == 1
    assert res["block_pass_frac"] == 0.5
    assert res["passes"] is False
    assert res["dominant_direction"] == "LOW"
    # relaxing the block-pass fraction lets it pass
    res2 = check_directional_consistency(np.column_stack([good, bad]), 10, 90, 0.85, 0.5, 3)
    assert res2["passes"] is True


def test_directional_tie_is_evaluated_but_never_passes():
    col = np.array([1.0, 1.0, 99.0, 99.0])
    res = check_directional_consistency(col[:, None], 10, 90, 0.5, 0.5, 3)
    assert res["per_variant_directions"] == ["TIE"]
    assert res["n_variants_evaluated"] == 1
    assert res["n_variants_passing"] == 0
    assert res["passes"] is False
    assert res["dominant_direction"] == "NA"


def test_directional_too_few_extreme_cells_is_NA():
    col = np.array([1.0, 1.0, 50.0, 50.0])
    res = check_directional_consistency(col[:, None], 10, 90, 0.5, 0.5, 3)
    assert res["per_variant_directions"] == ["NA"]
    assert np.isnan(res["per_variant_agreements"][0])
    assert res["n_variants_evaluated"] == 0
    assert res["passes"] is False


# ─────────────────────────────────────────────────────────────────────
# mean_profile
# ─────────────────────────────────────────────────────────────────────
@pytest.fixture
def af_df():
    return pd.DataFrame(
        {"v1": [0.0, 10.0, np.nan], "v2": [100.0, 90.0, np.nan], "v3": [np.nan, np.nan, np.nan]},
        index=["c1", "c2", "c3"],
    )


def test_mean_profile_basic(af_df):
    with np.errstate(all="ignore"), pytest.warns(RuntimeWarning):
        m = mean_profile(af_df, ["c1", "c2"], ["v1", "v2", "v3"])
    assert m[0] == 5.0 and m[1] == 95.0 and np.isnan(m[2])


def test_mean_profile_ignores_unknown_cells(af_df):
    m = mean_profile(af_df, ["c1", "nope"], ["v1", "v2"])
    np.testing.assert_array_equal(m, [0.0, 100.0])


@pytest.mark.parametrize(
    "cells,variants",
    [([], ["v1"]), (["c1"], []), (["nope"], ["v1"])],
)
def test_mean_profile_empty_returns_none(af_df, cells, variants):
    assert mean_profile(af_df, cells, variants) is None


def test_mean_profile_all_nan_returns_none(af_df):
    with pytest.warns(RuntimeWarning):
        assert mean_profile(af_df, ["c3"], ["v1", "v2"]) is None


# ─────────────────────────────────────────────────────────────────────
# fmt_float / fmt_pct / dedupe / vprint
# ─────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "x,nd,expected",
    [
        (1.23456, 3, "1.235"),
        (1, 2, "1.00"),
        ("1.5", 1, "1.5"),
        (np.float64(0.5), 1, "0.5"),
        (None, 3, "NA"),
        (float("nan"), 3, "NA"),
        (float("inf"), 3, "NA"),
        (np.nan, 3, "NA"),
        ("abc", 3, "NA"),
        (pd.NA, 3, "NA"),
    ],
)
def test_fmt_float(x, nd, expected):
    assert fmt_float(x, nd) == expected


@pytest.mark.xfail(
    reason="loh_config.py fmt_float only checks isinstance(x, float) for NaN/Inf, so a numpy "
    "float32 NaN renders as 'nan' instead of 'NA' (docstring promises 'NA' for NaN/Inf).",
    strict=True,
)
def test_fmt_float_numpy_float32_nan():
    assert fmt_float(np.float32("nan")) == "NA"


def test_vprint_respects_verbose(capsys):
    vprint("hello", LOHConfig(verbose=True))
    vprint("quiet", LOHConfig(verbose=False))
    out = capsys.readouterr().out
    assert "hello" in out and "quiet" not in out


# ─────────────────────────────────────────────────────────────────────
# build_cytoband_maps
# ─────────────────────────────────────────────────────────────────────
@pytest.fixture
def cytobands():
    return pd.DataFrame(
        {
            "chromosome arm": ["chr1p", "chr1p", "chr2q"],
            "start": [0, 201, 0],
            "end": [200, 400, 1000],
            "cytoband": ["p36.1", "p35", "q11"],
        }
    )


def test_build_cytoband_maps_basic(cytobands):
    variants = ["chr1:100:A/G", "chr1:200:A/G", "chr1:201-205:A/G", "chr1:999:A/G", "bad"]
    v2b, b2v = build_cytoband_maps(variants, cytobands, "chr1p")
    # boundaries inclusive; out-of-range and unparsable variants skipped
    assert v2b == {0: "p36.1", 1: "p36.1", 2: "p35"}
    assert b2v == {"p36.1": [0, 1], "p35": [2]}
    assert isinstance(b2v, dict) and not hasattr(b2v, "default_factory")


@pytest.mark.parametrize(
    "variants,bands_none,arm",
    [
        ([], False, "chr1p"),
        (["chr1:100:A/G"], True, "chr1p"),
        (["chr1:100:A/G"], False, "chr3p"),  # arm not present
        (["chr1:5000:A/G"], False, "chr1p"),  # no variant inside any band
        (["nonsense"], False, "chr1p"),
    ],
)
def test_build_cytoband_maps_returns_none_pair(cytobands, variants, bands_none, arm):
    bands = None if bands_none else cytobands
    assert build_cytoband_maps(variants, bands, arm) == (None, None)


def test_build_cytoband_maps_empty_frame(cytobands):
    assert build_cytoband_maps(["chr1:100:A/G"], cytobands.iloc[0:0], "chr1p") == (None, None)


# ─────────────────────────────────────────────────────────────────────
# LOHConfig
# ─────────────────────────────────────────────────────────────────────
def test_lohconfig_selected_defaults():
    cfg = LOHConfig()
    assert cfg.low_allele_threshold == 10.0
    assert cfg.high_allele_threshold == 90.0
    assert cfg.low_allele_threshold < cfg.high_allele_threshold
    assert cfg.min_contiguous_variants_initial == 4
    assert cfg.min_final_clone_size == 5
    assert cfg.min_different_amplicons == 3
    assert cfg.block_edge_tolerance == 1
    assert cfg.phase5_small_loh_target_label == lc.SMALL_LABEL
    assert cfg.ado_probability_qc_enabled is False
    assert cfg.amplicon_ado_rates is None
    assert cfg.phase4_arm_rescue_enabled is False
    assert cfg.phase4_arm_expansion_enabled is False
    assert cfg.verbose is True
    assert "NGT" in cfg.phase1_ngt_no_call_values
    assert cfg.phase1_ngt_numeric_no_call_values == (3,)


def test_lohconfig_probability_and_fraction_defaults_in_unit_interval():
    cfg = LOHConfig()
    for f in dataclasses.fields(cfg):
        name = f.name
        val = getattr(cfg, name)
        if isinstance(val, bool) or not isinstance(val, float):
            continue
        if "dynamic_range" in name:  # an AF-percentage range, not a fraction
            continue
        if name.endswith(("_frac", "_fraction", "_agreement", "_corr")) or "pvalue" in name:
            assert 0.0 <= val <= 1.0, name


def test_lohconfig_overrides_and_independent_instances():
    a = LOHConfig(min_cluster_size=7, verbose=False)
    b = LOHConfig()
    assert a.min_cluster_size == 7 and b.min_cluster_size == 5
    a.informative_variants = ["x"]
    assert b.informative_variants is None


def test_lohconfig_rejects_unknown_kwarg():
    with pytest.raises(TypeError):
        LOHConfig(not_a_real_option=1)


def test_lohconfig_no_value_validation():
    # LOHConfig is a plain dataclass: inconsistent values are accepted as-is.
    cfg = LOHConfig(low_allele_threshold=95.0, high_allele_threshold=5.0)
    assert cfg.low_allele_threshold == 95.0


# ─────────────────────────────────────────────────────────────────────
# PipelineState
# ─────────────────────────────────────────────────────────────────────
def test_pipeline_state_basic_mappings():
    st = _make_state()
    assert st.variant_cols == VARS
    assert st.cell_to_barcode == {"AAA": "AAA", "CCC": "CCC", "GGG": "GGG", "TTT": "TTT"}
    assert st.barcode_to_clone["CCC"] == "A LOH-C0 chr1p"
    assert dict(st.clone_to_cells) == {
        "A": ["AAA"], "A LOH-C0 chr1p": ["CCC"], "FP: A C1 chr1p": ["GGG"], "B": ["TTT"],
    }
    assert st.loh_clone_labels() == ["A LOH-C0 chr1p"]
    assert st.loh_cell_count() == 1
    assert st.sample_idx_for_barcode("GGG") == 2
    assert st.sample_idx_for_barcode("nope") is None
    assert st.results == {}
    # no DP available -> all-NaN aligned frame
    assert list(st.df_dp_numeric.columns) == VARS
    assert st.df_dp_numeric.isna().all().all()
    assert arm_variants("chr1p", st) == VARS
    assert arm_variants("chr9q", st) == []


def test_pipeline_state_set_clone_labels_refreshes():
    st = _make_state()
    st.set_clone_labels(pd.Series(["X LOH-C0 chr1p"] * 4))
    assert st.loh_clone_labels() == ["X LOH-C0 chr1p"]
    assert st.loh_cell_count() == 4
    st.set_clone_labels(["A", "B", "A", "B"])
    assert sorted(st.clone_to_cells) == ["A", "B"]
    assert st.loh_cell_count() == 0
    assert list(st.current_clones) == ["A", "B", "A", "B"]


def test_pipeline_state_object_columns_coerced_to_numeric():
    df = pd.DataFrame(
        {VARS[0]: ["1", "x", "3", "4"], VARS[1]: ["5"] * 4, VARS[2]: ["6"] * 4, "Clone": ["A"] * 4},
        index=["AAA", "CCC", "GGG", "TTT"],
    )
    st = _make_state(df=df)
    assert st.variant_cols == VARS  # 'Clone' excluded
    assert st.df_numeric[VARS[0]].tolist()[0] == 1.0
    assert np.isnan(st.df_numeric[VARS[0]].iloc[1])


def test_pipeline_state_no_variant_columns():
    df = pd.DataFrame({"foo": [1, 2, 3, 4]}, index=["AAA", "CCC", "GGG", "TTT"])
    st = _make_state(df=df)
    assert st.variant_cols == []
    assert st.df_numeric.shape == (4, 0)


def test_pipeline_state_integer_index_uses_barcode_column():
    df = pd.DataFrame(np.ones((4, 3)), columns=VARS)
    df["barcode"] = ["TTT", "GGG", "CCC", "AAA"]  # reversed order
    st = _make_state(df=df)
    assert st.cell_to_barcode == {0: "TTT", 1: "GGG", 2: "CCC", 3: "AAA"}
    assert st.clone_to_cells["B"] == [0]


def test_pipeline_state_integer_index_positional_fallback():
    df = pd.DataFrame(np.ones((4, 3)), columns=VARS)
    st = _make_state(df=df)
    assert st.cell_to_barcode == {0: "AAA", 1: "CCC", 2: "GGG", 3: "TTT"}


def test_pipeline_state_dp_aligned_by_index():
    idx = ["AAA", "CCC", "GGG", "TTT"]
    dp = pd.DataFrame(
        {VARS[0]: [1, 2, 3, 4], VARS[1]: ["5", "x", "7", "8"], "extra": 0},
        index=list(reversed(idx)),
    )
    st = _make_state(df_dp=dp)
    assert list(st.df_dp_numeric.index) == idx
    assert list(st.df_dp_numeric.columns) == VARS
    assert st.df_dp_numeric.loc["AAA", VARS[0]] == 4
    assert np.isnan(st.df_dp_numeric.loc["GGG", VARS[1]])
    assert st.df_dp_numeric[VARS[2]].isna().all()


def test_pipeline_state_dp_recovered_from_sample():
    idx = ["AAA", "CCC", "GGG", "TTT"]
    dp = pd.DataFrame({v: [10, 20, 30, 40] for v in VARS}, index=idx)
    st = _make_state(sample_dp=dp)
    assert st.df_dp_numeric.loc["TTT", VARS[1]] == 40


def test_pipeline_state_dp_without_matching_columns():
    dp = pd.DataFrame({"other": [1, 2, 3, 4]}, index=["AAA", "CCC", "GGG", "TTT"])
    st = _make_state(df_dp=dp)
    assert st.df_dp_numeric.isna().all().all()
