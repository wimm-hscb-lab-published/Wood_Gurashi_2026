"""Unit tests for pure helpers in LOH_Pipeline/phase4.py."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from LOH_Pipeline import phase4 as p4
from LOH_Pipeline.loh_config import LOHConfig


# ── label classification ─────────────────────────────────────────────

@pytest.mark.parametrize(
    "label, loh, non_loh",
    [
        ("A LOH-C0 chr5q", True, False),
        ("A", False, True),
        ("WT", False, True),
        ("FP: A C0 chr5q", False, False),
        ("SUB: A C0 chr5q", False, False),
        ("small", False, False),
        ("small: A LOH-C0 chr5q", False, False),
    ],
)
def test_is_active_loh_and_non_loh(label, loh, non_loh):
    assert p4._is_active_loh(label) is loh
    assert p4._is_active_non_loh(label) is non_loh


# ── _get_base_name ───────────────────────────────────────────────────

@pytest.mark.parametrize(
    "label, expected",
    [
        ("TP53_E204Ter_Hem_BCORL1_T907Lfs18_Hom LOH-C0 chr5q",
         "TP53_E204Ter_Hem_BCORL1_T907Lfs18_Hom"),
        ("A LOH-C0 chr5q LOH-C1 chr17p:p13.1", "A"),
        ("A", "A"),
        ("", ""),
        ("A LOH-C0", "A"),  # trailing LOH token without arm
        ("A B LOH-C2 chr1p", "A B"),
    ],
)
def test_get_base_name(label, expected):
    assert p4._get_base_name(label) == expected


def test_get_base_name_custom_tag():
    assert p4._get_base_name("A X-C0 chr5q", loh_tag="X") == "A"
    assert p4._get_base_name("A LOH-C0 chr5q", loh_tag="X") == "A LOH-C0 chr5q"


# ── _zygosity_normalized_base_name / _has_zygosity_tag ───────────────

@pytest.mark.parametrize(
    "label, expected",
    [
        ("TP53_R248W_Het", "TP53_R248W_ZYG"),
        ("TP53_R248W_Hem", "TP53_R248W_ZYG"),
        ("TP53_R248W_hom LOH-C0 chr17p", "TP53_R248W_ZYG"),
        ("TP53_R248W_Het_DNMT3A_X_Hom", "TP53_R248W_ZYG_DNMT3A_X_ZYG"),
        ("Het_TP53", "Het_TP53"),  # needs a leading underscore
        ("TP53_Hetero", "TP53_Hetero"),  # must end at '_' or string end
        ("WT", "WT"),
    ],
)
def test_zygosity_normalized_base_name(label, expected):
    assert p4._zygosity_normalized_base_name(label) == expected


@pytest.mark.parametrize(
    "label, tags, expected",
    [
        ("TP53_R248W_Het", ("Het",), True),
        ("TP53_R248W_Het", ("Hem", "Hom"), False),
        ("TP53_R248W_HEM LOH-C0 chr17p", ("Hem",), True),
        ("WT", ("Het",), False),
    ],
)
def test_has_zygosity_tag(label, tags, expected):
    assert p4._has_zygosity_tag(label, tags) is expected


# ── _base_names_compatible ───────────────────────────────────────────

@pytest.mark.parametrize(
    "src, tgt, allow, expected",
    [
        ("TP53_R248W_Hem", "TP53_R248W_Hem LOH-C0 chr17p", False, (True, "exact")),
        ("TP53_R248W_Het", "TP53_R248W_Hem", False, (False, "different_base")),
        ("TP53_R248W_Het", "TP53_R248W_Hem", True, (True, "zygosity_normalized_het_to_hem")),
        ("TP53_R248W_Het", "TP53_R248W_Hom LOH-C1 chr17p", True,
         (True, "zygosity_normalized_het_to_hem")),
        # Reverse direction is not allowed
        ("TP53_R248W_Hem", "TP53_R248W_Het", True, (False, "different_base")),
        # Different mutation never compatible
        ("TP53_R248W_Het", "TP53_R175H_Hem", True, (False, "different_base")),
        ("A", "B", True, (False, "different_base")),
    ],
)
def test_base_names_compatible(src, tgt, allow, expected):
    assert p4._base_names_compatible(src, tgt, allow_het_to_hem=allow) == expected


# ── _extract_mutation ────────────────────────────────────────────────

# ── _variant_pos / _build_variant_to_amplicon ────────────────────────

@pytest.mark.parametrize(
    "v, expected",
    [
        ("chr5:12345:A/G", 12345),
        ("chr1:100-200:A/T", 100),  # phase4 handles range-style ids
        ("bad", 10**18),
        (None, 10**18),
    ],
)
def test_variant_pos(v, expected):
    assert p4._variant_pos(v) == expected


def test_build_variant_to_amplicon_merges_and_falls_back():
    sample = SimpleNamespace(dna=SimpleNamespace(col_attrs={"id": ["v1"], "amplicon": ["A1"]}))
    st = SimpleNamespace(Sample=sample, df=pd.DataFrame(columns=["v1", "v2"]))
    assert p4._build_variant_to_amplicon(st) == {"v1": "A1", "v2": "v2"}


def test_build_variant_to_amplicon_nothing():
    assert p4._build_variant_to_amplicon(SimpleNamespace()) == {}


# ── _loh_block_variants_for_label ────────────────────────────────────

def _block_state(block_map=None, cytobands=None, grouped=None, cols=()):
    cols = list(cols)
    return SimpleNamespace(
        clone_block_variants=block_map,
        cytobands_df=cytobands,
        grouped_by_chr_arm=grouped or {},
        df=pd.DataFrame(columns=cols),
        df_numeric=pd.DataFrame(columns=cols),
    )


def test_block_variants_exact_key_filtered_to_df_columns():
    st = _block_state({"L": ["v1", "v2", "gone"]}, cols=["v1", "v2"])
    assert p4._loh_block_variants_for_label("L", st) == ["v1", "v2"]


def test_block_variants_embedded_original_label():
    st = _block_state({"A LOH-C0 chr5q": ["v1"]}, cols=["v1"])
    assert p4._loh_block_variants_for_label("A LOH-C0 chr5q LOH-C3 chr17p", st) == ["v1"]


def test_block_variants_arm_restriction():
    st = _block_state(
        {"L": ["v1", "v2"]},
        grouped={"chr5q": ["v1"], "chr17p": ["v2"]},
        cols=["v1", "v2"],
    )
    assert p4._loh_block_variants_for_label("L", st, arm="chr17p") == ["v2"]


def test_block_variants_none_when_nothing_found():
    st = _block_state({}, cols=["v1"])
    assert p4._loh_block_variants_for_label("L", st) is None


def test_block_variants_cytoband_fallback():
    cyto = pd.DataFrame({
        "chromosome arm": ["chr5q", "chr5q"],
        "start": [100, 200], "end": [199, 299], "cytoband": ["q31", "q32"],
    })
    vs = ["chr5:150:A/T", "chr5:250:A/T"]
    st = _block_state({}, cytobands=cyto, grouped={"chr5q": vs}, cols=vs)
    assert p4._loh_block_variants_for_label("A LOH-C0 chr5q:q32", st) == [vs[1]]
    assert p4._loh_block_variants_for_label("A LOH-C0 chr5q:q32", st, arm="chr17p") is None


def test_rescue_variants_falls_back_to_arm_variants():
    st = _block_state({"L": ["v1"]}, cols=["v1"])  # only 1 block variant (<2)
    cfg = SimpleNamespace(phase4_rescue_use_loh_block_variants=True)
    out = p4._rescue_variants_for_clone_arm("L", "chr5q", st, cfg, lambda arm, s: ["a", "b"])
    assert out == ["a", "b"]


def test_rescue_variants_uses_block_when_big_enough():
    st = _block_state({"L": ["v1", "v2"]}, cols=["v1", "v2"])
    cfg = SimpleNamespace(phase4_rescue_use_loh_block_variants=True)
    out = p4._rescue_variants_for_clone_arm("L", None, st, cfg, lambda arm, s: ["a"])
    assert out == ["v1", "v2"]
    cfg_off = SimpleNamespace(phase4_rescue_use_loh_block_variants=False)
    assert p4._rescue_variants_for_clone_arm("L", None, st, cfg_off, lambda a, s: ["a"]) == ["a"]


# ── _allowed_rescue_ado_amplicons ────────────────────────────────────

@pytest.mark.parametrize("n, expected", [(-3, 0), (0, 0), (3, 0), (4, 0), (5, 1), (20, 1)])
def test_allowed_rescue_ado_defaults(n, expected):
    assert p4._allowed_rescue_ado_amplicons(n, LOHConfig(verbose=False)) == expected


@pytest.mark.parametrize("n, expected", [(5, 1), (10, 1), (11, 2), (25, 3)])
def test_allowed_rescue_ado_expected_rate(n, expected):
    cfg = SimpleNamespace(
        phase4_rescue_min_amplicons_to_allow_ado=5,
        phase4_rescue_max_ado_amplicons=None,
        phase4_rescue_expected_amplicon_ado=0.10,
    )
    assert p4._allowed_rescue_ado_amplicons(n, cfg) == expected


def test_allowed_rescue_ado_capped_at_n_minus_one():
    cfg = SimpleNamespace(
        phase4_rescue_min_amplicons_to_allow_ado=1,
        phase4_rescue_max_ado_amplicons=10,
    )
    assert p4._allowed_rescue_ado_amplicons(3, cfg) == 2
    assert p4._allowed_rescue_ado_amplicons(1, cfg) == 0


def test_allowed_rescue_ado_negative_explicit_clamped():
    cfg = SimpleNamespace(phase4_rescue_min_amplicons_to_allow_ado=1,
                          phase4_rescue_max_ado_amplicons=-4)
    assert p4._allowed_rescue_ado_amplicons(10, cfg) == 0


# ── _informative_variant_set_for_rescue ──────────────────────────────

@pytest.mark.parametrize(
    "vals, expected",
    [(None, None), ([], None), (["a", 1], {"a", "1"}), (7, None)],
)
def test_informative_variant_set_for_rescue(vals, expected):
    assert p4._informative_variant_set_for_rescue(SimpleNamespace(informative_variants=vals)) == expected


# ── _direction ───────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "vals, expected",
    [
        ([], "NA"),
        ([np.nan, np.inf], "NA"),
        ([50.0], "MID"),
        ([5.0, 50.0], "LOW"),
        ([10.0], "LOW"),  # boundary inclusive
        ([90.0, 50.0], "HIGH"),  # boundary inclusive
        ([0.0, 100.0], "MIXED"),
        ([np.nan, 2.0], "LOW"),
    ],
)
def test_direction(vals, expected):
    assert p4._direction(np.array(vals, dtype=float), 10.0, 90.0) == expected


# ── _safe_similarity_corr ────────────────────────────────────────────

def test_safe_similarity_corr_pearson():
    x = np.array([0, 10, 20, 30.0])
    assert p4._safe_similarity_corr(x, 2 * x + 1, constant_profile_max_mad=5) == pytest.approx(1.0)
    assert p4._safe_similarity_corr(x, -x, constant_profile_max_mad=5) == pytest.approx(-1.0)


def test_safe_similarity_corr_too_few():
    assert np.isnan(p4._safe_similarity_corr([1.0], [1.0], constant_profile_max_mad=5))
    assert np.isnan(p4._safe_similarity_corr([1.0, np.nan], [1.0, 2.0], constant_profile_max_mad=5))


def test_safe_similarity_corr_constant_profiles():
    x = np.zeros(4)
    y = np.array([1.0, 2.0, 1.0, 2.0])
    assert p4._safe_similarity_corr(x, y, constant_profile_max_mad=5) == 1.0
    assert np.isnan(p4._safe_similarity_corr(x, y + 50, constant_profile_max_mad=5))
    assert np.isnan(p4._safe_similarity_corr(x, y, constant_profile_max_mad=5,
                                             constant_profiles_are_match=False))


# ── _profile_extreme_fraction ────────────────────────────────────────

@pytest.mark.parametrize(
    "vals, expected",
    [
        ([0, 50, 100, 50], 0.5),
        ([10, 90], 1.0),
        ([11, 89], 0.0),
        ([np.nan, 0], 1.0),
    ],
)
def test_profile_extreme_fraction(vals, expected):
    assert p4._profile_extreme_fraction(np.array(vals, float), 10, 90) == pytest.approx(expected)


def test_profile_extreme_fraction_empty():
    assert np.isnan(p4._profile_extreme_fraction(np.array([]), 10, 90))
    assert np.isnan(p4._profile_extreme_fraction(np.array([np.nan]), 10, 90))


# ── _directions_compatible ───────────────────────────────────────────

@pytest.mark.parametrize(
    "a, b, expected",
    [
        ("LOW", "LOW", True),
        ("HIGH", "HIGH", True),
        ("LOW", "HIGH", False),
        ("MIXED", "LOW", True),
        ("HIGH", "MIXED", True),
        ("MIXED", "MIXED", True),
        ("MID", "MID", True),
        ("MID", "LOW", False),
        ("MID", "MIXED", True),
        ("NA", "NA", False),
        ("NA", "MIXED", False),
        ("LOW", "NA", False),
    ],
)
def test_directions_compatible(a, b, expected):
    assert p4._directions_compatible(a, b) is expected
    assert p4._directions_compatible(b, a) is expected


# ── _next_residual_loh_number ────────────────────────────────────────

@pytest.mark.parametrize(
    "labels, expected",
    [([], 0), (["A"], 0), (["A LOH-C4 chr5q", "B LOH-C1 chr1p LOH-C9 chr2q"], 10)],
)
def test_next_residual_loh_number(labels, expected):
    st = SimpleNamespace(clone_to_cells={lbl: [] for lbl in labels})
    assert p4._next_residual_loh_number(st) == expected


# ── _contiguous_runs ─────────────────────────────────────────────────

@pytest.mark.parametrize(
    "mask, expected",
    [
        ([], []),
        ([False, False], []),
        ([True], [(0, 0)]),
        ([True, True, False, True], [(0, 1), (3, 3)]),
        ([False, True, True, True], [(1, 3)]),
        ([[True, False], [True, True]], [(0, 0), (2, 3)]),  # raveled
    ],
)
def test_contiguous_runs(mask, expected):
    assert p4._contiguous_runs(np.array(mask, dtype=bool)) == expected


# ── _residual_component_homogeneous ──────────────────────────────────

def test_residual_component_insufficient():
    ok, info = p4._residual_component_homogeneous(
        np.zeros((1, 5)), min_variants=4, min_mean_r=0.8, max_mad=15)
    assert not ok and info["reason"] == "insufficient_cells_or_variants"
    ok, info = p4._residual_component_homogeneous(
        np.zeros((3, 3)), min_variants=4, min_mean_r=0.8, max_mad=15)
    assert not ok and info["n_usable_r"] == 0
    ok, _ = p4._residual_component_homogeneous(
        np.zeros(5), min_variants=4, min_mean_r=0.8, max_mad=15)
    assert not ok


def test_residual_component_correlated_passes():
    base = np.array([0, 100, 0, 100, 50.0])
    X = np.vstack([base, base + 1, base - 1])
    ok, info = p4._residual_component_homogeneous(X, min_variants=4, min_mean_r=0.8, max_mad=15)
    assert ok
    assert info["reason"] == "pass"
    assert info["n_usable_r"] == 3
    assert info["mean_r"] > 0.99


def test_residual_component_anticorrelated_fails():
    base = np.array([0, 100, 0, 100.0])
    X = np.vstack([base, base, 100 - base])
    ok, info = p4._residual_component_homogeneous(X, min_variants=4, min_mean_r=0.8, max_mad=15)
    assert not ok
    assert info["reason"] == "mean_correlation_or_mad"


def test_residual_component_constant_profile_distance():
    X = np.zeros((3, 4))
    ok, info = p4._residual_component_homogeneous(X, min_variants=4, min_mean_r=0.8, max_mad=15)
    assert ok
    assert info["reason"] == "constant_profile_distance_pass"
    assert info["median_abs_diff"] == 0.0


# ── _existing_exact_arm_target / _final_residual_lineage_size ────────

def test_existing_exact_arm_target():
    st = SimpleNamespace(clone_to_cells={
        "A LOH-C0 chr17p LOH-C1 chr5q": ["c1"],
        "A LOH-C2 chr5q": ["c2"],
        "B LOH-C3 chr5q": ["c3"],
        "A LOH-C4 chr7p": [],  # empty -> ignored
    })
    assert p4._existing_exact_arm_target(base_label="A", arms=("chr5q", "chr17p"), state=st) == \
        "A LOH-C0 chr17p LOH-C1 chr5q"
    assert p4._existing_exact_arm_target(base_label="A", arms=("chr5q",), state=st) == "A LOH-C2 chr5q"
    assert p4._existing_exact_arm_target(base_label="A", arms=("chr7p",), state=st) is None


def test_existing_exact_arm_target_ambiguous():
    st = SimpleNamespace(clone_to_cells={"A LOH-C0 chr5q": ["c1"], "A LOH-C1 chr5q:q31": ["c2"]})
    assert p4._existing_exact_arm_target(base_label="A", arms=("chr5q",), state=st) is None


def test_final_residual_lineage_size():
    st = SimpleNamespace(clone_to_cells={
        "A": ["c1", "c2"], "A LOH-C0 chr5q": ["c3"], "B": ["c4"], "FP: A C1 chr5q": ["c5"],
    })
    # FP label's base is "FP: A C1 chr5q" under get_base_clone, so not counted.
    assert p4._final_residual_lineage_size(st, "A") == 3
    assert p4._final_residual_lineage_size(st, "Z") == 0


# ── _final_residual_direction_signature ──────────────────────────────

_SIG_KW = dict(min_variants=2, low=10, high=90, min_per_variant_agreement=0.9,
               min_block_pass_frac=1.0, min_extreme_cells_per_variant=2)


def test_direction_signature_consistent():
    X = np.array([[0, 100, 0], [5, 95, 2], [1, 99, 50]], float)
    ok, info = p4._final_residual_direction_signature(X, **_SIG_KW)
    assert ok
    assert info["n_evaluated"] == 3 and info["n_passing"] == 3
    assert info["block_pass_frac"] == 1.0


def test_direction_signature_inconsistent():
    X = np.array([[0, 100], [100, 0], [0, 100]], float)
    ok, info = p4._final_residual_direction_signature(X, **_SIG_KW)
    assert not ok
    assert info["reason"] == "directional_signature_inconsistent"
    assert info["n_passing"] == 0
    assert info["per_variant_agreements"] == pytest.approx([2 / 3, 2 / 3])


def test_direction_signature_insufficient():
    ok, info = p4._final_residual_direction_signature(np.zeros((1, 3)), **_SIG_KW)
    assert not ok and info["reason"] == "insufficient_cells_or_variants"
    assert np.isnan(info["block_pass_frac"])


def test_direction_signature_no_extreme_cells():
    X = np.full((3, 3), 50.0)
    ok, info = p4._final_residual_direction_signature(X, **_SIG_KW)
    assert not ok
    assert info["n_evaluated"] == 0 and np.isnan(info["block_pass_frac"])
