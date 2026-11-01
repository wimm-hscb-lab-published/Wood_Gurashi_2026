"""Unit tests for pure helpers in LOH_Pipeline/phase5.py and phase6_subclone_scan.py."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy.stats import binom

from LOH_Pipeline import phase5 as p5
from LOH_Pipeline import phase6_subclone_scan as p6


# ═════════════════════════════ Phase 5 ═════════════════════════════

@pytest.mark.parametrize(
    "label, loh, non_loh, sub",
    [
        ("A LOH-C0 chr5q", True, False, False),
        ("A", False, True, False),
        ("SUB: A C0 chr5q", False, False, True),
        ("FP: A C0 chr5q", False, False, False),
        ("small", False, False, False),
    ],
)
def test_phase5_label_predicates(label, loh, non_loh, sub):
    assert p5._is_active_loh(label) is loh
    assert p5._is_active_non_loh(label) is non_loh
    assert p5._is_subclonal_loh(label) is sub


# ── _contiguous_true_runs ────────────────────────────────────────────

@pytest.mark.parametrize(
    "mask, expected",
    [
        ([], []),
        ([False], []),
        ([True], [[0]]),
        ([True, True, False, True], [[0, 1], [3]]),
        ([False, True, True], [[1, 2]]),
    ],
)
def test_phase5_contiguous_true_runs(mask, expected):
    assert p5._contiguous_true_runs(mask) == expected


# ── _format_variant_block_span ───────────────────────────────────────

VARS = ["chr5:150:A/T", "chr5:250:A/T", "chr5:350:A/T"]
CYTO = pd.DataFrame({
    "chromosome arm": ["chr5q"] * 3,
    "start": [100, 200, 300], "end": [199, 299, 399],
    "cytoband": ["q31", "q32", "q33.1"],
})


def test_format_span_cytoband_range():
    st = SimpleNamespace(cytobands_df=CYTO)
    assert p5._format_variant_block_span(VARS, [0, 1, 2], "chr5q", st) == "q31-q33.1"


def test_format_span_single_band():
    st = SimpleNamespace(cytobands_df=CYTO)
    assert p5._format_variant_block_span(VARS, [1], "chr5q", st) == "q32"


def test_format_span_fallback_to_positions():
    st = SimpleNamespace(cytobands_df=None)
    assert p5._format_variant_block_span(VARS, [0, 2], "chr5q", st) == "150-350"
    # arm without cytobands also falls back
    st2 = SimpleNamespace(cytobands_df=CYTO)
    assert p5._format_variant_block_span(VARS, [0, 1], "chr17p", st2) == "150-250"


def test_format_span_unparseable():
    st = SimpleNamespace()
    assert p5._format_variant_block_span(["bad"], [0], "chr5q", st) == "block"
    assert p5._format_variant_block_span(VARS, [], "chr5q", st) == "block"


# ── _binom_sf_geq ────────────────────────────────────────────────────

@pytest.mark.parametrize("n", [1, 3, 5, 10, 40])
@pytest.mark.parametrize("p", [0.01, 0.1, 0.25, 0.5, 0.9])
def test_binom_sf_geq_matches_scipy(n, p):
    for k in range(1, n + 1):
        assert p5._binom_sf_geq(k, n, p) == pytest.approx(binom.sf(k - 1, n, p), abs=1e-10)


@pytest.mark.parametrize(
    "k, n, p, expected",
    [
        (0, 5, 0.3, 1.0),
        (-2, 5, 0.3, 1.0),
        (6, 5, 0.3, 0.0),
        (1, 0, 0.3, 0.0),
        (2, 5, 0.0, 0.0),
        (2, 5, -1.0, 0.0),
        (2, 5, 1.0, 1.0),
        (2, 5, 1.5, 1.0),
    ],
)
def test_binom_sf_geq_edges(k, n, p, expected):
    assert p5._binom_sf_geq(k, n, p) == expected


def test_binom_sf_geq_is_in_unit_interval_large_n():
    v = p5._binom_sf_geq(3, 2000, 0.001)
    assert 0.0 <= v <= 1.0
    assert v == pytest.approx(binom.sf(2, 2000, 0.001), rel=1e-6)


# ── _base_lineage_size_for_clone ─────────────────────────────────────

def test_base_lineage_size_for_clone():
    st = SimpleNamespace(clone_to_cells={
        "A": ["c1", "c2"], "A LOH-C0 chr5q": ["c3"], "A LOH-C1 chr17p": ["c4", "c5"], "B": ["c6"],
    })
    assert p5._base_lineage_size_for_clone(st, "A LOH-C0 chr5q") == 5
    assert p5._base_lineage_size_for_clone(st, "B") == 1
    assert p5._base_lineage_size_for_clone(st, "missing") == 0


# ── _loh_confidence_from_min_amplicons ───────────────────────────────

@pytest.mark.parametrize(
    "n, cat",
    [(None, ""), ("x", ""), (0, "low"), (3, "low"), ("3", "low"), (4, "mid"), (5, "high"), (12, "high")],
)
def test_loh_confidence_from_min_amplicons(n, cat):
    out_cat, out_msg = p5._loh_confidence_from_min_amplicons(n)
    assert out_cat == cat
    assert (out_msg == "") == (cat == "")


# ═════════════════════════════ Phase 6 ═════════════════════════════

# ── _normalise_clone_labels ──────────────────────────────────────────

def test_normalise_clone_labels_string():
    assert p6._normalise_clone_labels(" A ") == ["A"]


def test_normalise_clone_labels_dedup_order_and_strip():
    assert p6._normalise_clone_labels(["B", "A", " B", "", "A"]) == ["B", "A"]


def test_normalise_clone_labels_none_raises():
    with pytest.raises(ValueError, match="required"):
        p6._normalise_clone_labels(None)


@pytest.mark.parametrize("bad", [[], ["", "  "], ""])
def test_normalise_clone_labels_empty_raises(bad):
    with pytest.raises(ValueError, match="usable"):
        p6._normalise_clone_labels(bad)


# ── _eligible_parent_label ───────────────────────────────────────────

@pytest.mark.parametrize(
    "label, expected",
    [
        ("TP53_R248W_Hem", (True, "eligible")),
        ("", (False, "empty_label")),
        ("   ", (False, "empty_label")),
        ("ADO", (False, "ado_label")),
        ("small", (False, "small_label")),
        ("FP: A C0 chr5q", (False, "possible_loh_fp_label")),
        ("SUB: A C0 chr5q", (False, "subclonal_loh_sub_label")),
        ("A LOH-C0 chr5q", (False, "already_active_loh")),
    ],
)
def test_eligible_parent_label(label, expected):
    assert p6._eligible_parent_label(label) == expected


# ── _variant_position / _variant_to_amplicon ─────────────────────────

@pytest.mark.parametrize(
    "v, expected",
    [("chr1:500:A/T", 500), ("chr1:100-200:A/T", 100), ("x", 10**18), (None, 10**18)],
)
def test_variant_position(v, expected):
    assert p6._variant_position(v) == expected


def test_variant_to_amplicon():
    st = SimpleNamespace(Sample=SimpleNamespace(dna=SimpleNamespace(
        col_attrs={"id": ["v1", "v2"], "amplicon": ["a", "b"]})))
    assert p6._variant_to_amplicon(st) == {"v1": "a", "v2": "b"}
    st_bad = SimpleNamespace(Sample=SimpleNamespace(dna=SimpleNamespace(
        col_attrs={"id": ["v1"], "amplicon": []})))
    assert p6._variant_to_amplicon(st_bad) == {}
    assert p6._variant_to_amplicon(SimpleNamespace()) == {}


# ── _remove_arm_token ────────────────────────────────────────────────

@pytest.mark.parametrize(
    "label, arm, expected",
    [
        ("A LOH-C0 chr5q", "chr5q", "A"),
        ("A LOH-C0 chr5q:q31-q33 LOH-C1 chr17p", "chr5q", "A LOH-C1 chr17p"),
        ("A LOH-C0 chr5q LOH-C1 chr17p", "chr17p", "A LOH-C0 chr5q"),
        ("A LOH-C0 chr5q", "chr17p", "A LOH-C0 chr5q"),
        ("A LOH-C0 chr17p", "chr1p", "A LOH-C0 chr17p"),
        ("A", "chr5q", "A"),
    ],
)
def test_remove_arm_token(label, arm, expected):
    assert p6._remove_arm_token(label, arm) == expected


# ── _append_semicolon_token ──────────────────────────────────────────

@pytest.mark.parametrize(
    "value, token, expected",
    [
        ("", "chr5q", "chr5q"),
        (None, "chr5q", "chr5q"),
        ("chr5q", "chr5q", "chr5q"),
        ("chr5q", "chr17p", "chr5q;chr17p"),
        (" chr5q ; ;chr7p", " chr17p ", "chr5q;chr7p;chr17p"),
        ("chr5q", "", "chr5q"),
    ],
)
def test_append_semicolon_token(value, token, expected):
    assert p6._append_semicolon_token(value, token) == expected


# ── _safe_profile_similarity ─────────────────────────────────────────

def test_safe_profile_similarity_correlated():
    x = np.array([0, 50, 100, 25.0])
    r, d, n = p6._safe_profile_similarity(x, x + 2, min_variants=3, constant_profile_max_abs_delta=15)
    assert r == pytest.approx(1.0)
    assert d == pytest.approx(2.0)
    assert n == 4


def test_safe_profile_similarity_too_few():
    x = np.array([0, np.nan, 100.0])
    r, d, n = p6._safe_profile_similarity(x, x, min_variants=3, constant_profile_max_abs_delta=15)
    assert np.isnan(r) and np.isnan(d) and n == 2


def test_safe_profile_similarity_constant():
    x = np.zeros(4)
    r, d, n = p6._safe_profile_similarity(x, x + 5, min_variants=3, constant_profile_max_abs_delta=15)
    assert (r, d, n) == (1.0, 5.0, 4)
    r, d, _ = p6._safe_profile_similarity(x, x + 50, min_variants=3, constant_profile_max_abs_delta=15)
    assert (r, d) == (0.0, 50.0)


# ── contiguous run helpers ───────────────────────────────────────────

@pytest.mark.parametrize(
    "mask, expected",
    [
        ([], []),
        ([False, False], []),
        ([True, True], [(0, 1)]),
        ([True, False, True, True, False], [(0, 0), (2, 3)]),
    ],
)
def test_phase6_contiguous_runs(mask, expected):
    arr = np.array(mask, dtype=bool)
    assert p6._contiguous_runs_from_mask(arr) == expected
    assert p6._contiguous_true_runs(arr) == expected


# ── row-attr helpers on a fake state ─────────────────────────────────

def _row_state(clones, extra=None):
    row_attrs = {"Clone": np.asarray(clones, dtype=object),
                 "barcode": np.asarray([f"bc{i}" for i in range(len(clones))], dtype=object)}
    row_attrs.update(extra or {})
    bc2idx = {f"bc{i}": i for i in range(len(clones))}
    return SimpleNamespace(
        Sample=SimpleNamespace(dna=SimpleNamespace(row_attrs=row_attrs)),
        cell_to_barcode={f"cell{i}": f"bc{i}" for i in range(len(clones))},
        sample_idx_for_barcode=lambda bc: bc2idx.get(str(bc)),
    )


def test_sample_row_attr_array_missing_and_misaligned():
    st = _row_state(["A", "B", "C"], extra={"bad": np.array(["x"])})
    np.testing.assert_array_equal(p6._sample_row_attr_array(st, "nope", fill_value="-"),
                                  np.array(["-", "-", "-"], dtype=object))
    np.testing.assert_array_equal(p6._sample_row_attr_array(st, "bad"),
                                  np.array(["", "", ""], dtype=object))


def test_sample_row_attr_array_returns_copy():
    st = _row_state(["A", "B"], extra={"k": np.array(["x", "y"], dtype=object)})
    arr = p6._sample_row_attr_array(st, "k")
    arr[0] = "changed"
    assert st.Sample.dna.row_attrs["k"][0] == "x"


def test_record_secondary_arm_evidence():
    st = _row_state(["A", "B", "C"])
    p6._record_phase6_secondary_arm_evidence(st, ["cell0", "cell2", "unknown_cell"], "chr5q")
    p6._record_phase6_secondary_arm_evidence(st, ["cell0"], "chr17p")
    p6._record_phase6_secondary_arm_evidence(st, ["cell0"], "chr5q")
    out = list(st.Sample.dna.row_attrs["phase6_secondary_loh_arms"])
    assert out == ["chr5q;chr17p", "", "chr5q"]


def test_set_primary_metadata():
    st = _row_state(["A", "B"])
    p6._set_phase6_primary_metadata(st, ["cell1"], child_label="A LOH-C0 chr5q",
                                    chr_arm="chr5q", member_role="seed")
    ra = st.Sample.dna.row_attrs
    assert list(ra["phase6_primary_loh_child"]) == ["", "A LOH-C0 chr5q"]
    assert list(ra["phase6_primary_loh_arm"]) == ["", "chr5q"]
    assert list(ra["phase6_primary_loh_member_role"]) == ["", "seed"]


# ── _next_phase6_child_number ────────────────────────────────────────

def test_next_phase6_child_number():
    st = _row_state(["A", "A LOH-C2 chr5q", "B LOH-C9 chr1p", "A LOH-C0 chr7p LOH-C4 chr17p"])
    assert p6._next_phase6_child_number(st, "A") == 5
    assert p6._next_phase6_child_number(st, "B") == 10
    assert p6._next_phase6_child_number(st, "C") == 0
