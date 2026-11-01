"""Unit tests for pure helpers in LOH_Pipeline/phase3.py."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from LOH_Pipeline import phase3 as p3


def _state(clone_to_cells=None, grouped=None, columns=(), **kw):
    return SimpleNamespace(
        clone_to_cells=clone_to_cells or {},
        grouped_by_chr_arm=grouped or {},
        df_numeric=pd.DataFrame(columns=list(columns)),
        **kw,
    )


# ── _loh_clone_map ───────────────────────────────────────────────────

def test_loh_clone_map_keeps_only_active_loh():
    st = _state({
        "A LOH-C0 chr5q": ["c1"],
        "A": ["c2"],
        "FP: A C1 chr5q": ["c3"],
        "SUB: A C2 chr5q": ["c4"],
        "small": ["c5"],
    })
    assert p3._loh_clone_map(st) == {"A LOH-C0 chr5q": ["c1"]}


# ── _loh_arm_variants ────────────────────────────────────────────────

def test_loh_arm_variants_unique_and_filtered_to_columns():
    st = _state(
        grouped={"chr5q": ["v1", "v2", "vX"], "chr17p": ["v3", "v1"]},
        columns=["v1", "v2", "v3"],
    )
    out = p3._loh_arm_variants("A LOH-C0 chr5q LOH-C1 chr17p", st)
    assert out == ["v1", "v2", "v3"]


def test_loh_arm_variants_no_arms():
    st = _state(grouped={"chr5q": ["v1"]}, columns=["v1"])
    assert p3._loh_arm_variants("A", st) == []


# ── _loh_block_variants ──────────────────────────────────────────────

CYTO = pd.DataFrame({
    "chromosome arm": ["chr5q", "chr5q", "chr5q"],
    "start": [100, 200, 300],
    "end": [199, 299, 399],
    "cytoband": ["q31", "q32", "q33.1"],
})
V5 = ["chr5:150:A/T", "chr5:250:A/T", "chr5:350:A/T"]


def test_loh_block_variants_prefers_persisted_block():
    st = _state(clone_block_variants={"L": ["a", "b"]}, cytobands_df=None)
    assert p3._loh_block_variants("L", st) == ["a", "b"]


def test_loh_block_variants_none_without_cytobands():
    st = _state(clone_block_variants={}, cytobands_df=None)
    assert p3._loh_block_variants("A LOH-C0 chr5q:q32", st) is None


def test_loh_block_variants_from_cytoband_range():
    st = _state(grouped={"chr5q": V5}, columns=V5, cytobands_df=CYTO)
    assert p3._loh_block_variants("A LOH-C0 chr5q:q32-q33.1", st) == V5[1:]


def test_loh_block_variants_single_band():
    st = _state(grouped={"chr5q": V5}, columns=V5, cytobands_df=CYTO)
    assert p3._loh_block_variants("A LOH-C0 chr5q:q31", st) == V5[:1]


def test_loh_block_variants_unannotated_label_returns_none():
    st = _state(grouped={"chr5q": V5}, columns=V5, cytobands_df=CYTO)
    assert p3._loh_block_variants("A LOH-C0 chr5q", st) is None


# ── _variant_to_amplicon_map ─────────────────────────────────────────

def test_variant_to_amplicon_map_falls_back_to_variant_id():
    sample = SimpleNamespace(dna=SimpleNamespace(col_attrs={"id": ["v1"], "amplicon": ["A1"]}))
    st = _state(columns=["v1", "v2"], Sample=sample)
    assert p3._variant_to_amplicon_map(st) == {"v1": "A1", "v2": "v2"}


def test_variant_to_amplicon_map_no_sample():
    st = _state(columns=["v1"])
    assert p3._variant_to_amplicon_map(st) == {"v1": "v1"}


# ── _directional_summary ─────────────────────────────────────────────

def test_directional_summary_not_applied():
    assert p3._directional_summary({"multi_arm_directional_mode": "not_applied"}) == \
        "dir=not_applied_multi_arm"


@pytest.mark.parametrize("metrics", [{}, {"dir_res": None}, {"dir_res": {"n_variants_evaluated": 0}}])
def test_directional_summary_not_evaluated(metrics):
    assert p3._directional_summary(metrics) == "dir=not_evaluated"


def test_directional_summary_single_arm():
    m = {"dir_res": {"n_variants_evaluated": 4, "n_variants_passing": 3,
                     "block_pass_frac": 0.75, "dominant_direction": "LOW"}}
    assert p3._directional_summary(m) == "dir=LOW (3/4=75%)"


def test_directional_summary_single_arm_ado_and_nan_frac():
    m = {"dir_ado_tolerated": True,
         "dir_res": {"n_variants_evaluated": 4, "n_variants_passing": 4,
                     "block_pass_frac": np.nan}}
    assert p3._directional_summary(m) == "dir=NA (4/4=NA); ADO-tolerated"


def test_directional_summary_per_arm():
    m = {"per_arm_metrics": {
        "chr5q": {"dir_res": {"n_variants_evaluated": 2, "n_variants_passing": 2,
                              "block_pass_frac": 1.0, "dominant_direction": "HIGH"},
                  "dir_ado_tolerated": True},
        "chr17p": {"dir_res": None},
    }}
    assert p3._directional_summary(m) == \
        "dir=per-arm[chr5q:HIGH (2/2=100%);ADO-tolerated | chr17p:not_evaluated]"


# ── _next_loh_clone_number ───────────────────────────────────────────

@pytest.mark.parametrize(
    "labels, expected",
    [
        ([], 0),
        (["A", "B"], 0),
        (["A LOH-C0 chr5q"], 1),
        (["A LOH-C3 chr5q LOH-C7 chr17p", "B LOH-C2 chr1p"], 8),
        (["FP: A C9 chr5q"], 0),  # FP tail uses C<n>, not LOH-C<n>
    ],
)
def test_next_loh_clone_number(labels, expected):
    st = _state({lbl: [] for lbl in labels})
    assert p3._next_loh_clone_number(st) == expected


# ── _multi_arm_loh_tokens ────────────────────────────────────────────

def test_multi_arm_loh_tokens_order_and_cytoband():
    out = p3._multi_arm_loh_tokens("TP53 LOH-C0 chr5q LOH-C1 chr17p:p13.1-p13.3")
    assert out == [("chr5q", "LOH-C0 chr5q"), ("chr17p", "LOH-C1 chr17p:p13.1-p13.3")]


def test_multi_arm_loh_tokens_dedupes_arms():
    out = p3._multi_arm_loh_tokens("A LOH-C0 chr5q LOH-C2 chr5q")
    assert out == [("chr5q", "LOH-C0 chr5q")]


@pytest.mark.parametrize("label", ["A", "", "FP: A C0 chr5q"])
def test_multi_arm_loh_tokens_none(label):
    assert p3._multi_arm_loh_tokens(label) == []


def test_multi_arm_loh_tokens_sex_chromosome():
    assert p3._multi_arm_loh_tokens("A LOH-C4 chrXp") == [("chrXp", "LOH-C4 chrXp")]


# ── _longest_contiguous_variant_run ──────────────────────────────────

ORDER = ["v0", "v1", "v2", "v3", "v4", "v5", "v6"]


@pytest.mark.parametrize(
    "variants, expected",
    [
        ([], []),
        (["zz"], []),
        (["v3"], ["v3"]),
        (["v0", "v1", "v3", "v4", "v5"], ["v3", "v4", "v5"]),
        (["v5", "v4", "v3", "v0"], ["v3", "v4", "v5"]),  # order follows arm_order
        (["v0", "v1", "v3", "v4"], ["v0", "v1"]),  # tie -> first run wins
        (ORDER, ORDER),
    ],
)
def test_longest_contiguous_variant_run(variants, expected):
    assert p3._longest_contiguous_variant_run(variants, ORDER) == expected


# ── _fp_parent_label_info ────────────────────────────────────────────

def test_fp_parent_label_info_single_arm():
    base, parent, arms = p3._fp_parent_label_info("FP: TP53_R248W_Hem C0 chr7p:p22.1-p14.1")
    assert base == "TP53_R248W_Hem"
    assert parent == "TP53_R248W_Hem LOH-C0 chr7p:p22.1-p14.1"
    assert arms == ("chr7p",)


def test_fp_parent_label_info_multi_arm_sorted():
    base, parent, arms = p3._fp_parent_label_info("FP: A_Het C3 chr17p C1 chr5q")
    assert base == "A_Het"
    assert parent == "A_Het LOH-C3 chr17p LOH-C1 chr5q"
    assert arms == ("chr5q", "chr17p")


def test_fp_parent_label_info_no_arms():
    assert p3._fp_parent_label_info("FP: A_Het") == ("A_Het", "A_Het", ())


# ── _active_loh_label_info ───────────────────────────────────────────

def test_active_loh_label_info():
    assert p3._active_loh_label_info("A LOH-C1 chr17p LOH-C0 chr5q") == ("A", ("chr5q", "chr17p"))


def test_active_loh_label_info_dedupes():
    assert p3._active_loh_label_info("A LOH-C1 chr5q LOH-C2 chr5q:q31") == ("A", ("chr5q",))


def test_active_loh_label_info_plain():
    assert p3._active_loh_label_info("WT") == ("WT", ())
