"""Unit tests for DNA_Utils.filter_outliers helpers."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from DNA_Utils import filter_outliers as F


def make_dna(labels, ids=("chr17:7577539:G/A", "chr2:200:C/G")):
    from missionbio.mosaic.dna import Dna

    n, m = len(labels), len(ids)
    ngt = np.zeros((n, m), dtype=int)
    return Dna(
        name="dna",
        metadata={"sample_name": np.array(["s1"])},
        layers={"NGT": ngt, "AF": ngt.astype(float), "DP": np.full((n, m), 10)},
        row_attrs={"barcode": np.array([f"BC{i}" for i in range(n)]),
                   "sample_name": np.array(["s1"] * n), "label": np.array(labels)},
        col_attrs={"id": np.array(ids)},
    )


@pytest.mark.parametrize(
    "s, expected",
    [("chr1:100:A/T", "chr1_100_A_T"), ("a b\\c", "a_b_c"), (12, "12"), ("", "")],
)
def test_safe_filename(s, expected):
    assert F._safe_filename(s) == expected


@pytest.mark.parametrize(
    "v, expected",
    [
        ("TP53:chr17:7577539:G/A", "chr17:7577539:G/A"),
        ("chr17:7577539:G/A", "chr17:7577539:G/A"),
        ("  CHR1:5:A/T ", "CHR1:5:A/T"),
        ("GENE:sub:chrX:1:A/T", "chrX:1:A/T"),
        ("no_chr_here", "no_chr_here"),
        ("", ""),
    ],
)
def test_canonical_variant_key(v, expected):
    assert F._canonical_variant_key(v) == expected


@pytest.mark.parametrize(
    "value, accepted, expected",
    [
        ("Het (50%)", ("het", "hom"), True),
        ("Hom (99%)", ("het", "hom"), True),
        ("Hemi (99%)", ("hom",), True),
        ("homozygous", ("HOM",), True),
        ("1/1", ("hom",), True),
        ("0/1", ("hom",), False),
        ("Het", ("hom",), False),
        ("Hom", ("het",), False),
        ("WT (0.1%)", ("het", "hom"), False),
        (np.nan, ("het", "hom"), False),
        ("Het", (), False),
    ],
)
def test_clone_call_matches(value, accepted, expected):
    assert F._clone_call_matches(value, accepted) is expected


def test_prepare_clone_variant_calls_wrapped_header():
    df = pd.DataFrame({"Variant\nname": ["TP53:chr17:1:G/A", "chr2:5:C/T"], "WT": ["WT", "Het"]})
    out = F._prepare_clone_variant_calls(df, "Variant name")
    assert out.index.tolist() == ["chr17:1:G/A", "chr2:5:C/T"]
    assert "Variant\nname" in out.columns  # drop=False
    assert df.index.tolist() == [0, 1]  # input not modified


def test_prepare_clone_variant_calls_index_fallback_and_type_error():
    df = pd.DataFrame({"WT": ["Het"]}, index=["GENE:chr1:1:A/T"])
    out = F._prepare_clone_variant_calls(df, "Variant name")
    assert out.index.tolist() == ["chr1:1:A/T"]
    with pytest.raises(TypeError):
        F._prepare_clone_variant_calls([1, 2], "Variant name")


def _sample_with_barcodes(bcs):
    return SimpleNamespace(dna=SimpleNamespace(row_attrs={"barcode": np.array(bcs)}))


def test_ensure_barcode_index_positional_full():
    s = _sample_with_barcodes(["a", "b", "c"])
    af = pd.DataFrame({"v": [1, 2, 3]})
    dp = pd.DataFrame({"v": [4, 5, 6]})
    a2, d2 = F._ensure_barcode_index(af, dp, s)
    assert a2.index.tolist() == ["a", "b", "c"] and d2.index.tolist() == ["a", "b", "c"]
    assert af.index.tolist() == [0, 1, 2]


def test_ensure_barcode_index_positional_subset():
    s = _sample_with_barcodes(["a", "b", "c"])
    af = pd.DataFrame({"v": [1, 2]}, index=[2, 7])
    a2, d2 = F._ensure_barcode_index(af, af.copy(), s)
    assert a2.index.tolist() == ["c", "7"]


def test_ensure_barcode_index_string_index_and_empty():
    s = _sample_with_barcodes(["a", "b"])
    af = pd.DataFrame({"v": [1]}, index=["b"])
    a2, _ = F._ensure_barcode_index(af, af.copy(), s)
    assert a2.index.tolist() == ["b"]
    # same length but already barcodes -> kept as-is
    af2 = pd.DataFrame({"v": [1, 2]}, index=["b", "a"])
    assert F._ensure_barcode_index(af2, af2, s)[0].index.tolist() == ["b", "a"]
    with pytest.raises(ValueError):
        F._ensure_barcode_index(pd.DataFrame({"v": []}), pd.DataFrame({"v": []}), s)


def test_sd_filter_per_label():
    df = pd.DataFrame({
        "label": ["A"] * 6 + ["B"] * 2 + ["C"] * 3 + [None],
        "y": [10, 10, 10, 10, 10, 100, 1, 1000, 5, 5, 5, 3],
    })
    kept, audit = F._sd_filter_per_label(df, "y", k=1.5, min_n=3)
    assert 5 not in kept.index  # outlier in A removed
    assert {6, 7}.issubset(kept.index)  # B below min_n -> kept
    assert {8, 9, 10}.issubset(kept.index)  # C zero SD -> kept
    assert 11 not in kept.index and 11 not in audit.index  # NaN label dropped
    assert audit.loc[5, "status"] == "removed"
    assert (audit.drop(index=5)["status"] == "kept").all()
    assert np.isinf(audit.loc[6, "lo"]) and np.isinf(audit.loc[8, "hi"])
    a_rows = audit[audit["label"] == "A"]
    assert a_rows["mu"].iloc[0] == pytest.approx(25.0)
    assert a_rows["sd"].iloc[0] == pytest.approx(np.std([10] * 5 + [100], ddof=1))


def test_sd_filter_per_label_singleton_and_empty():
    df = pd.DataFrame({"label": ["A"], "y": [3.0]})
    kept, audit = F._sd_filter_per_label(df, "y")
    assert len(kept) == 1 and np.isnan(audit["sd"].iloc[0])
    e_kept, e_audit = F._sd_filter_per_label(pd.DataFrame({"label": ["A"], "y": ["x"]}), "y")
    assert e_kept.empty and e_audit.empty


def test_sd_filter_per_label_boundary_inclusive():
    # mean 0, sd 1 for [-1, 1, -1, 1] (ddof=1 -> sd = 1.1547); k such that bound == 1
    x = np.array([-1.0, 1.0, -1.0, 1.0])
    k = 1.0 / np.std(x, ddof=1)
    kept, _ = F._sd_filter_per_label(pd.DataFrame({"label": "A", "y": x}), "y", k=k)
    assert len(kept) == 4


def test_get_het_clone_barcodes_by_variant():
    dna = make_dna(["WT", "WT", "Clone1", "Clone2", "Clone2"])
    sample = SimpleNamespace(dna=dna)
    calls = pd.DataFrame({
        "Variant\nname": ["TP53:chr17:7577539:G/A", "chr2:200:C/G"],
        "WT": ["WT (0%)", "Het (50%)"],
        "Clone1": ["Het (48%)", "WT (0%)"],
        "Clone2": ["Hom (99%)", "WT (1%)"],
        "Small subclones (3)": ["Het", "Het"],
    })
    elig, clones, audit = F._get_het_clone_barcodes_by_variant(
        sample, ["chr17:7577539:G/A", "chr2:200:C/G", "chr9:1:A/T"], calls)
    assert clones["chr17:7577539:G/A"] == ["Clone1", "Clone2"]
    assert elig["chr17:7577539:G/A"] == {"BC2", "BC3", "BC4"}
    assert elig["chr2:200:C/G"] == {"BC0", "BC1"}
    assert elig["chr9:1:A/T"] == set()
    assert audit["status"].tolist() == ["ok", "ok", "missing_from_clone_variant_calls"]

    _, clones_het, _ = F._get_het_clone_barcodes_by_variant(
        sample, ["chr17:7577539:G/A"], calls, eligible_genotypes=("het",))
    assert clones_het["chr17:7577539:G/A"] == ["Clone1"]


def test_get_het_clone_barcodes_by_variant_no_matching_columns():
    sample = SimpleNamespace(dna=make_dna(["A", "B"]))
    calls = pd.DataFrame({"Variant name": ["chr2:200:C/G"], "Zzz": ["Het"]})
    with pytest.raises(ValueError):
        F._get_het_clone_barcodes_by_variant(sample, ["chr2:200:C/G"], calls)
