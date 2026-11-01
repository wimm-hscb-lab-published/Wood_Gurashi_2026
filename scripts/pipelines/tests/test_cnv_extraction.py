import numpy as np
import pandas as pd
import pytest

from CNV_Utils import cnv_extraction as ce
from _cnv_pro_fakes import FakeAssay, FakeSample


def _cnv_assay(ploidy=None, labels=None):
    # 4 cells x 4 amplicons, columns deliberately out of genomic order
    ids = ["AMP_2a", "AMP_1b", "AMP_X", "AMP_1a"]
    chroms = ["2", "1", "X", "1"]
    starts = [100, 500, 10, 50]
    if ploidy is None:
        ploidy = np.array(
            [
                [2.0, 2.1, 1.0, 2.2],
                [2.0, 1.9, 1.2, 2.4],
                [1.0, 3.0, 2.0, 2.0],
                [1.2, 3.2, 2.2, np.nan],
            ]
        )
    labels = labels if labels is not None else ["WT", "WT", "C1", "C1"]
    return FakeAssay(
        layers={"ploidy": ploidy},
        barcodes=["b0", "b1", "b2", "b3"],
        ids=ids,
        row_attrs={"label": labels},
        col_attrs={"CHROM": chroms, "start_pos": starts},
    )


def test_get_cnv_info_genomic_order_and_shapes():
    sample = FakeSample(cnv=_cnv_assay())
    df_cnv, grouped, vars_, info = ce.get_cnv_info(sample, verbose=False)

    expected_order = ["AMP_1a", "AMP_1b", "AMP_2a", "AMP_X"]
    assert list(df_cnv.columns) == ["label"] + expected_order
    assert list(info["id"]) == expected_order
    assert list(df_cnv.index) == ["b0", "b1", "b2", "b3"]
    assert vars_ is None

    assert list(grouped.index) == expected_order
    assert list(grouped.columns) == ["C1", "WT"]  # sorted when no order given
    assert grouped.loc["AMP_2a", "WT"] == pytest.approx(2.0)
    assert grouped.loc["AMP_1b", "C1"] == pytest.approx(3.1)
    # nanmedian ignores the NaN for AMP_1a in C1
    assert grouped.loc["AMP_1a", "C1"] == pytest.approx(2.0)


def test_get_cnv_info_custom_order_filters_missing_labels():
    sample = FakeSample(cnv=_cnv_assay())
    _, grouped, _, _ = ce.get_cnv_info(
        sample, desired_plotting_order=["WT", "Ghost", "C1"], verbose=False
    )
    assert list(grouped.columns) == ["WT", "C1"]


def test_get_cnv_info_chr_prefixed_chromosomes_sort_numerically():
    a = _cnv_assay()
    a.col_attrs["CHROM"] = np.array(["chr10", "chr2", "chrMT", "chr2"], dtype=object)
    _, _, _, info = ce.get_cnv_info(FakeSample(cnv=a), verbose=False)
    assert list(info["CHROM"]) == ["chr2", "chr2", "chr10", "chrMT"]


def test_get_cnv_info_variant_mapping():
    sample = FakeSample(cnv=_cnv_assay())
    mapping = {
        "variant": ["chr1:60:A/T", "chr2:150:C/G", "chr9:1:A/C"],
        "amplicon": ["AMP_1a", "AMP_2a", "AMP_MISSING"],
    }
    df_cnv, _, vars_, _ = ce.get_cnv_info(sample, df_mut_to_amplicon=mapping, verbose=False)
    # metadata = every df_cnv column not used as a mapped amplicon
    assert list(vars_.columns) == ["label", "AMP_1b", "AMP_X", "chr1:60:A/T", "chr2:150:C/G", "chr9:1:A/C"]
    pd.testing.assert_series_equal(vars_["chr1:60:A/T"], df_cnv["AMP_1a"], check_names=False)
    assert vars_["chr9:1:A/C"].isna().all()


def test_get_cnv_info_missing_required_col_attrs():
    a = _cnv_assay()
    del a.col_attrs["start_pos"]
    with pytest.raises(KeyError, match="start_pos"):
        ce.get_cnv_info(FakeSample(cnv=a), verbose=False)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "cnv_extraction.py get_cnv_info: df_cnv.drop_duplicates() is applied to the "
        "cell x amplicon table, so distinct cells with identical label+ploidy "
        "profiles (common with integer/rounded ploidy) are silently collapsed."
    ),
)
def test_get_cnv_info_keeps_identical_cells():
    ploidy = np.full((4, 4), 2.0)
    sample = FakeSample(cnv=_cnv_assay(ploidy=ploidy, labels=["WT"] * 4))
    df_cnv, _, _, _ = ce.get_cnv_info(sample, verbose=False)
    assert len(df_cnv) == 4


@pytest.mark.xfail(
    strict=True,
    reason=(
        "cnv_extraction.py get_cnv_info: df_cnv_grouped.drop_duplicates() drops "
        "amplicons whose per-clone median profile equals another amplicon's (e.g. "
        "every diploid amplicon at 2.0); the later reindex(target_ids) reinserts "
        "them as all-NaN rows, losing the clone medians."
    ),
)
def test_get_cnv_info_grouped_keeps_amplicons_with_identical_medians():
    ploidy = np.full((4, 4), 2.0)
    sample = FakeSample(cnv=_cnv_assay(ploidy=ploidy))
    _, grouped, _, _ = ce.get_cnv_info(sample, verbose=False)
    assert not grouped.isna().any().any()


# --------------------------------------------------------------------------
# private helpers
# --------------------------------------------------------------------------
def _simple(barcodes, labels=None, ids=("v1", "v2")):
    ra = {} if labels is None else {"label": labels}
    return FakeAssay(
        layers={"x": np.zeros((len(barcodes), len(ids)))},
        barcodes=barcodes,
        ids=list(ids),
        row_attrs=ra,
    )


def test_transfer_labels_by_barcode_maps_and_fills_nan():
    src = _simple(["a", "b", "c"], labels=["L1", "L2", "L3"])
    tgt = _simple(["c", "z", "a"], labels=["?", "?", "?"])
    ce._transfer_labels_by_barcode(src, tgt)
    out = tgt.row_attrs["label"]
    assert out[0] == "L3" and out[2] == "L1"
    assert pd.isna(out[1])


def test_transfer_labels_custom_column_and_non_str_barcodes():
    src = FakeAssay({"x": np.zeros((2, 1))}, [1, 2], ["v"], row_attrs={"clone": ["A", "B"]})
    tgt = FakeAssay({"x": np.zeros((2, 1))}, ["2", "1"], ["v"])
    ce._transfer_labels_by_barcode(src, tgt, label_col="clone")
    assert list(tgt.row_attrs["clone"]) == ["B", "A"]


def test_sync_dna_cnv_by_barcode_preserves_dna_order():
    dna = _simple(["c", "a", "b", "x"], ids=("var1",))
    cnv = FakeAssay(
        {"ploidy": np.arange(6).reshape(3, 2)}, ["a", "b", "c"], ["AMP1", "AMP2"]
    )
    s = FakeSample(dna=dna, cnv=cnv)
    shared = ce._sync_dna_cnv_by_barcode(s)
    assert shared == ["c", "a", "b"]
    assert list(s.dna.barcodes()) == ["c", "a", "b"]
    assert list(s.cnv.barcodes()) == ["c", "a", "b"]
    assert list(s.cnv.ids()) == ["AMP1", "AMP2"]


def test_sync_dna_cnv_by_barcode_subsets_amplicons():
    dna = _simple(["a", "b"], ids=("var1",))
    cnv = FakeAssay({"ploidy": np.arange(6).reshape(2, 3)}, ["b", "a"], ["A1", "A2", "A3"])
    s = FakeSample(dna=dna, cnv=cnv)
    ce._sync_dna_cnv_by_barcode(s, amplicons=["A3", "A1"])
    assert list(s.cnv.ids()) == ["A3", "A1"]
    np.testing.assert_array_equal(s.cnv.layers["ploidy"], [[5, 3], [2, 0]])


def test_sync_no_shared_barcodes():
    s = FakeSample(dna=_simple(["a"]), cnv=_simple(["b"]))
    assert ce._sync_dna_cnv_by_barcode(s) == []


def test_get_labeled_attribute_inserts_label_first():
    a = FakeAssay(
        {"ploidy": [[1.0, 2.0], [3.0, 4.0]]}, ["a", "b"], ["A1", "A2"], row_attrs={"label": ["X", "Y"]}
    )
    df = ce._get_labeled_attribute(a, "ploidy")
    assert list(df.columns) == ["label", "A1", "A2"]
    assert list(df["label"]) == ["X", "Y"]
    # original layer frame untouched
    assert "label" not in a.get_attribute("ploidy").columns


# --------------------------------------------------------------------------
# extract_cnv_data (input validation + a fake end-to-end run without ploidy)
# --------------------------------------------------------------------------
def test_extract_cnv_data_requires_out_dir():
    with pytest.raises(ValueError, match="OUT_DIR"):
        ce.extract_cnv_data(Sample=None, wd="/tmp", EXPERIMENT=None, verbose=False)


def _extract_sample():
    dna = FakeAssay(
        {"AF": np.zeros((3, 2))},
        ["a", "b", "c"],
        ["chr1:60:A/T", "chr2:150:C/G"],
        row_attrs={"label": ["WT", "C1", "C1"]},
        col_attrs={"amplicon": ["AMP_1a", "AMP_2a"]},
    )
    cnv = _cnv_assay()
    cnv.row_attrs["barcode"] = np.array(["c", "a", "zz", "b"], dtype=object)
    cnv.row_attrs["label"] = np.array(["?"] * 4, dtype=object)
    return FakeSample(dna=dna, cnv=cnv)


@pytest.mark.parametrize("diploid_label", ["NOPE", ["NOPE", "ALSO_NO"]])
def test_extract_cnv_data_no_diploid_cells(tmp_path, diploid_label):
    with pytest.raises(ValueError, match="diploid"):
        ce.extract_cnv_data(
            _extract_sample(),
            desired_plotting_order=["WT", "C1"],
            OUT_DIR=str(tmp_path),
            diploid_label=diploid_label,
            verbose=False,
        )


def test_extract_cnv_data_end_to_end_without_ploidy(tmp_path):
    out = ce.extract_cnv_data(
        _extract_sample(),
        desired_plotting_order=["WT", "C1"],
        OUT_DIR=str(tmp_path),
        output_tag="_t",
        compute_ploidy=False,
        verbose=False,
    )
    assert out["shared_barcodes"] == ["a", "b", "c"]
    assert out["variant_amplicons"] == ["AMP_1a", "AMP_2a"]
    assert out["variant_to_amp_dict"] == {"chr1:60:A/T": "AMP_1a", "chr2:150:C/G": "AMP_2a"}
    # labels transferred from DNA to CNV by barcode
    assert list(out["df_ploidy"]["label"]) == ["WT", "C1", "C1"]
    assert list(out["df_cnv_vars"].columns) == ["label", "chr1:60:A/T", "chr2:150:C/G"]
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(
        [
            "Amplicons_to_variant_t.csv",
            "Cell_x_ploidy_data_t.csv",
            "Clone_x_ploidy_data_t.csv",
            "Variants_x_ploidy_data_t.csv",
            "Ploidy_metadata_t.csv",
        ]
    )
