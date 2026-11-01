import numpy as np
import pandas as pd
import pytest

from PRO_Utils import update_labels as ul
from _cnv_pro_fakes import FakeAssay, FakeSample


def _sample(barcodes=("a", "b", "c"), labels=None):
    ra = {} if labels is None else {"label": labels}
    prot = FakeAssay({"read_counts": np.zeros((len(barcodes), 1))}, list(barcodes), ["CD3"], row_attrs=ra)
    return FakeSample(protein=prot)


def test_index_barcodes_with_existing_labels_kept_for_unmatched():
    s = _sample(labels=["old_a", "old_b", "old_c"])
    anno = pd.DataFrame({"label": ["B", "NK"]}, index=["c", "a"])
    ul.update_protein_labels_from_dataframe(s, anno, verbose=False)
    assert list(s.protein.row_attrs["label"]) == ["NK", "old_b", "B"]


def test_unmatched_without_existing_attr_gets_unknown():
    s = _sample()
    anno = pd.DataFrame({"celltype": ["B"]}, index=["b"])
    ul.update_protein_labels_from_dataframe(
        s, anno, label_col="celltype", target_attr="celltype", verbose=False
    )
    assert list(s.protein.row_attrs["celltype"]) == ["Unknown", "B", "Unknown"]


def test_barcode_column():
    s = _sample()
    anno = pd.DataFrame({"bc": ["a", "b", "c"], "label": ["X", "Y", "Z"]})
    ul.update_protein_labels_from_dataframe(s, anno, barcode_col="bc", verbose=False)
    assert list(s.protein.row_attrs["label"]) == ["X", "Y", "Z"]


def test_duplicate_barcodes_last_wins():
    s = _sample()
    anno = pd.DataFrame({"label": ["first", "second"]}, index=["a", "a"])
    ul.update_protein_labels_from_dataframe(s, anno, verbose=False)
    assert s.protein.row_attrs["label"][0] == "second"


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"barcode_col": "nope"}, "nope"),
        ({"label_col": "missing"}, "missing"),
    ],
)
def test_missing_columns_raise(kwargs, match):
    s = _sample()
    anno = pd.DataFrame({"label": ["X"]}, index=["a"])
    with pytest.raises(ValueError, match=match):
        ul.update_protein_labels_from_dataframe(s, anno, verbose=False, **kwargs)


def test_returns_none_and_verbose_output(capsys):
    s = _sample()
    anno = pd.DataFrame({"label": ["X"]}, index=["a"])
    assert ul.update_protein_labels_from_dataframe(s, anno, verbose=True) is None
    out = capsys.readouterr().out
    assert "Matched: 1 cells (33.3%)" in out
    assert "LABELS UPDATED" in out


def test_quiet_mode_is_silent(capsys):
    s = _sample()
    ul.update_protein_labels_from_dataframe(s, pd.DataFrame({"label": ["X"]}, index=["a"]), verbose=False)
    assert capsys.readouterr().out == ""


@pytest.mark.xfail(
    strict=True,
    raises=ZeroDivisionError,
    reason=(
        "update_labels.py: the 'Matched/Not matched' percentage f-strings divide by "
        "len(protein_barcodes) and are evaluated even when verbose=False, so an "
        "assay with zero cells raises ZeroDivisionError."
    ),
)
def test_empty_protein_assay():
    s = _sample(barcodes=())
    ul.update_protein_labels_from_dataframe(s, pd.DataFrame({"label": ["X"]}, index=["a"]), verbose=False)
    assert len(s.protein.row_attrs["label"]) == 0
