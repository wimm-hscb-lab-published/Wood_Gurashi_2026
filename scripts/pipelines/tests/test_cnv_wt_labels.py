"""WT/reference label helpers in CNV_Utils.informativeness.

The module defines these helpers twice (lines ~16-53 and ~106-121); the
second definitions are the live ones at import time. Both copies are
behaviourally identical, so these tests cover the live versions.
"""
import pandas as pd
import pytest

from CNV_Utils import informativeness as inf


@pytest.mark.parametrize(
    "wt_labels, expected",
    [
        (None, ["WT"]),
        ("WT", ["WT"]),
        ("Ref", ["Ref"]),
        ("", [""]),
        (["WT", "Normal"], ["WT", "Normal"]),
        (("A",), ["A"]),
        ([], []),
    ],
)
def test_normalize_wt_labels(wt_labels, expected):
    assert inf._normalize_wt_labels(wt_labels) == expected


def test_normalize_returns_new_list():
    src = ["WT"]
    out = inf._normalize_wt_labels(src)
    out.append("X")
    assert src == ["WT"]


def test_build_wt_mask_exact_match_and_strip():
    labels = pd.Series([" WT", "WT ", "WT LOH-C2 chr7q:q22.1", "wt", "Clone1", "WT"])
    mask = inf._build_wt_mask(labels, None)
    assert mask.tolist() == [True, True, False, False, False, True]


def test_build_wt_mask_multiple_labels_and_index_preserved():
    labels = pd.Series(["A", "B", "C"], index=["x", "y", "z"])
    mask = inf._build_wt_mask(labels, ["A", "C"])
    assert mask.tolist() == [True, False, True]
    assert list(mask.index) == ["x", "y", "z"]


def test_build_wt_mask_non_string_labels_are_cast():
    labels = pd.Series([0, 1, 2])
    assert inf._build_wt_mask(labels, "1").tolist() == [False, True, False]
    # integer reference label never matches because labels are cast to str
    assert inf._build_wt_mask(labels, [1]).tolist() == [False, False, False]


def test_build_wt_mask_empty():
    assert inf._build_wt_mask(pd.Series([], dtype=object), None).tolist() == []


@pytest.mark.parametrize(
    "label, wt, expected",
    [
        ("WT", None, True),
        ("  WT  ", None, True),
        ("WT-1", None, False),
        ("Normal", ["WT", "Normal"], True),
        (5, "5", True),
        (float("nan"), None, False),
        ("WT", [], False),
    ],
)
def test_is_wt_label(label, wt, expected):
    assert inf._is_wt_label(label, wt) is expected


def test_reference_labels_are_not_stripped():
    # Cell labels are stripped but the reference labels are not, so padded
    # reference labels never match. Documented current behaviour.
    assert inf._is_wt_label("WT", " WT ") is False
