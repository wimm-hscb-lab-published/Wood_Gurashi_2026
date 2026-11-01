import re

import pytest

from PRO_Utils import colors
from PRO_Utils import get_celltype_color, CELLTYPE_COLOR_MAP


def test_package_reexports():
    assert get_celltype_color is colors.get_celltype_color
    assert CELLTYPE_COLOR_MAP is colors.CELLTYPE_COLOR_MAP


def test_all_map_values_are_hex():
    assert all(re.fullmatch(r"#[0-9A-Fa-f]{6}", c) for c in CELLTYPE_COLOR_MAP.values())
    assert re.fullmatch(r"#[0-9A-Fa-f]{6}", colors.DEFAULT_COLOR)


@pytest.mark.parametrize("name", list(CELLTYPE_COLOR_MAP))
def test_exact_match(name):
    assert get_celltype_color(name) == CELLTYPE_COLOR_MAP[name]


@pytest.mark.parametrize(
    "name, key",
    [
        ("erythroid", "Erythroid"),
        ("ERYTHROID", "Erythroid"),
        ("pro myeloid", "Pro-Myeloid"),
        ("pro_myeloid", "Pro-Myeloid"),
        ("hsc-mpp", "HSC_MPP"),
        ("cd14 monocyte", "CD14_monocyte"),
        ("PDC", "pDC"),
        ("other-t", "Other_T"),
    ],
)
def test_normalised_match(name, key):
    assert get_celltype_color(name) == CELLTYPE_COLOR_MAP[key]


@pytest.mark.parametrize("name", ["", "Unknown", "T cell", " Erythroid", "Erythroid "])
def test_fallback_grey(name):
    # Leading/trailing spaces become underscores, so they do not match.
    assert get_celltype_color(name) == colors.DEFAULT_COLOR


def test_numpy_str_works():
    np = pytest.importorskip("numpy")
    assert get_celltype_color(np.str_("NK")) == CELLTYPE_COLOR_MAP["NK"]
