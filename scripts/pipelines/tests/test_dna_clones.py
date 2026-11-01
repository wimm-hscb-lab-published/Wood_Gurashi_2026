"""Unit tests for DNA_Utils.clones, DNA_Utils.plot_clones and small somatic helpers."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from DNA_Utils import clones as C
from DNA_Utils import plot_clones as P
from DNA_Utils import somatic as S


def make_dna(ngt, labels, ids=None, af=None):
    from missionbio.mosaic.dna import Dna

    ngt = np.asarray(ngt)
    n, m = ngt.shape
    ids = np.array(ids if ids is not None else [f"chr1:{i + 1}:A/T" for i in range(m)])
    if af is None:
        af = np.where(ngt == 1, 50.0, np.where(ngt == 2, 100.0, 0.0))
    return Dna(
        name="dna",
        metadata={"sample_name": np.array(["s1"])},
        layers={"NGT": ngt, "AF": np.asarray(af, dtype=float), "DP": np.full(ngt.shape, 20),
                "GQ": np.full(ngt.shape, 30)},
        row_attrs={"barcode": np.array([f"BC{i}" for i in range(n)]),
                   "sample_name": np.array(["s1"] * n), "label": np.array(labels)},
        col_attrs={"id": ids},
    )


# ── assign_patients_from_whitelist ───────────────────────────────────────────

# ── clones_to_dataframe (fake workflow) ──────────────────────────────────────

class _FakeWF:
    def __init__(self):
        self.kwargs = None

    def get_clones(self, variants, max_ado_score, min_clone_size):
        self.kwargs = dict(variants=variants, max_ado_score=max_ado_score,
                           min_clone_size=min_clone_size)
        rows = [
            {"vn": "chr1:1:A/T", "c2": "Het", "c1": "WT", "junk": 1},
            {"vn": "  ", "c2": "x", "c1": "y", "junk": 2},
            {"vn": "chr1:2:A/T", "c2": "Hom", "c1": "WT", "junk": 3},
        ]
        cols = [{"id": "vn", "name": "Variant\nname"},
                {"id": "c1", "name": "Clone 1"},
                {"id": "c2", "name": "Clone 2"}]
        return rows, cols, None


def test_clones_to_dataframe_normalises_and_orders():
    wf = _FakeWF()
    df = C.clones_to_dataframe(
        wf, [" a ", np.array(["b", "a"]), pd.Series(["c"]), ["d"], ("e",), 7, ""],
        max_ado_score=0.5, min_clone_size=3)
    assert wf.kwargs == {"variants": ["a", "b", "c", "d", "e", "7"],
                         "max_ado_score": 0.5, "min_clone_size": 3}
    assert list(df.columns) == ["Variant\nname", "Clone 1", "Clone 2"]
    assert df["Variant\nname"].tolist() == ["chr1:1:A/T", "chr1:2:A/T"]
    assert df.index.tolist() == [0, 1]


@pytest.mark.xfail(
    strict=True,
    reason="clones.py:95 filters `x is not None`, but _normalize() has already "
    "stringified every element, so None entries survive as the variant 'None' "
    "and are passed to wf.get_clones.",
)
def test_clones_to_dataframe_drops_none():
    wf = _FakeWF()
    C.clones_to_dataframe(wf, ["a", None])
    assert wf.kwargs["variants"] == ["a"]


@pytest.mark.parametrize(
    "variants, expected",
    [
        ([["n1", ["n2"]]], ["n1", "['n2']"]),  # only one nesting level is flattened
        ("chr1:1:A/T", ["chr1:1:A/T"]),
        (np.array([["x", "y"]]), ["x", "y"]),
        (pd.Series([1, 1]), ["1"]),
        ({"q"}, ["q"]),
        (5, ["5"]),
    ],
)
def test_clones_to_dataframe_variant_inputs(variants, expected):
    wf = _FakeWF()
    C.clones_to_dataframe(wf, variants)
    assert wf.kwargs["variants"] == expected


# ── get_clones_size / get_VAF on a real Dna ──────────────────────────────────

def test_get_clones_size():
    dna = make_dna(np.zeros((5, 2), dtype=int), ["A", "B", "B", "C", "C"])
    sample = SimpleNamespace(dna=dna)
    out = C.get_clones_size(sample, verbose=False, return_percentage=True)
    assert dict(zip(out["Clone"], out["Count"])) == {"A": 1, "B": 2, "C": 2}
    assert out["Percentage"].sum() == pytest.approx(100.0)

    ordered = C.get_clones_size(sample, desired_order=["C", "A"], verbose=False)
    assert ordered["Clone"].astype(str).tolist() == ["C", "A", "B"]
    assert ordered["Count"].tolist() == [2, 1, 2]


def test_get_VAF_per_clone(tmp_path):
    ngt = np.array([[1, 0], [1, 0], [2, 0], [0, 1]])
    dna = make_dna(ngt, ["A", "A", "A", "B"])
    sample = SimpleNamespace(dna=dna)
    out_csv = tmp_path / "out" / "vaf.csv"
    df = C.get_VAF(sample, clone_order=["B"], output_path=str(out_csv), verbose=False)
    assert "NGT_FILTERED" in dna.layers
    assert list(df.columns) == ["B", "A"]
    assert df.shape == (2, 2)
    # clone A, variant 1: (1+1+2)/(2*3) = 66.67% ; clone B, variant 2: 50%
    assert df.iloc[0]["A"] == pytest.approx(100 * 4 / 6, abs=0.1)
    assert df.iloc[1]["B"] == pytest.approx(50.0, abs=0.1)
    assert df.iloc[1]["A"] == pytest.approx(0.0)
    assert out_csv.exists()
    assert pd.read_csv(out_csv).columns[0] == "Variants"


# ── plot_clones._get_clone_color ─────────────────────────────────────────────

@pytest.mark.parametrize(
    "name, color",
    [
        ("WT", "#007da5"),
        ("wt", "#007da5"),
        ("TP53 Hom", "#a50000"),
        ("TP53_hem", "#a50000"),
        ("TP53 Het TP53 Het", "#a50000"),
        ("TP53 Het", "#f3b900"),
        ("TP53_Het DNMT3A_Het", "#f3b900"),
        ("DNMT3A Het", "#00A77F"),
        ("DNMT3A Hom", "#4B0DD2"),
        ("DNMT3A Het TET2 Hom", "#00A77F"),  # het checked before hom
        ("TP53", "#6f00a5"),
        ("Clone 3", "#6f00a5"),
        ("Hetero", "#6f00a5"),  # 'het' must be a whole token
        ("", "#6f00a5"),
        (None, "#6f00a5"),
    ],
)
def test_get_clone_color(name, color):
    assert P._get_clone_color(name) == color


# ── somatic._protein_umap_frame ──────────────────────────────────────────────

class _FakeProtein:
    def __init__(self, row_attrs, barcodes=None):
        self.row_attrs = row_attrs
        self._bc = barcodes

    def barcodes(self):
        if self._bc is None:
            raise RuntimeError
        return self._bc

    def get_attribute(self, name, constraint=None):
        raise KeyError(name)


def test_protein_umap_frame():
    prot = _FakeProtein({"umap": np.array([[0, 1], [2, 3]]),
                         "Final_annotation": np.array(["T", "B"])},
                        barcodes=np.array(["a", "b"]))
    df = S._protein_umap_frame(SimpleNamespace(protein=prot))
    assert df.index.name == "barcode" and df.index.tolist() == ["a", "b"]
    assert df["umap_x"].tolist() == [0, 2] and df["umap_y"].tolist() == [1, 3]
    assert df["annot"].tolist() == ["T", "B"]


def test_protein_umap_frame_fallbacks():
    prot = _FakeProtein({"barcode": np.array(["a", "b"]), "umap": np.zeros((2, 3)),
                         "Final_annotation": np.array(["only_one"])})
    df = S._protein_umap_frame(SimpleNamespace(protein=prot))
    assert df["annot"].tolist() == ["all", "all"]
    with pytest.raises(ValueError):
        S._protein_umap_frame(SimpleNamespace(protein=_FakeProtein({"umap": np.zeros(2)}, ["a", "b"])))
