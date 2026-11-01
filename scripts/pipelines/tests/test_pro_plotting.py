"""Cheap smoke tests for PRO_Utils plotting helpers (no image export)."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from PRO_Utils import plot_umap, homogeneity_silhouette as hs
from PRO_Utils.colors import CELLTYPE_COLOR_MAP, DEFAULT_COLOR
from _cnv_pro_fakes import FakeAssay, FakeSample


@pytest.fixture(autouse=True)
def _close_figs():
    yield
    plt.close("all")


def _umap_sample(labels, ncomp=2):
    n = len(labels)
    coords = np.arange(n * ncomp, dtype=float).reshape(n, ncomp)
    prot = FakeAssay(
        {"umap": coords}, [f"b{i}" for i in range(n)], [f"UMAP{j}" for j in range(ncomp)],
        row_attrs={"label": labels},
    )
    return FakeSample(protein=prot)


# --------------------------------------------------------------------------
# plot_protein_umap
# --------------------------------------------------------------------------
def test_plot_protein_umap_traces_and_colours():
    s = _umap_sample(["NK", "B", "NK", "Mystery"])
    fig = plot_umap.plot_protein_umap(s, show_plot=False, verbose=False)
    names = [t.name for t in fig.data]
    assert names == ["B", "Mystery", "NK"]
    colours = {t.name: t.marker.color for t in fig.data}
    assert colours["NK"] == CELLTYPE_COLOR_MAP["NK"]
    assert colours["Mystery"] == DEFAULT_COLOR
    nk = fig.data[2]
    assert list(nk.x) == [0.0, 4.0]


def test_plot_protein_umap_bad_attribute_returns_none():
    s = _umap_sample(["NK"])
    assert plot_umap.plot_protein_umap(s, umap_attribute="nope", show_plot=False, verbose=False) is None
    assert plot_umap.plot_protein_umap(s, celltype_column="nope", show_plot=False, verbose=False) is None


@pytest.mark.xfail(
    strict=True,
    raises=AttributeError,
    reason=(
        "colors.py get_celltype_color calls celltype_name.lower(); plot_protein_umap "
        "passes np.unique(labels) items straight through, so numeric cluster labels "
        "(e.g. Leiden ids) raise AttributeError instead of falling back to grey."
    ),
)
def test_plot_protein_umap_numeric_labels():
    s = _umap_sample(np.array([0, 1, 1]))
    fig = plot_umap.plot_protein_umap(s, show_plot=False, verbose=False)
    assert len(fig.data) == 2


# --------------------------------------------------------------------------
# plot_protein_homogeneity_silhouette
# --------------------------------------------------------------------------
def _pro_df(seed=0):
    rng = np.random.default_rng(seed)
    a = rng.normal(0, 0.1, size=(20, 3))
    b = rng.normal(5, 0.1, size=(20, 3))
    df = pd.DataFrame(np.vstack([a, b]), columns=["CD3", "CD19", "CD34"])
    df["label"] = ["T"] * 20 + ["B"] * 20
    return df


def test_homogeneity_well_separated():
    per_label, gh = hs.plot_protein_homogeneity_silhouette(_pro_df(), show_plot=False, verbose=False)
    assert gh == pytest.approx(1.0)
    assert set(per_label["label"]) == {"T", "B"}
    assert (per_label["count"] == 20).all()
    assert (per_label["mean"] > 0.9).all()
    assert (per_label["homogeneity_like"] == per_label["mean"].clip(lower=0)).all()


def test_homogeneity_non_numeric_and_duplicate_columns_handled():
    df = _pro_df()
    df.columns = [1, "1", "CD34", "label"]  # duplicate after str cast
    df["junk"] = "x"  # coerced to NaN -> 0
    per_label, _ = hs.plot_protein_homogeneity_silhouette(df, scale=False, show_plot=False, verbose=False)
    assert len(per_label) == 2


@pytest.mark.parametrize(
    "df, exc",
    [
        (pd.DataFrame({"x": [1, 2]}), KeyError),
        (pd.DataFrame({"x": [1, 2], "label": ["A", "A"]}), ValueError),
    ],
)
def test_homogeneity_validation(df, exc):
    with pytest.raises(exc):
        hs.plot_protein_homogeneity_silhouette(df, show_plot=False, verbose=False)


def test_homogeneity_save_csv(tmp_path):
    path = tmp_path / "sub" / "stats.csv"
    hs.plot_protein_homogeneity_silhouette(
        _pro_df(), save_csv=True, csv_path=str(path), show_plot=False, verbose=False
    )
    out = pd.read_csv(path)
    assert list(out.columns) == ["label", "mean", "std", "count"]


def test_homogeneity_save_csv_requires_path():
    with pytest.raises(ValueError, match="csv_path"):
        hs.plot_protein_homogeneity_silhouette(_pro_df(), save_csv=True, show_plot=False, verbose=False)


@pytest.mark.xfail(
    strict=True,
    raises=FileNotFoundError,
    reason=(
        "homogeneity_silhouette.py: os.makedirs(os.path.dirname(csv_path)) is called "
        "with '' when csv_path is a bare filename, raising FileNotFoundError."
    ),
)
def test_homogeneity_save_csv_bare_filename(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    hs.plot_protein_homogeneity_silhouette(
        _pro_df(), save_csv=True, csv_path="stats.csv", show_plot=False, verbose=False
    )
    assert (tmp_path / "stats.csv").exists()
