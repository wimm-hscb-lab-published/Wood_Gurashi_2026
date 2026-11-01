"""Unit tests for DNA_Utils.qc_filters helpers and per-cell QC flaggers."""
from types import SimpleNamespace

import matplotlib
import numpy as np
import pandas as pd
import pytest

from DNA_Utils import qc_filters as Q


# ── _corr_complete_obs ───────────────────────────────────────────────────────

def test_corr_complete_obs_basic():
    x = np.arange(50, dtype=float)
    assert Q._corr_complete_obs(x, 2 * x + 1) == pytest.approx(1.0)
    assert Q._corr_complete_obs(x, -x) == pytest.approx(-1.0)


def test_corr_complete_obs_nan_and_min_shared():
    x = np.arange(50, dtype=float)
    y = x.copy()
    y[:15] = np.nan
    assert np.isnan(Q._corr_complete_obs(x, y, min_shared=40))  # 35 shared
    assert Q._corr_complete_obs(x, y, min_shared=35) == pytest.approx(1.0)


def test_corr_complete_obs_constant():
    x = np.arange(50, dtype=float)
    assert np.isnan(Q._corr_complete_obs(x, np.ones(50)))
    assert np.isnan(Q._corr_complete_obs(np.array([]), np.array([]), min_shared=0))


# ── piecewise_elbow_threshold ────────────────────────────────────────────────

def test_piecewise_elbow_too_few_values():
    thr, k, info = Q.piecewise_elbow_threshold(np.arange(19))
    assert np.isnan(thr) and k == -1 and info == {}


def test_piecewise_elbow_drops_nan_before_length_check():
    vals = np.r_[np.arange(15, dtype=float), [np.nan] * 10]
    thr, k, _ = Q.piecewise_elbow_threshold(vals)
    assert np.isnan(thr) and k == -1


def test_piecewise_elbow_kmax_le_kmin():
    thr, k, info = Q.piecewise_elbow_threshold(np.arange(30), max_frac=0.1)
    assert np.isnan(thr) and k == -1 and info == {}


def test_piecewise_elbow_finds_knee():
    # 10 steep values, then 90 nearly-flat ones
    steep = np.linspace(100, 20, 10)
    flat = np.linspace(10, 0, 90)
    rng = np.random.default_rng(1)
    vals = rng.permutation(np.r_[steep, flat])
    thr, k, info = Q.piecewise_elbow_threshold(vals)
    assert 8 <= k <= 12
    assert info["ratio"] >= 3.0
    y = np.sort(vals)[::-1]
    assert thr == y[k]


def test_piecewise_elbow_fallback_when_no_ratio():
    # perfectly linear curve: slope ratio ~1, falls back to min SSE split
    thr, k, info = Q.piecewise_elbow_threshold(np.arange(100, dtype=float),
                                               slope_ratio_min=3.0)
    assert info["ratio"] == pytest.approx(1.0, rel=1e-6)
    assert 5 <= k <= 40
    assert np.isfinite(thr)


# ── Clustering helpers ───────────────────────────────────────────────────────

def test_prep_for_clustering_imputes_column_median():
    A = np.array([[1.0, np.nan], [3.0, np.nan], [np.nan, np.nan]])
    out = Q._prep_for_clustering(A)
    assert out[2, 0] == 2.0
    assert (out[:, 1] == 0.0).all()
    assert A[2, 0] != A[2, 0]  # input not modified
    assert Q._prep_for_clustering(np.empty((0, 3))).size == 0


def test_cluster_columns_small_and_order():
    df = pd.DataFrame(np.ones((4, 2)), columns=["a", "b"])
    assert Q._cluster_columns(df) == ["a", "b"]

    rng = np.random.default_rng(0)
    base1, base2 = rng.normal(size=30), rng.normal(size=30)
    df = pd.DataFrame({
        "a1": base1, "b1": base2, "a2": base1 + 0.01, "const": 1.0, "b2": base2 + 0.01,
    })
    order = Q._cluster_columns(df)
    assert sorted(order) == sorted(df.columns)
    assert order[-1] == "const"  # zero-variance column appended at tail
    assert abs(order.index("a1") - order.index("a2")) == 1
    assert abs(order.index("b1") - order.index("b2")) == 1


def test_cluster_columns_subsamples_rows():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(size=(50, 4)), columns=list("wxyz"))
    assert sorted(Q._cluster_columns(df, max_cells=10)) == list("wxyz")


def test_cluster_row_index():
    df = pd.DataFrame(np.ones((2, 3)), index=["r1", "r2"])
    assert list(Q._cluster_row_index(df)) == ["r1", "r2"]

    rng = np.random.default_rng(0)
    p1, p2 = rng.normal(size=10), rng.normal(size=10)
    df = pd.DataFrame([p1, p2, p1 + 0.01, np.zeros(10), p2 + 0.01],
                      index=["a1", "b1", "a2", "z", "b2"])
    idx = list(Q._cluster_row_index(df))
    assert sorted(idx) == sorted(df.index) and idx[-1] == "z"
    assert abs(idx.index("a1") - idx.index("a2")) == 1

    big = pd.DataFrame(rng.normal(size=(40, 5)))
    assert len(Q._cluster_row_index(big, max_cells=7)) == 7


def test_cluster_group_order():
    rng = np.random.default_rng(0)
    p1, p2 = rng.normal(size=8), rng.normal(size=8)
    rows = [p1, p1, p2, p2, p1 + 0.01, p2 + 0.01]
    labels = pd.Series(["A", "A", "B", "B", "C", "D"], index=range(6))
    df = pd.DataFrame(rows, index=range(6))
    out = Q._cluster_group_order(df, labels, ["A", "B", "C", "D", "missing"])
    assert sorted(out) == ["A", "B", "C", "D", "missing"]
    assert out[-1] == "missing"
    assert abs(out.index("A") - out.index("C")) == 1
    # Fewer than 3 groups -> unchanged
    assert Q._cluster_group_order(df, labels, ["A", "B"]) == ["A", "B"]


# ── Labels / palettes / colormaps ────────────────────────────────────────────

def test_natkey_sort():
    vals = ["10", "2", "b", "1", "a", "Mixed"]
    assert sorted(vals, key=Q._natkey) == ["1", "2", "10", "Mixed", "a", "b"]
    assert Q._natkey(3) == (0, 3)


def test_group_palette_stable_by_universe():
    p_all = Q._group_palette(["b", "a"], all_labels=["c", "a", "b"])
    p_sub = Q._group_palette(["a"], all_labels=["a", "b", "c"])
    assert p_all["a"] == p_sub["a"]
    assert set(p_all) == {"a", "b", "c"}
    p_tab = Q._group_palette(["x", "y"], source="tab20")
    assert p_tab["x"] == matplotlib.colormaps["tab20"].colors[0]


def test_group_palette_cycles():
    labels = [f"L{i:03d}" for i in range(100)]
    p = Q._group_palette(labels)
    base = Q._missionbio_colors()
    assert p["L000"] == base[0]
    assert p[labels[len(base)]] == base[0]


def test_palette_from_plotly_scatter():
    import plotly.graph_objects as go

    fig = go.Figure([
        go.Scatter(x=[0], y=[0], name="A", marker=dict(color="red")),
        go.Scatter(x=[0], y=[0], name="B", marker=dict(color=["blue", "blue"])),
        go.Scatter(x=[0], y=[0], name="C", marker=dict(color=[1.0, 2.0])),  # numeric -> skip
        go.Scatter(x=[0], y=[0], name="D"),
    ])
    assert Q.palette_from_plotly_scatter(fig) == {"A": "red", "B": "blue"}


def test_palette_from_plotly_scatter_fallback():
    empty = SimpleNamespace(data=[])
    assert Q.palette_from_plotly_scatter(empty) == {}
    pal = Q.palette_from_plotly_scatter(empty, label_values=["y", "x", "y"])
    assert set(pal) == {"x", "y"}
    assert Q.palette_from_plotly_scatter(object()) == {}  # no .data -> swallowed


def test_coerce_palette():
    assert Q._coerce_palette({1: "red"}) == {"1": "red"}
    pal = Q._coerce_palette(None, groups=["b", "a"])
    assert set(pal) == {"a", "b"}
    pal2 = Q._coerce_palette(object(), labels=["q"])
    assert set(pal2) == {"q"}
    assert Q._coerce_palette(None) == {}


def test_auto_scatterplot_palette():
    import plotly.graph_objects as go

    assert Q._auto_scatterplot_palette(SimpleNamespace()) == {}

    calls = []

    def scatter(attribute, colorby):
        calls.append(attribute)
        if attribute == "umap":
            raise RuntimeError("no umap")
        return go.Figure([go.Scatter(x=[0], y=[0], name=1, marker=dict(color="green"))])

    assay = SimpleNamespace(scatterplot=scatter)
    assert Q._auto_scatterplot_palette(SimpleNamespace(dna=assay)) == {"1": "green"}
    assert calls == ["umap", "pca"]


def test_resolve_cmap():
    cm = Q._resolve_cmap(None)
    assert cm.name == "tapestri_af_white_to_red"
    assert cm.N == len(Q.AF_HEATMAP_CMAP_COLORS)
    bad = matplotlib.colors.to_rgba("#dcdcdc")
    assert tuple(cm(np.ma.masked_invalid([np.nan]))[0]) == pytest.approx(bad)
    v = Q._resolve_cmap("viridis", nan_color="red")
    assert v.name == "viridis"
    assert tuple(v(np.ma.masked_invalid([np.nan]))[0]) == pytest.approx((1, 0, 0, 1))
    # Global viridis not mutated
    assert matplotlib.colormaps["viridis"].get_bad()[0] != 1 or \
        tuple(matplotlib.colormaps["viridis"].get_bad()) != (1, 0, 0, 1)


@pytest.mark.parametrize(
    "args, expected",
    [
        ((0, 0), (8.0 + 1 / 800, 8.0)),  # clamped to >= 1 row/col
        ((11500, 1600), (10.0, 6.0 + 11500 / 320)),
        ((10 ** 7, 10 ** 7), (18.0, 55.0)),
    ],
)
def test_auto_heatmap_figsize(args, expected):
    assert Q._auto_heatmap_figsize(*args) == pytest.approx(expected)


def test_auto_heatmap_figsize_explicit():
    assert Q._auto_heatmap_figsize(10, 10, width=3, height=4) == (3.0, 4.0)


# ── Assay-label helpers & profile correlation ────────────────────────────────

class _FakeAssay:
    def __init__(self, af, labels):
        self._af = af
        self.row_attrs = {"label": np.asarray(labels)}
        self.set_calls = []

    def get_attribute(self, attr, constraint=None):
        return self._af

    def set_labels(self, clusters):
        self.set_calls.append(clusters)


# ── Per-cell QC flaggers ─────────────────────────────────────────────────────

def _donor_frame(n_cells=60, n_var=40, n_bad=3, seed=0):
    rng = np.random.default_rng(seed)
    profile = rng.choice([0.0, 0.5, 1.0], size=n_var)
    profile[: n_var // 2] = 0.5
    A = profile + rng.normal(0, 0.02, size=(n_cells, n_var))
    A[:n_bad] = rng.uniform(0, 1, size=(n_bad, n_var))  # garbage cells
    df = pd.DataFrame(np.clip(A, 0, 1) * 100, columns=[f"v{i}" for i in range(n_var)])
    df["label"] = "D1"
    return df


def test_flag_amplicon_loss_cells_flags_garbage():
    df = _donor_frame()
    fail, scores = Q.flag_amplicon_loss_cells(df, min_stable_loci=5, q=0.5)
    assert fail[:3].all()
    assert fail[3:].sum() == 0
    assert list(scores.columns) == ["noise_mad", "one_minus_corr", "nan_frac",
                                    "composite_score", "threshold", "n_stable_loci"]
    assert (scores["n_stable_loci"] > 0).all()


def test_flag_amplicon_loss_cells_default_quantile_caps_flag_rate():
    # Threshold is max(MAD rule, quantile q); with the default q=0.995 at most
    # ~0.5% (here: the single worst) of a donor's cells can ever be flagged.
    df = _donor_frame(n_bad=3)
    fail, scores = Q.flag_amplicon_loss_cells(df, min_stable_loci=5)
    assert fail.sum() == 1 and fail[:3].any()


def test_flag_amplicon_loss_cells_skips_small_and_excluded():
    df = _donor_frame(n_cells=10)
    df.loc[:, "label"] = "Mixed"
    fail = Q.flag_amplicon_loss_cells(df, return_scores=False)
    assert not fail.any() and fail.dtype == bool

    df2 = _donor_frame(n_cells=10)
    fail2, scores2 = Q.flag_amplicon_loss_cells(df2, min_cells=20)
    assert not fail2.any() and scores2["composite_score"].isna().all()


def test_flag_amplicon_loss_cells_missing_label():
    with pytest.raises(AssertionError):
        Q.flag_amplicon_loss_cells(pd.DataFrame({"v": [1]}))


def test_flag_spurious_gain_cells():
    n_cells, n_var = 30, 20
    A = np.zeros((n_cells, n_var))
    A[:, :5] = 50.0
    A[0, 5:] = 80.0  # cell 0 spikes in all silent variants
    A[1, 5:8] = 80.0  # 3/15 = 0.2 below threshold
    df = pd.DataFrame(A, columns=[f"v{i}" for i in range(n_var)])
    df["label"] = "D"
    fail, scores = Q.flag_spurious_gain_cells(df)
    assert fail.tolist() == [True] + [False] * (n_cells - 1)
    assert scores.loc[0, "frac_spike_vs_silent"] == pytest.approx(1.0)
    assert scores.loc[1, "frac_spike_vs_silent"] == pytest.approx(0.2)
    assert (scores["silent_count"] == 15).all()

    # not enough silent variants -> nothing flagged
    fail2 = Q.flag_spurious_gain_cells(df, min_silent=16, return_scores=False)
    assert not np.asarray(fail2).any()
