import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from PRO_Utils import hto_calling as hto
from _cnv_pro_fakes import FakeAssay


N_NEG, N_POS = 300, 100


def _bimodal(seed=0, name="HTO1"):
    rng = np.random.default_rng(seed)
    vals = np.r_[rng.poisson(5, N_NEG), rng.poisson(500, N_POS)]
    return pd.Series(vals, index=[f"bc{i}" for i in range(len(vals))], name=name)


def _unimodal(seed=1, name="HTO2"):
    rng = np.random.default_rng(seed)
    vals = rng.poisson(20, N_NEG + N_POS)
    return pd.Series(vals, index=[f"bc{i}" for i in range(len(vals))], name=name)


@pytest.fixture(autouse=True)
def _close_figs():
    yield
    plt.close("all")


# --------------------------------------------------------------------------
# call_single_hashtag
# --------------------------------------------------------------------------
def test_bimodal_call():
    res = hto.call_single_hashtag(_bimodal())
    assert res.hashtag == "HTO1"
    assert res.bimodal is True
    assert res.note == ""
    assert (res.call[:N_NEG] == "Negative").all()
    assert (res.call[N_NEG:] == "Positive").all()
    assert res.p_pos.shape == (N_NEG + N_POS,)
    assert np.all((res.p_pos >= 0) & (res.p_pos <= 1))
    assert res.means[0] < res.means[1]
    assert res.means[0] == pytest.approx(np.log1p(5), abs=0.5)
    assert res.means[1] == pytest.approx(np.log1p(500), abs=0.2)
    assert res.threshold is not None and 5 < res.threshold < 500
    assert res.gmm is not None


def test_bimodal_call_is_deterministic():
    a = hto.call_single_hashtag(_bimodal())
    b = hto.call_single_hashtag(_bimodal())
    np.testing.assert_array_equal(a.call, b.call)
    np.testing.assert_allclose(a.p_pos, b.p_pos)
    assert a.threshold == b.threshold


def test_unimodal_call_all_negative():
    res = hto.call_single_hashtag(_unimodal())
    assert res.bimodal is False
    assert (res.call == "Negative").all()
    assert (res.p_pos == 0).all()
    assert res.threshold is None
    assert "unimodal" in res.note
    assert len(res.means) == 2


def test_min_mean_separation_can_reject_split():
    res = hto.call_single_hashtag(_bimodal(), min_mean_separation=100.0)
    assert res.bimodal is False
    assert (res.call == "Negative").all()


def test_uncertain_band():
    # Overlapping components so some posteriors land mid-band.
    rng = np.random.default_rng(3)
    vals = np.r_[rng.poisson(10, 300), rng.poisson(60, 300)]
    s = pd.Series(vals, name="HTO")
    banded = hto.call_single_hashtag(s, uncertain_band=(0.05, 0.95))
    hard = hto.call_single_hashtag(s, uncertain_band=None)
    assert banded.bimodal and hard.bimodal
    assert set(hard.call) <= {"Positive", "Negative"}
    unc = banded.call == "Uncertain"
    assert unc.any()
    assert np.all((banded.p_pos[unc] > 0.05) & (banded.p_pos[unc] < 0.95))
    # Outside the band the calls agree with the hard split.
    np.testing.assert_array_equal(banded.call[~unc], hard.call[~unc])


def test_posterior_cutoff_monotone():
    s = _bimodal()
    lo = hto.call_single_hashtag(s, posterior_cutoff=0.01, uncertain_band=None)
    hi = hto.call_single_hashtag(s, posterior_cutoff=0.99, uncertain_band=None)
    assert (lo.call == "Positive").sum() >= (hi.call == "Positive").sum()


def test_single_cell_raises():
    with pytest.raises(ValueError):
        hto.call_single_hashtag(pd.Series([10], name="H"))


# --------------------------------------------------------------------------
# _extract_hto_counts
# --------------------------------------------------------------------------
def _counts_df():
    return pd.DataFrame({"HTO1": _bimodal().values, "HTO2": _unimodal().values, "CD3": 1},
                        index=_bimodal().index)


def _tapestri_assay(df):
    return FakeAssay({"read_counts": df.values}, list(df.index), list(df.columns))


def test_extract_from_tapestri_assay():
    df = _counts_df()
    out = hto._extract_hto_counts(_tapestri_assay(df), ["HTO2", "HTO1"])
    assert list(out.columns) == ["HTO2", "HTO1"]
    assert list(out.index) == list(df.index)


def test_extract_falls_back_to_to_df():
    df = _counts_df()

    class WithToDf:
        def to_df(self):
            return df

    out = hto._extract_hto_counts(WithToDf(), ["HTO1"])
    out.iloc[0, 0] = -1  # returns a copy
    assert df.iloc[0, 0] != -1


def test_extract_falls_back_to_anndata_like():
    class AnnLike:
        X = np.array([[1, 2], [3, 4]])
        obs_names = ["a", "b"]
        var_names = ["H1", "H2"]

    out = hto._extract_hto_counts(AnnLike(), ["H2"])
    assert out["H2"].tolist() == [2, 4]


def test_extract_missing_hashtag():
    with pytest.raises(KeyError, match="NOPE"):
        hto._extract_hto_counts(_tapestri_assay(_counts_df()), ["HTO1", "NOPE"])


# --------------------------------------------------------------------------
# call_hto_present_absent / filter_assay_to_hto_positive / plot
# --------------------------------------------------------------------------
def test_call_hto_present_absent():
    df = _counts_df()
    adata, summary, results = hto.call_hto_present_absent(
        _tapestri_assay(df), ["HTO1", "HTO2"], plot=False, verbose=False
    )
    assert adata.shape == (N_NEG + N_POS, 2)
    assert list(adata.var_names) == ["HTO1", "HTO2"]
    for col in ["HTO1", "HTO1_p_pos", "HTO1_call", "HTO2_call", "n_positive"]:
        assert col in adata.obs
    assert list(adata.obs["HTO1_call"].cat.categories) == ["Negative", "Uncertain", "Positive"]
    assert (adata.obs["n_positive"] == (adata.obs["HTO1_call"] == "Positive").astype(int)).all()

    assert list(summary.index) == ["HTO1", "HTO2"]
    assert summary.loc["HTO1", "bimodal"] and not summary.loc["HTO2", "bimodal"]
    assert summary.loc["HTO1", "n_positive"] == N_POS
    assert summary.loc["HTO1", "pct_positive"] == pytest.approx(25.0)
    assert summary.loc["HTO2", "n_positive"] == 0
    assert pd.isna(summary.loc["HTO2", "threshold_count"])
    assert set(results) == {"HTO1", "HTO2"}
    assert isinstance(results["HTO1"], hto.HTOResult)


def test_call_hto_forwards_kwargs_and_prints(capsys):
    df = _counts_df()
    _, summary, _ = hto.call_hto_present_absent(
        _tapestri_assay(df), ["HTO1"], plot=True, verbose=True, min_mean_separation=100.0
    )
    assert not summary.loc["HTO1", "bimodal"]
    assert "[HTO1] bimodal=False" in capsys.readouterr().out


@pytest.mark.parametrize("make", [_bimodal, _unimodal])
def test_plot_hashtag_distribution_smoke(make):
    s = make()
    res = hto.call_single_hashtag(s)
    fig, ax = plt.subplots()
    out = hto.plot_hashtag_distribution(s, res, ax=ax)
    assert out is ax
    assert ax.get_title().startswith(s.name)
    assert ("UNIMODAL" in ax.get_title()) is (not res.bimodal)
    # dashed threshold line only when a threshold exists
    assert any(l.get_linestyle() == "--" for l in ax.get_lines()) is (res.threshold is not None)
