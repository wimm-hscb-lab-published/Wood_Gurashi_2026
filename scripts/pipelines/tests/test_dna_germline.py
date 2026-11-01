"""Unit tests for DNA_Utils.germline helpers and cohort-calibration logic."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from DNA_Utils import germline as G


# ── Fixtures / helpers ───────────────────────────────────────────────────────

def make_dna(ngt, af=None, dp=None, ids=None, barcodes=None, labels=None):
    from missionbio.mosaic.dna import Dna

    ngt = np.asarray(ngt)
    n, m = ngt.shape
    if af is None:
        af = np.where(ngt == 1, 50.0, np.where(ngt == 2, 100.0, 0.0))
    if dp is None:
        dp = np.full(ngt.shape, 20)
    ids = np.array(ids if ids is not None else [f"chr1:{i + 1}:A/T" for i in range(m)])
    barcodes = np.array(barcodes if barcodes is not None else [f"BC{i}" for i in range(n)])
    labels = np.array(labels if labels is not None else ["c"] * n)
    return Dna(
        name="dna",
        metadata={"sample_name": np.array(["s1"])},
        layers={"NGT": ngt, "AF": np.asarray(af, dtype=float), "DP": np.asarray(dp)},
        row_attrs={"barcode": barcodes, "sample_name": np.array(["s1"] * n), "label": labels},
        col_attrs={"id": ids},
    )


def fake_sample(**col_attrs):
    return SimpleNamespace(dna=SimpleNamespace(col_attrs={k: np.asarray(v) for k, v in col_attrs.items()}))


# ── get_real_somatic_variants ────────────────────────────────────────────────

def test_get_real_somatic_variants_protein_filter():
    s = fake_sample(id=["v1", "v2", "v3", "v4"], Protein=["p.R175H", "p.p.?", "", "p.G12D"])
    out = G.get_real_somatic_variants([["v1", "v2"], ["v3", "v4"]], ["v4"], s)
    assert out.tolist() == ["v1"]


def test_get_real_somatic_variants_consequence_and_order():
    s = fake_sample(id=["a", "b", "c"], Protein=["", "", ""],
                    Consequence=["missense", "intron_variant", "INTRONIC"])
    assert G.get_real_somatic_variants(["c", "a", "b", "zz"], None, s).tolist() == ["a"]
    # Consequence ignored when disabled -> blank Protein => all intronic
    assert G.get_real_somatic_variants(["a"], None, s, use_consequence_if_available=False).size == 0


def test_get_real_somatic_variants_empty_and_missing_id():
    s = fake_sample(id=["a"])
    out = G.get_real_somatic_variants([], None, s)
    assert out.size == 0 and out.dtype.kind == "U"
    assert G.get_real_somatic_variants(["a"], ["a"], s).size == 0
    with pytest.raises(KeyError):
        G.get_real_somatic_variants(["a"], None, fake_sample(name=["a"]))


def test_get_real_somatic_variants_no_protein_column_keeps_all():
    s = fake_sample(id=["a", "b"])
    # Missing Protein column -> NaN fallback series -> 'nan' strings -> nothing
    # flagged intronic, so every variant present in the assay is kept.
    assert G.get_real_somatic_variants(["a", "b", "c"], None, s).tolist() == ["a", "b"]


@pytest.mark.xfail(
    strict=True,
    reason="germline.py:57-61 calls .astype(str) before .fillna(''), so a NaN Protein "
    "becomes the string 'nan' and is NOT treated as blank/intronic (fillna is a no-op).",
)
def test_get_real_somatic_variants_nan_protein_treated_as_blank():
    s = SimpleNamespace(dna=SimpleNamespace(col_attrs={
        "id": np.array(["a", "b"]), "Protein": np.array(["p.R1H", np.nan], dtype=object)}))
    assert G.get_real_somatic_variants(["a", "b"], None, s).tolist() == ["a"]


# ── Small pure helpers ───────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "vid, expected",
    [
        ("chr1:100:A/T", False),
        ("chr1:100:A/AT", True),
        ("chr1:100:AT/A", True),
        ("chr13:28602226:*/A+AAG", True),
        ("chr1:100:A/*", True),
        ("chr1:100:A/T+G", True),
        ("TP53:chr17:1:G/A", False),
        ("garbage", True),
        ("chr1:100:A", True),
        ("chr1:100:A/T/G", True),
    ],
)
def test_is_indel(vid, expected):
    assert G._is_indel(vid) is expected


def test_dedupe_keep_order():
    assert G._dedupe_keep_order(["b", "a", "b", None, np.nan, 1, "1"]) == ["b", "a", "1"]
    assert G._dedupe_keep_order([]) == []


@pytest.mark.parametrize(
    "x, expected",
    [
        (None, []),
        (["a", "b", "a"], ["a", "b"]),
        ({"a"}, ["a"]),
        (("x", "y"), ["x", "y"]),
        (pd.Index(["a", "a"]), ["a"]),
        (pd.Series(["s", None, "s"]), ["s"]),
        (pd.DataFrame({"foo": [1], "id": ["v1"]}), ["v1"]),
        (pd.DataFrame({"variant": ["v2"], "id": ["v1"]}), ["v2"]),
        (pd.DataFrame({"foo": ["first"], "bar": ["second"]}), ["first"]),
        ("ab", ["a", "b"]),  # a bare string is iterated character-wise
    ],
)
def test_normalise_variant_list(x, expected):
    assert G._normalise_variant_list(x) == expected


@pytest.mark.parametrize(
    "s, expected",
    [
        (pd.Series([True, False]), [True, False]),
        (pd.Series([True, None], dtype="boolean"), [True, False]),
        (pd.Series([0, 1, 2, np.nan]), [False, True, True, False]),
        (pd.Series(["True", " yes", "Y", "t", "1", "no", "false", None, "0"]),
         [True, True, True, True, True, False, False, False, False]),
        (pd.Series([True, None, "x"], dtype=object), [True, False, False]),
    ],
)
def test_coerce_bool_series(s, expected):
    assert G._coerce_bool_series(s).tolist() == expected


def test_coerce_bool_series_none():
    assert G._coerce_bool_series(None) is None


def test_coerce_bool_like_extra_tokens():
    s = pd.Series(["pass", "PASSED", "positive", "informative", "fail", "1"])
    assert G._coerce_bool_like(s).tolist() == [True, True, True, True, False, True]
    assert G._coerce_bool_like(None) is None
    assert G._coerce_bool_like(pd.Series([0.0, 3.0, np.nan])).tolist() == [False, True, False]


@pytest.mark.parametrize(
    "args, kw, expected",
    [
        ((0.1, 0.8, 0.1, 50), {}, "het"),
        ((0.1, np.nan, 0.1, 50), {}, "uncalled"),
        ((0.1, None, 0.1, 50), {}, "uncalled"),
        ((0.1, 0.8, 0.1, 0), {}, "uncalled"),
        ((0.1, 0.8, 0.1, None), {}, "uncalled"),
        ((0.1, 0.8, 0.1, 5), {"min_cells": 10}, "low_coverage"),
        ((0.0, 0.5, 0.5, 50), {}, "hom_alt"),
        ((0.95, 0.05, 0.0, 50), {}, "hom_ref"),
        ((0.5, 0.4, 0.1, 50), {}, "low_occupancy"),
        ((0.1, 0.8, np.nan, 50), {}, "het"),
        ((np.nan, 0.8, 0.1, 50), {}, "het"),
        # boundaries are inclusive
        ((0.0, 0.55, 0.2, 50), {}, "hom_alt"),
        ((0.9, 0.55, 0.0, 50), {}, "hom_ref"),
        ((0.3, 0.55, 0.15, 50), {}, "het"),
    ],
)
def test_donor_genotype_call(args, kw, expected):
    kwargs = dict(min_het=0.55, max_hom=0.20, max_wt=0.90)
    kwargs.update(kw)
    assert G._donor_genotype_call(*args, **kwargs) == expected


def test_safe_patient_export_id():
    assert G._safe_patient_export_id("Exp 1", "P/01") == "Exp_1_P_01"
    assert G._safe_patient_export_id("__a", "b!!") == "a_b"
    assert G._safe_patient_export_id("x.y", "z-1") == "x.y_z-1"


def test_get_clone_columns_from_clone_df():
    df = pd.DataFrame(columns=["Variant\nname", "WT", "TP53 Het", "Missing GT clones (3)",
                               "Small subclones (7)", "ADO", "fp", "Clone 2"])
    assert G._get_clone_columns_from_clone_df(df) == ["WT", "TP53 Het", "Clone 2"]
    assert G._get_clone_columns_from_clone_df(None) == []
    assert G._get_clone_columns_from_clone_df(["not", "df"]) == []


@pytest.mark.parametrize(
    "v, expected",
    [
        ("chr1:100:A/T", (1, 100, "chr1:100:A/T")),
        ("chrX:5:G/A", (23, 5, "chrX:5:G/A")),
        ("chrY:5:G/A", (24, 5, "chrY:5:G/A")),
        ("chrMT:5:G/A", (25, 5, "chrMT:5:G/A")),
        ("chrM:5:G/A", (25, 5, "chrM:5:G/A")),
        ("2:7:C/G", (2, 7, "2:7:C/G")),
        ("chrUn:7:C/G", (999, 7, "chrUn:7:C/G")),
        ("bad", (999, np.inf, "bad")),
        ("chr1:100", (999, np.inf, "chr1:100")),
    ],
)
def test_chr_pos_key(v, expected):
    assert G._chr_pos_key(v) == expected


def test_chr_pos_key_sorting():
    vs = ["chrX:1:A/T", "chr10:5:A/T", "chr2:300:A/T", "chr2:30:A/T", "junk"]
    assert sorted(vs, key=G._chr_pos_key) == [
        "chr2:30:A/T", "chr2:300:A/T", "chr10:5:A/T", "chrX:1:A/T", "junk"]


def test_variant_key_series():
    out = G._variant_key_series([" chr1:1:A/T ", "chr1: 2 :A/T", "x\ty"])
    assert out.tolist() == ["chr1:1:A/T", "chr1:2:A/T", "xy"]


def test_extract_variant_list():
    assert G._extract_variant_list(None) == []
    assert G._extract_variant_list(["a ", "a", "b"]) == ["a", "b"]
    assert G._extract_variant_list(np.array(["a", "b"])) == ["a", "b"]
    assert G._extract_variant_list(pd.Series(["x", "x"])) == ["x"]
    assert G._extract_variant_list("solo") == ["solo"]
    assert G._extract_variant_list(pd.DataFrame({"Variant_ID": ["q"]})) == ["q"]
    with pytest.raises(KeyError):
        G._extract_variant_list(pd.DataFrame({"nope": [1]}))


def test_first_existing_col():
    df = pd.DataFrame(columns=["a", "b"])
    assert G._first_existing_col(df, ["z", "b", "a"]) == "b"
    assert G._first_existing_col(df, ["z"]) is None
    assert G._first_existing_col(df, []) is None


def test_normalise_cell_clone_labels():
    s = G._normalise_cell_clone_labels({"b1": 1, "b2": "WT"})
    assert s.name == "clone_label" and s.to_dict() == {"b1": "1", "b2": "WT"}

    df = pd.DataFrame({"barcode": [1, 2], "clone_label": ["A", "B"]})
    assert G._normalise_cell_clone_labels(df).to_dict() == {"1": "A", "2": "B"}

    ser = pd.Series([3, 4], index=[10, 11])
    out = G._normalise_cell_clone_labels(ser)
    assert out.index.tolist() == ["10", "11"] and out.tolist() == ["3", "4"]

    with pytest.raises(ValueError):
        G._normalise_cell_clone_labels(pd.DataFrame({"x": [1]}))
    with pytest.raises(TypeError):
        G._normalise_cell_clone_labels(["a"])


def test_get_empirical_cohort_germline_thresholds():
    full = G.get_empirical_cohort_germline_thresholds()
    assert full == G.EMPIRICAL_COHORT_GERMLINE_THRESHOLDS
    assert full is not G.EMPIRICAL_COHORT_GERMLINE_THRESHOLDS
    slim = G.get_empirical_cohort_germline_thresholds(include_audit_metrics=False)
    assert "usable_anchor_recall" not in slim
    assert {"usable_min_frac_het", "usable_max_frac_hom", "usable_max_frac_wt",
            "clean_min_frac_het"} <= set(slim)
    full["min_frac_het"] = -1
    assert G.EMPIRICAL_COHORT_GERMLINE_THRESHOLDS["min_frac_het"] == 0.601


# ── Assay-backed genotype structure ──────────────────────────────────────────

@pytest.fixture
def structure_sample():
    # 10 cells x 3 variants
    ngt = np.array([
        [1, 0, 2],
        [1, 0, 2],
        [1, 0, 2],
        [1, 0, 2],
        [1, 1, 2],
        [1, 0, 2],
        [0, 0, 1],
        [0, 0, 2],
        [3, 3, 3],
        [1, 0, 2],
    ])
    af = np.where(ngt == 1, np.linspace(40, 58, 10)[:, None], np.where(ngt == 2, 99.0, 1.0))
    dp = np.full(ngt.shape, 20)
    dp[9, :] = 2  # low depth, excluded at min_dp=5
    ids = ["chr1:10:A/T", "chr2:20:C/G", "chr3:30:G/A"]
    return SimpleNamespace(dna=make_dna(ngt, af=af, dp=dp, ids=ids))


def test_compute_genotype_structure(structure_sample):
    out = G.compute_genotype_structure(
        structure_sample, ["chr1:10:A/T", "chr2:20:C/G", "chr3:30:G/A", "absent"],
        min_dp=5, lower=40, upper=60)
    assert out["variant"].tolist() == ["chr1:10:A/T", "chr2:20:C/G", "chr3:30:G/A"]
    r = out.set_index("variant")
    # NGT=3 cell and DP<5 cell excluded -> 8 cells
    assert (r["n_cells"] == 8).all()
    assert r.loc["chr1:10:A/T", "frac_HET"] == pytest.approx(6 / 8)
    assert r.loc["chr1:10:A/T", "frac_WT"] == pytest.approx(2 / 8)
    assert r.loc["chr2:20:C/G", "frac_WT"] == pytest.approx(7 / 8)
    assert r.loc["chr3:30:G/A", "frac_HOM"] == pytest.approx(7 / 8)
    assert bool(r.loc["chr3:30:G/A", "hom_dominated"]) is True
    assert bool(r.loc["chr1:10:A/T", "hom_dominated"]) is False
    assert r.loc["chr2:20:C/G", "het_af_median"] == pytest.approx(48.0)  # single HET cell
    assert r.loc["chr1:10:A/T", "margin_to_lower"] == pytest.approx(
        r.loc["chr1:10:A/T", "het_af_p10"] - 40)


def test_compute_genotype_structure_empty(structure_sample):
    out = G.compute_genotype_structure(structure_sample, ["nope"])
    assert out.empty


def test_population_structure_table(structure_sample):
    t = G.population_structure_table(
        structure_sample, ["chr1:10:A/T", "chr2:20:C/G", "chr3:30:G/A"],
        label="wt", min_cells=5)
    assert list(t.columns) == ["variant", "wt_n_cells", "wt_frac_WT", "wt_frac_HET",
                               "wt_frac_HOM", "wt_het_af_median", "wt_donor_genotype"]
    assert t.set_index("variant")["wt_donor_genotype"].to_dict() == {
        "chr1:10:A/T": "het", "chr2:20:C/G": "low_occupancy", "chr3:30:G/A": "hom_alt"}
    t2 = G.population_structure_table(structure_sample, ["chr1:10:A/T"], min_cells=100)
    assert t2["wt_donor_genotype"].tolist() == ["low_coverage"]
    assert G.population_structure_table(structure_sample, ["missing"]).columns.tolist() == ["variant"]


def test_add_population_concordance(structure_sample):
    all_info = pd.DataFrame({
        "variant": ["chr1:10:A/T", "chr3:30:G/A", "chrZ:1:A/T"],
        "donor_genotype": ["het", "het", "het"],
    })
    out = G.add_population_concordance(all_info, structure_sample, min_called_cells_for_genotype=5,
                                       verbose=False)
    assert out["wt_germline_het_seed"].tolist() == [True, False, False]
    assert out["concordant_germline_het"].tolist() == [True, False, False]
    assert len(out) == 3


def test_genotype_af_cutoffs_no_plot():
    rng = np.random.default_rng(0)
    n_cells, n_var = 200, 5
    af = rng.uniform(0, 100, size=(n_cells, n_var))
    ngt = np.where(af < 25, 0, np.where(af > 80, 2, 1))
    dna = make_dna(ngt, af=af, dp=np.full(ngt.shape, 30))
    cut, binned, bands, placement = G.genotype_af_cutoffs(
        SimpleNamespace(dna=dna), af_bin=5.0, min_bin_n=10, plot=False,
        evaluate_variants=["chr1:1:A/T"])
    assert 20 <= cut["lower"] <= 30
    assert 75 <= cut["upper"] <= 85
    assert len(binned) == 20
    assert bands["n_cells"].tolist() == [0, n_cells * n_var, 0]
    assert np.isnan(bands.loc[0, "lower"])
    assert placement["variant"].tolist() == ["chr1:1:A/T"]


# ── Cohort calibration ───────────────────────────────────────────────────────

@pytest.fixture
def pooled():
    return pd.DataFrame({
        "variant": ["v1", "v1", "v1", "v2", "v2", "v3", "v4"],
        "sample_id": ["s1", "s2", "s3", "s1", "s2", "s1", "s2"],
        "frac_HET": [0.8, 0.7, 0.9, 0.3, 0.6, 0.9, 0.7],
        "frac_HOM": [0.05, 0.1, 0.0, 0.6, 0.1, 0.0, 0.05],
        "frac_WT": [0.15, 0.2, 0.1, 0.1, 0.3, 0.1, 0.25],
        "AF_HET_median": [50, 48, 52, 70, 45, 51, 49],
        "donor_genotype": ["het", "het", "het", "hom_alt", "het", "het", "uncalled"],
        "usable_germline_het_seed": [True, True, True, False, False, True, False],
        "is_blacklisted": [False, False, False, False, False, True, False],
        "germline_broad_class": ["", "", "", "likely_hom_alt", "", "", ""],
    })


def test_resolve_calibration_columns_priority():
    df = pd.DataFrame(columns=["frac_HET", "placement_frac_HET", "variant_id", "z"])
    cm = G.resolve_calibration_columns(df)
    assert cm == {"frac_het": "placement_frac_HET", "variant": "variant_id", "af_z": "z"}
    assert G.resolve_calibration_columns(pd.DataFrame()) == {}


def test_cal_col_helpers(pooled):
    cm = G.resolve_calibration_columns(pooled)
    assert G._cal_col(pooled, cm, "missing") is None
    assert G._cal_col(pooled, {"x": "nope"}, "x") is None
    num = G._numeric_cal_col(pooled, cm, "af_z", default=7.0)
    assert (num == 7.0).all() and num.dtype == float
    assert G._numeric_cal_col(pd.DataFrame({"f": ["1", "x"]}), {"frac_het": "f"},
                              "frac_het").tolist()[0] == 1.0
    assert G._bool_cal_col(pooled, cm, "pathogenic").sum() == 0
    assert G._bool_cal_col(pooled, cm, "blacklisted").tolist() == [False] * 5 + [True, False]
    st = G._string_cal_col(pd.DataFrame({"status": ["a", None]}), {"status": "status"}, "status")
    assert st.tolist() == ["a", ""]
    assert G._string_cal_col(pooled, cm, "status", default="z").eq("z").all()


def test_masks(pooled):
    cm = G.resolve_calibration_columns(pooled)
    assert G._het_mask(pooled, cm).tolist() == [True, True, True, False, True, True, False]
    assert G._not_excluded_mask(pooled, cm).tolist() == [True, True, True, False, True, False, True]

    # Fallbacks
    df = pooled.drop(columns=["donor_genotype"]).assign(n_cells=[0, 1, 2, 3, 4, 5, np.nan])
    cm2 = G.resolve_calibration_columns(df)
    assert G._het_mask(df, cm2).tolist() == df["usable_germline_het_seed"].tolist()
    with pytest.raises(ValueError):
        G._het_mask(pd.DataFrame({"a": [1]}), {})


def test_apply_calibrated_germline_rules(pooled):
    th = {"min_frac_het": 0.65, "max_frac_hom": 0.15, "max_frac_wt": 0.25,
          "af_lower": 45, "af_upper": 55, "usable_min_frac_het": 0.5}
    out = G.apply_calibrated_germline_rules(pooled, th)
    assert out["calibrated_clean_het_seed"].tolist() == [True, True, True, False, False, False, False]
    # row 4: frac_wt .3 > .25 and not a usable seed -> not practical
    assert out["calibrated_usable_het_seed"].tolist() == [True, True, True, False, False, False, False]
    # row 5 was a usable seed but blacklisted -> rejected
    assert out["calibrated_rejected_het_seed"].tolist() == [False] * 5 + [True, False]
    reasons = out["calibrated_reject_reasons"].tolist()
    assert reasons[0] == ""
    assert reasons[3] == "excluded_or_bad_class;not_donor_het;low_frac_HET;high_frac_HOM"
    assert reasons[4] == "high_frac_WT"
    assert reasons[5] == "excluded_or_bad_class"
    assert reasons[6] == "not_donor_het"
    assert "calibrated_clean_het_seed" not in pooled.columns  # input untouched


def test_apply_calibrated_rules_rescue_and_concordant(pooled):
    p = pooled.copy()
    p.loc[4, "status"] = "af_skewed"
    p["status"] = p["status"].fillna("")
    p["review_verdict"] = ["", "", "", "", "likely_usable_het", "", ""]
    p["concordant_germline_het"] = [False, False, False, False, False, False, False]
    th = {"min_frac_het": 0.99}
    out = G.apply_calibrated_germline_rules(p, th)
    assert out["calibrated_rescued_het_seed"].tolist() == [False] * 4 + [True, False, False]
    assert out.loc[4, "calibrated_usable_het_seed"]
    assert not out["calibrated_clean_het_seed"].any()


def test_apply_calibrated_rules_empty_thresholds(pooled):
    out = G.apply_calibrated_germline_rules(pooled, {})
    # No numeric thresholds: clean == allowed & donor_het
    assert out["calibrated_clean_het_seed"].tolist() == [True, True, True, False, True, False, False]
