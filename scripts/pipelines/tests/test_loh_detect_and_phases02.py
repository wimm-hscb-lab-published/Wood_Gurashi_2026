"""Tests for LOH_Pipeline/detect_loh.py, phase0_setup.py and phase2.py."""
import numpy as np
import pandas as pd
import pytest

from LOH_Pipeline.detect_loh import detect_loh_and_update_clones
from LOH_Pipeline.loh_config import LOHConfig, PipelineState
from LOH_Pipeline.phase0_setup import precheck_variant_availability, run_phase0


# ─────────────────────────────────────────────────────────────────────
# Fakes
# ─────────────────────────────────────────────────────────────────────
class _FakeDNA:
    def __init__(self, barcodes, clones):
        self.row_attrs = {
            "barcode": np.array(barcodes, dtype=object),
            "Clone": np.array(clones, dtype=object),
        }
        self.col_attrs = {}

    def get_attribute(self, *a, **k):
        raise KeyError("DP")


class _FakeSample:
    def __init__(self, barcodes, clones):
        self.dna = _FakeDNA(barcodes, clones)


ARM = "chr1p"
VARS = [f"chr1:{(i + 1) * 100}:A/G" for i in range(12)]


def _detect_fixture(seed=0):
    """12 'A' cells (10 LOH + 2 het) and 10 'B' het cells over 12 chr1p variants.

    The LOH cells are LOW on the first 6 variants and HIGH on the last 6.
    Columns are shuffled to check position sorting.
    """
    rng = np.random.default_rng(seed)
    n_loh, n_a_het, n_b = 10, 2, 10
    barcodes = [f"BC{i:02d}" for i in range(n_loh + n_a_het + n_b)]
    clones = ["A"] * (n_loh + n_a_het) + ["B"] * n_b
    loh = np.hstack([rng.uniform(0, 5, (n_loh, 6)), rng.uniform(95, 100, (n_loh, 6))])
    het = rng.uniform(35, 65, (n_a_het + n_b, 12))
    df = pd.DataFrame(np.vstack([loh, het]), index=barcodes, columns=VARS)
    shuffled = list(rng.permutation(VARS))
    df = df[shuffled]
    sample = _FakeSample(barcodes, clones)
    return sample, df, {ARM: shuffled + ["chr1:99999:missing"]}, barcodes


def _detect(sample, df, grouped, **kw):
    kw.setdefault("verbose", False)
    return detect_loh_and_update_clones(sample, df, grouped, ARM, **kw)


# ─────────────────────────────────────────────────────────────────────
# detect_loh_and_update_clones
# ─────────────────────────────────────────────────────────────────────
def test_detect_finds_single_loh_clone():
    sample, df, grouped, bcs = _detect_fixture()
    seg, loh, corr, hom = _detect(sample, df, grouped)

    clones = list(sample.dna.row_attrs["Clone"])
    assert clones[:10] == ["A LOH-C0 chr1p"] * 10
    assert clones[10:12] == ["A", "A"]
    assert clones[12:] == ["B"] * 10
    # backup was created from the original labels
    assert list(sample.dna.row_attrs["Clone_backup"]) == ["A"] * 12 + ["B"] * 10

    assert sorted(loh.index) == bcs[:10]
    assert set(loh["loh_cluster"]) == {0}
    assert not loh["assigned_by_correlation"].any()
    assert (loh["total_loh_n_variants"] == 12).all()

    assert len(seg) == 10
    assert (seg["n_variants"] == 12).all()
    assert (seg["start_variant"] == VARS[0]).all()  # position-sorted
    assert (seg["end_variant"] == VARS[-1]).all()
    assert (seg["length_bp"] == 1100).all()
    assert (seg["extreme_fraction"] == 1.0).all()
    assert (seg["cluster_shared_low"] == 6).all() and (seg["cluster_shared_high"] == 6).all()

    assert len(hom) == 1
    assert hom.loc[0, "final_clone"] == "A LOH-C0 chr1p"
    assert hom.loc[0, "status"] == "KEPT"
    assert hom.loc[0, "n_correlations"] == 45  # 10 choose 2
    assert "final_mean_correlation" in loh.columns

    # A-LOH vs A-base comparison recorded; A base (12) larger than LOH (10) -> no expansion
    assert list(corr["loh_clone"]) == ["A LOH-C0 chr1p"]
    assert list(corr["non_loh_clone"]) == ["A"]


def test_detect_is_idempotent_on_labels():
    sample, df, grouped, _ = _detect_fixture()
    _detect(sample, df, grouped)
    first = list(sample.dna.row_attrs["Clone"])
    _detect(sample, df, grouped)
    assert list(sample.dna.row_attrs["Clone"]) == first


def test_detect_restore_from_backup():
    sample, df, grouped, _ = _detect_fixture()
    sample.dna.row_attrs["Clone_backup"] = sample.dna.row_attrs["Clone"].copy()
    sample.dna.row_attrs["Clone"] = np.array(["junk"] * len(df), dtype=object)
    _detect(sample, df, grouped, restore_from_backup=True)
    clones = list(sample.dna.row_attrs["Clone"])
    assert clones[0] == "A LOH-C0 chr1p" and clones[-1] == "B"


def test_detect_no_variants_returns_empty_and_creates_backup():
    sample, df, grouped, _ = _detect_fixture()
    out = detect_loh_and_update_clones(sample, df, {}, "chr9q", verbose=False)
    assert len(out) == 4 and all(isinstance(x, pd.DataFrame) and x.empty for x in out)
    assert "Clone_backup" in sample.dna.row_attrs
    assert list(sample.dna.row_attrs["Clone"]) == list(sample.dna.row_attrs["Clone_backup"])


def test_detect_no_extreme_values_returns_empty():
    sample, df, grouped, _ = _detect_fixture()
    df_mid = pd.DataFrame(50.0, index=df.index, columns=df.columns)
    out = _detect(sample, df_mid, grouped)
    assert all(x.empty for x in out)
    assert "LOH" not in " ".join(map(str, sample.dna.row_attrs["Clone"]))


def test_detect_cluster_too_small_returns_empty():
    sample, df, grouped, _ = _detect_fixture()
    out = _detect(sample, df, grouped, min_cluster_size=11)
    assert all(x.empty for x in out)


def test_detect_final_size_filter_returns_empty():
    sample, df, grouped, _ = _detect_fixture()
    out = _detect(sample, df, grouped, min_final_clone_size=11)
    assert all(x.empty for x in out)
    assert "LOH" not in " ".join(map(str, sample.dna.row_attrs["Clone"]))


def test_detect_skip_homogeneity_returns_empty_hom():
    sample, df, grouped, _ = _detect_fixture()
    seg, loh, corr, hom = _detect(sample, df, grouped, skip_homogeneity_check=True)
    assert hom.empty and len(loh) == 10


def test_detect_homogeneity_failure_strips_labels():
    sample, df, grouped, _ = _detect_fixture()
    seg, loh, corr, hom = _detect(sample, df, grouped, min_homogeneity_fraction=1.01)
    assert seg.empty and loh.empty
    assert list(hom["status"]) == ["REMOVED_CORRELATION"]
    assert list(sample.dna.row_attrs["Clone"]) == ["A"] * 12 + ["B"] * 10


def test_detect_preserves_other_arm_annotation_on_removal():
    sample, df, grouped, _ = _detect_fixture()
    clones = sample.dna.row_attrs["Clone"].copy()
    sample.dna.row_attrs["Clone_backup"] = clones.copy()
    clones[:10] = "A LOH-C5 chr2q"
    sample.dna.row_attrs["Clone"] = clones
    _detect(sample, df, grouped, min_homogeneity_fraction=1.01)
    assert list(sample.dna.row_attrs["Clone"][:10]) == ["A LOH-C5 chr2q"] * 10


def test_detect_reference_labels_dict_does_not_touch_backup():
    sample, df, grouped, bcs = _detect_fixture()
    ref = {bc: "REF" for bc in bcs}
    seg, loh, corr, hom = _detect(sample, df, grouped, clone_reference_labels=ref)
    assert "Clone_backup" not in sample.dna.row_attrs
    assert len(loh) == 10
    # annotation appended to the *current* label
    assert sample.dna.row_attrs["Clone"][0] == "A LOH-C0 chr1p"


def test_detect_reference_labels_series_by_barcode():
    sample, df, grouped, bcs = _detect_fixture()
    ref = pd.Series({bc: "REF" for bc in bcs[:5]})  # partial -> mapped by barcode, rest ''
    _, loh, _, _ = _detect(sample, df, grouped, clone_reference_labels=ref, skip_homogeneity_check=True)
    assert len(loh) > 0


def test_detect_reference_labels_wrong_length_raises():
    sample, df, grouped, _ = _detect_fixture()
    with pytest.raises(ValueError, match="clone_reference_labels"):
        _detect(sample, df, grouped, clone_reference_labels=["A", "B"])


# ─────────────────────────────────────────────────────────────────────
# Phase 0
# ─────────────────────────────────────────────────────────────────────
def _phase0_inputs():
    cols = [f"chr1:{i}:A/G" for i in range(1, 6)] + [f"chr2:{i}:A/G" for i in range(1, 4)] + ["chr3:1:A/G"]
    df = pd.DataFrame(np.full((3, len(cols)), 50.0), index=["a", "b", "c"], columns=cols)
    grouped = {
        "chr2q": cols[5:8],  # 3 variants
        "chr1p": cols[:5],  # 5 variants
        "chr3p": cols[8:9] + ["chr3:2:A/G", "chr3:3:A/G", "chr3:4:A/G"],  # only 1 in df
        "chr4p": [],
        "chr1q": cols[:4],  # exactly 4 variants
    }
    sample = _FakeSample(["a", "b", "c"], ["A", "A", "B"])
    return sample, df, grouped


def test_precheck_variant_availability_thresholds():
    _, df, grouped = _phase0_inputs()
    cfg = LOHConfig(verbose=False)
    avail, skipped = precheck_variant_availability(
        grouped, df, ["chr1p", "chr1q", "chr2q", "chr3p", "chr4p", "chr9p"], cfg
    )
    assert avail == ["chr1p", "chr1q"]
    assert skipped == ["chr2q", "chr3p", "chr4p", "chr9p"]


def test_precheck_respects_custom_minimum():
    _, df, grouped = _phase0_inputs()
    cfg = LOHConfig(verbose=False, min_contiguous_variants_initial=1)
    avail, skipped = precheck_variant_availability(grouped, df, ["chr2q", "chr3p", "chr4p"], cfg)
    assert avail == ["chr2q", "chr3p"] and skipped == ["chr4p"]


def test_precheck_empty_arms():
    _, df, grouped = _phase0_inputs()
    assert precheck_variant_availability(grouped, df, [], LOHConfig(verbose=False)) == ([], [])


def test_run_phase0_builds_state_with_sorted_available_arms():
    sample, df, grouped = _phase0_inputs()
    st = run_phase0(sample, df, grouped, None, LOHConfig(verbose=False))
    assert isinstance(st, PipelineState)
    assert st.chr_arms == ["chr1p", "chr1q"]
    assert len(st.variant_cols) == 9
    assert dict(st.clone_to_cells) == {"A": ["a", "b"], "B": ["c"]}


def test_run_phase0_none_when_no_available_arms():
    sample, df, grouped = _phase0_inputs()
    assert run_phase0(sample, df, grouped, ["chr2q", "chr4p"], LOHConfig(verbose=False)) is None


def test_run_phase0_none_when_no_variant_like_columns():
    df = pd.DataFrame(np.ones((2, 4)), index=["a", "b"], columns=["v1", "v2", "v3", "v4"])
    sample = _FakeSample(["a", "b"], ["A", "A"])
    st = run_phase0(sample, df, {"chr1p": ["v1", "v2", "v3", "v4"]}, None, LOHConfig(verbose=False))
    assert st is None


def _entry(cells, label, seg_rows=1):
    seg = pd.DataFrame({"cell": cells[:seg_rows]})
    loh = pd.DataFrame({"label": [label] * len(cells)}, index=cells)
    return (seg, loh, pd.DataFrame(), pd.DataFrame({"final_clone": [label]}))


# ─────────────────────────────────────────────────────────────────────
# run_phase2
# ─────────────────────────────────────────────────────────────────────
def _phase2_state(labels, results):
    bcs = [f"c{i}" for i in range(len(labels))]
    df = pd.DataFrame(np.full((len(labels), 1), 50.0), index=bcs, columns=["chr1:1:A/G"])
    st = PipelineState(
        Sample=_FakeSample(bcs, labels), df=df, grouped_by_chr_arm={}, chr_arms=[],
        config=LOHConfig(verbose=False),
    )
    st.results = results
    return st
