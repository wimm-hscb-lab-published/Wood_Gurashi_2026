"""CNV_Utils.informativeness.

Functions whose names start with ``test_`` are accessed via the module
object so pytest does not collect them.
"""
import numpy as np
import pandas as pd
import pytest

from CNV_Utils import informativeness as inf


# --------------------------------------------------------------------------
# test_cnv_informative_variants (ploidy + depth ratio)
# --------------------------------------------------------------------------
V_DEL = "chr5:100:A/T"
V_AMP = "chr7:100:C/G"
V_UNMAPPED = "chr9:100:G/A"


def _cnv_inputs(seed=0, n=10):
    rng = np.random.default_rng(seed)
    labels = ["WT"] * n + ["DEL"] * n + ["AMP"] * n + ["SAME"] * n
    bcs = [f"bc{i}" for i in range(len(labels))]
    ploidy_del = np.r_[2 + rng.normal(0, 0.05, n), 1 + rng.normal(0, 0.05, n),
                       2 + rng.normal(0, 0.05, 2 * n)]
    ploidy_amp = np.r_[2 + rng.normal(0, 0.05, n), 2 + rng.normal(0, 0.05, n),
                       3 + rng.normal(0, 0.05, n), 2 + rng.normal(0, 0.05, n)]
    df_cnv = pd.DataFrame(
        {"label": labels, V_DEL: ploidy_del, V_AMP: ploidy_amp, V_UNMAPPED: 2.0}, index=bcs
    )
    depth_del = np.r_[[100] * n, [40] * n, [100] * 2 * n]
    depth_amp = np.r_[[100] * 2 * n, [180] * n, [100] * n]
    alt = pd.DataFrame({V_DEL: depth_del // 2, V_AMP: depth_amp // 2}, index=bcs)
    ref = pd.DataFrame({V_DEL: depth_del - depth_del // 2, V_AMP: depth_amp - depth_amp // 2}, index=bcs)
    v2a = {V_DEL: "AMP5", V_AMP: "AMP7"}
    grouped = {"chr5q": [V_DEL], "chr7q": [V_AMP], "chr9q": [V_UNMAPPED]}
    return df_cnv, alt, ref, v2a, grouped


# --------------------------------------------------------------------------
# test_variant_informativeness (AF/LOH)
# --------------------------------------------------------------------------
VARS = ["chr5:100:A/T", "chr5:200:C/G", "chr5:300:G/A"]


def _af_inputs(seed=0, n=30, wt_name="WT"):
    rng = np.random.default_rng(seed)
    labels = [wt_name] * n + ["LOH"] * n + ["Other"] * n
    bcs = [f"c{i}" for i in range(len(labels))]
    alt, ref = {}, {}
    for v in VARS:
        depth = rng.integers(40, 60, size=len(bcs))
        a = rng.binomial(depth, 0.5)
        a[n:2 * n] = 0  # LOH clone has lost the ALT allele
        alt[v], ref[v] = a, depth - a
    ALT = pd.DataFrame(alt, index=bcs)
    REF = pd.DataFrame(ref, index=bcs)
    AF = 100 * ALT / (ALT + REF)
    AF["label"] = labels
    v2a = {v: f"AMP{i}" for i, v in enumerate(VARS)}
    return AF, ALT, REF, v2a


def test_variant_informativeness_detects_loh_clone(tmp_path):
    AF, ALT, REF, v2a = _af_inputs()
    out = inf.test_variant_informativeness(
        AF, ALT, REF, {"chr5q": VARS}, str(tmp_path), variant_to_amplicon=v2a, verbose=False
    )
    res = out["chr5q"]
    assert set(res["label"]) == {"LOH", "Other"}
    loh = res[res["label"] == "LOH"]
    other = res[res["label"] == "Other"]
    assert loh["informative"].all()
    assert (loh["event_type"] == "cnLOH").all()
    assert not other["informative"].any()
    assert (other["informativeness_tier"] == "not_informative").all()
    assert (tmp_path / "wilcoxon_results_chr5q_enhanced.csv").exists()


def test_variant_informativeness_custom_wt_labels(tmp_path):
    AF, ALT, REF, v2a = _af_inputs(wt_name="Normal")
    out = inf.test_variant_informativeness(
        AF, ALT, REF, {"chr5q": VARS}, str(tmp_path),
        variant_to_amplicon=v2a, wt_labels=["Normal"], verbose=False,
    )
    assert set(out["chr5q"]["label"]) == {"LOH", "Other"}


def test_variant_informativeness_requires_label(tmp_path):
    AF, ALT, REF, _ = _af_inputs()
    with pytest.raises(KeyError, match="label"):
        inf.test_variant_informativeness(AF.drop(columns="label"), ALT, REF, {}, str(tmp_path), verbose=False)


def test_variant_informativeness_requires_wt_cells(tmp_path):
    AF, ALT, REF, _ = _af_inputs()
    with pytest.raises(ValueError, match="No WT/reference cells"):
        inf.test_variant_informativeness(AF, ALT, REF, {}, str(tmp_path), wt_labels="Missing", verbose=False)


def test_variant_informativeness_rejects_bad_dropout_rescue_cap(tmp_path):
    AF, ALT, REF, _ = _af_inputs()
    with pytest.raises(ValueError, match="dropout_rescue_max_clone_cells"):
        inf.test_variant_informativeness(
            AF, ALT, REF, {}, str(tmp_path), dropout_rescue_max_clone_cells=0, verbose=False
        )


# --------------------------------------------------------------------------
# expand_informative_variants: only the up-front argument validation is
# exercised; the rest needs a real H5 reload through missionbio.mosaic.
# --------------------------------------------------------------------------
@pytest.mark.parametrize("wt_label", ["", "   ", [None], [], [float("nan"), " "]])
def test_expand_informative_variants_rejects_empty_wt_label(tmp_path, wt_label):
    with pytest.raises(ValueError, match="wt_label"):
        inf.expand_informative_variants(
            H5_PATH=str(tmp_path / "missing.h5"),
            Sample=None,
            cytobands=pd.DataFrame(),
            output_dir=str(tmp_path),
            desired_plotting_order=[],
            whitelist=[],
            germline_whitelist=[],
            sample_raw_size=0,
            num_cores=1,
            wt_label=wt_label,
            verbose=False,
        )
