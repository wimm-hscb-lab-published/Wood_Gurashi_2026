"""Cheap smoke test for one CNV_Utils.visualization heatmap.

The other visualisation functions (plot_integrated_cnv_baf_with_significance,
ploidy/binary variants, subclone plots) need large, tightly coupled inputs and
are intentionally not covered here.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from CNV_Utils import visualization as vis


@pytest.fixture(autouse=True)
def _close_figs():
    yield
    plt.close("all")


@pytest.mark.parametrize("categorical", [False, True])
def test_visualise_variants_by_clone_writes_png_and_pdf(tmp_path, monkeypatch, categorical):
    monkeypatch.setattr(plt, "show", lambda *a, **k: None)
    rng = np.random.default_rng(0)
    variants = ["chr1:10:A/T", "chr1:150:C/G", "chr5:1000:G/A"]
    df = pd.DataFrame(rng.uniform(0, 100, size=(12, 3)), columns=variants)
    df["label"] = ["WT"] * 6 + ["C1"] * 6
    cyto = pd.DataFrame(
        {
            "chromosome": ["1", "1", "5"],
            "arm": ["p", "q", "p"],
            "start": [0, 101, 0],
            "end": [100, 200, 5000],
            "cytoband": ["p36", "q21", "p15"],
        }
    )
    vis.visualise_variants_by_clone(
        df, variants, cyto, str(tmp_path), output_prefix="smoke",
        plot_categorical=categorical, desired_clone_plotting_order=["WT", "C1"], verbose=False,
    )
    written = sorted(p.suffix for p in tmp_path.iterdir())
    assert ".png" in written and ".pdf" in written
