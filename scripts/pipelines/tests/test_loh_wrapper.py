"""Unit tests for LOH_Pipeline/loh_wrapper.py (build_config, _final_assignment_summary)."""
import dataclasses
import inspect
from types import SimpleNamespace

import numpy as np
import pytest

from LOH_Pipeline import loh_wrapper as w
from LOH_Pipeline.loh_config import LOHConfig


BUILD_PARAMS = inspect.signature(w.build_config).parameters
CFG_FIELDS = {f.name: f for f in dataclasses.fields(LOHConfig)}


# ── build_config ─────────────────────────────────────────────────────

def test_build_config_returns_lohconfig():
    assert isinstance(w.build_config(), LOHConfig)


def test_every_build_config_param_is_a_config_field():
    missing = [p for p in BUILD_PARAMS if p not in CFG_FIELDS]
    assert missing == []


@pytest.mark.parametrize("name", list(BUILD_PARAMS))
def test_build_config_forwards_each_argument(name):
    sentinel = object()
    cfg = w.build_config(**{name: sentinel})
    assert getattr(cfg, name) is sentinel


def test_build_config_defaults_match_signature():
    cfg = w.build_config()
    for name, param in BUILD_PARAMS.items():
        assert getattr(cfg, name) == param.default, name


def test_build_config_fields_not_exposed_keep_lohconfig_defaults():
    cfg = w.build_config()
    ref = LOHConfig()
    for name in CFG_FIELDS:
        if name not in BUILD_PARAMS:
            assert getattr(cfg, name) == getattr(ref, name), name


def test_build_config_default_divergences_from_lohconfig_are_known():
    """Pin the (intentional?) places where wrapper defaults differ from LOHConfig."""
    ref = LOHConfig()
    diverging = {
        name: (param.default, getattr(ref, name))
        for name, param in BUILD_PARAMS.items()
        if param.default != getattr(ref, name)
    }
    assert diverging == {
        "phase3_direction_ado_min_block_variants": (6, 5),
        "phase3_direction_ado_min_good_variants": (5, 4),
        "phase4_arm_expansion_min_cells": (8, 5),
        "phase5_qc_alert_ado_rate_per_amplicon": (0.10, 0.20),
    }


def test_build_config_rejects_unknown_kwarg():
    with pytest.raises(TypeError):
        w.build_config(not_a_parameter=1)


# ── _final_assignment_summary ────────────────────────────────────────

def _summary_state(barcodes, clones, cell_to_barcode=None):
    bc2idx = {str(b): i for i, b in enumerate(barcodes)}
    return SimpleNamespace(
        Sample=SimpleNamespace(dna=SimpleNamespace(row_attrs={"Clone": np.asarray(clones, dtype=object)})),
        cell_to_barcode=cell_to_barcode or {},
        sample_idx_for_barcode=lambda bc: bc2idx.get(str(bc)),
    )


def test_final_assignment_summary_categories_and_order(capsys):
    barcodes = [f"bc{i}" for i in range(8)]
    clones = [
        "A LOH-C0 chr5q", "A LOH-C0 chr5q", "SUB: A C1 chr5q", "FP: A C2 chr5q",
        "A", "ADO", "small", "nan",
    ]
    st = _summary_state(barcodes, clones)
    out = w._final_assignment_summary(st, barcodes + ["not_in_sample"])

    assert list(out.columns) == ["category", "final_label", "n_cells", "percent_initial"]
    assert out["category"].tolist() == [
        "LOH", "subclonal_LOH", "possible_LOH", "non_LOH", "ADO", "small", "unassigned", "unassigned",
    ]
    assert out["n_cells"].sum() == 9
    loh = out.iloc[0]
    assert loh["final_label"] == "A LOH-C0 chr5q"
    assert loh["n_cells"] == 2
    assert loh["percent_initial"] == pytest.approx(100 * 2 / 9)

    unassigned = out[out["category"] == "unassigned"]
    assert set(unassigned["final_label"]) == {"nan", "<unassigned>"}

    printed = capsys.readouterr().out
    assert "FINAL CELL ASSIGNMENT SUMMARY" in printed
    assert "Initial cells: 9" in printed
    assert "LOH=2" in printed


def test_final_assignment_summary_uses_cell_to_barcode():
    st = _summary_state(["bcX", "bcY"], ["B", "A"], cell_to_barcode={"c1": "bcY", 2: "bcX"})
    out = w._final_assignment_summary(st, ["c1", 2])
    assert dict(zip(out["final_label"], out["n_cells"])) == {"A": 1, "B": 1}
    # Ties in n_cells are ordered by label
    assert out["final_label"].tolist() == ["A", "B"]


def test_final_assignment_summary_sorts_by_count_within_category():
    st = _summary_state(["b0", "b1", "b2"], ["A", "B", "B"])
    out = w._final_assignment_summary(st, ["b0", "b1", "b2"])
    assert out["final_label"].tolist() == ["B", "A"]
    assert out["n_cells"].tolist() == [2, 1]


def test_final_assignment_summary_empty(capsys):
    st = _summary_state([], [])
    out = w._final_assignment_summary(st, [])
    assert out.empty
    assert "Initial cells: 0" in capsys.readouterr().out
