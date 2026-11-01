"""LOH_Pipeline: loss-of-heterozygosity (LOH) detection and clone refinement.

The pipeline runs in phases on a shared ``PipelineState``:
Phase 0 setup → Phase 1 per-arm LOH detection → (Phase 2 p/q merge: skipped) →
Phase 3 clone QC → Phase 4 rescue and merging → Phase 5 final LOH-block QC →
Phase 6 nested subclone scan.

Usage
-----
    from LOH_Pipeline import build_config, run

    cfg = build_config(...)
    results = run(...)

The individual phases (``run_phase0`` … ``run_phase5``, ``run_post_phase5_subclone_scan``),
``LOHConfig``, ``PipelineState``, ``detect_loh_and_update_clones`` and ``plot_loh_clones``
are also exported. See README.md for details.
"""

# ── Config & state ────────────────────────────────────────────────────────────
from .loh_config import (
    LOHConfig,
    PipelineState,
)

# ── Wrapper (entry point + build_config) ──────────────────────────────────────
from .loh_wrapper import build_config, run

# ── Pipeline phases ───────────────────────────────────────────────────────────
from .phase0_setup import run_phase0
from .phase1 import run_phase1
from .phase3 import run_phase3
from .phase4 import run_phase4
from .phase5 import run_phase5
from .phase6_subclone_scan import run_post_phase5_subclone_scan
from .ado_probability import (
    score_ado_combination,
)

# ── Detection function ────────────────────────────────────────────────────────
from .detect_loh import detect_loh_and_update_clones

# ── Plotting ──────────────────────────────────────────────────────────────────
from .plot_loh_clones import plot_loh_clones

__all__ = [
    "LOHConfig",
    "PipelineState",
    "build_config",
    "run",
    "run_phase0",
    "run_phase1",
    "run_phase3",
    "run_phase4",
    "run_phase5",
    "run_post_phase5_subclone_scan",
    "score_ado_combination",
    "detect_loh_and_update_clones",
    "plot_loh_clones",
]