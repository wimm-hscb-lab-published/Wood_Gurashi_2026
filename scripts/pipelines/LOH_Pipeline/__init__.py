"""LOH_Pipeline – Loss-of-heterozygosity detection and clone refinement.

Usage
-----
    from LOH_Pipeline import (
        build_config, run_phase0, run_phase1,
        run_phase3, run_phase4, run_phase5,
        LOHConfig, PipelineState,
        detect_loh_and_update_clones,
        plot_loh_clones,
    )

    cfg     = build_config(...)
    # Defaults are already strict/profile-only. The wrapper run(...) skips Phase 2,
    # skips Phase 4c broad rescue/arm expansion, and Phase 5 does not run ADO probability QC.
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
from .phase2 import run_phase2
from .phase3 import run_phase3
from .phase4 import run_phase4
from .phase5 import run_phase5
from .phase6_subclone_scan import run_post_phase5_subclone_scan
from .ado_probability import (
    run_ado_probability_qc,
    score_ado_combination,
    score_loh_clone_ado_probability,
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
    "run_phase2",
    "run_phase3",
    "run_phase4",
    "run_phase5",
    "run_post_phase5_subclone_scan",
    "run_ado_probability_qc",
    "score_ado_combination",
    "score_loh_clone_ado_probability",
    "detect_loh_and_update_clones",
    "plot_loh_clones",
]