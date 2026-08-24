"""
LOH Pipeline – Phase 0: Setup baseline clone labels & pre-check variant availability.

Returns a PipelineState ready for Phase 1, or None if nothing can be processed.
"""
from __future__ import annotations

import pandas as pd
from typing import Optional

from .loh_config import (
    LOHConfig,
    PipelineState,
    vprint,
    sort_chromosome_arms,
)


# ─────────────────────────────────────────────────────────────────────
# Phase 0a: Variant availability pre-check
# ─────────────────────────────────────────────────────────────────────
def precheck_variant_availability(
    grouped_by_chr_arm: dict,
    df: pd.DataFrame,
    chr_arms: list[str],
    cfg: LOHConfig,
) -> tuple[list[str], list[str]]:
    """
    Check which requested chromosome arms have enough variants for LOH analysis.

    An arm is skipped if:
      - It has zero variants in df, OR
      - It has fewer variants than ``cfg.min_contiguous_variants_initial``
        (the minimum contiguous run required to call an LOH segment).

    Returns
    -------
    available : list[str]
        Arms with sufficient variants.
    skipped : list[str]
        Arms that were dropped (with reason logged).
    """
    min_vars = cfg.min_contiguous_variants_initial

    vprint("\n" + "-" * 60, cfg)
    vprint("Pre-check: Variant availability per arm", cfg)
    vprint(f"What: Verify each arm has ≥ {min_vars} informative variants in df.", cfg)
    vprint("Why:  Arms with fewer variants than the minimum contiguous segment", cfg)
    vprint("      length cannot produce a valid LOH call.", cfg)
    vprint("-" * 60, cfg)

    available: list[str] = []
    skipped: list[str] = []

    for arm in chr_arms:
        variants = [v for v in grouped_by_chr_arm.get(arm, []) if v in df.columns]
        n = len(variants)

        if n == 0:
            vprint(f"  ✗ {arm}: no informative variants — skipped", cfg)
            skipped.append(arm)
        elif n < min_vars:
            vprint(
                f"  ✗ {arm}: {n} variant{'s' if n != 1 else ''} "
                f"< {min_vars} required — insufficient for LOH detection, skipped",
                cfg,
            )
            skipped.append(arm)
        else:
            vprint(f"  ✓ {arm}: {n} variants", cfg)
            available.append(arm)

    vprint("\nSummary:", cfg)
    vprint(f"  Will process: {len(available)} arms", cfg)
    vprint(f"  Will skip:    {len(skipped)} arms", cfg)

    return available, skipped


# ─────────────────────────────────────────────────────────────────────
# Phase 0: orchestrator
# ─────────────────────────────────────────────────────────────────────
def run_phase0(
    Sample,
    df: pd.DataFrame,
    grouped_by_chr_arm: dict,
    chr_arms: Optional[list[str]],
    cfg: LOHConfig,
    df_dp: pd.DataFrame | None = None,
) -> Optional[PipelineState]:
    """
    Full Phase 0:
      1. Resolve which chromosome arms to process.
      2. Pre-check variant availability.
      3. Build and return a ``PipelineState``.

    Returns ``None`` if there are no processable arms or no variant
    columns, signalling the caller to exit early.
    """
    vprint("\n" + "=" * 60, cfg)
    vprint("MULTI-CHROMOSOME LOH DETECTION", cfg)
    vprint("=" * 60, cfg)

    # 1. Resolve chromosome arms
    if chr_arms is None:
        chr_arms = list(grouped_by_chr_arm.keys())
    chr_arms = sort_chromosome_arms(chr_arms)

    vprint(f"\nChromosome arms requested: {len(chr_arms)}", cfg)
    vprint(f"  {', '.join(chr_arms)}", cfg)

    # 2. Variant availability
    available, _skipped = precheck_variant_availability(
        grouped_by_chr_arm, df, chr_arms, cfg
    )

    if not available:
        vprint("\n⚠️  No chromosome arms have available variants", cfg)
        return None

    # 3. Build shared state
    state = PipelineState(
        Sample=Sample,
        df=df,
        grouped_by_chr_arm=grouped_by_chr_arm,
        chr_arms=available,
        config=cfg,
        df_dp=df_dp,
    )

    if len(state.variant_cols) == 0:
        vprint("⚠️  No variant columns found in df!", cfg)
        return None

    return state


__all__ = ["precheck_variant_availability", "run_phase0"]