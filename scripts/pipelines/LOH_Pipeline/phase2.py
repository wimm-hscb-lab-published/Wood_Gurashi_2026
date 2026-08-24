"""
Phase 2 – p/q Arm Merging
==========================
If LOH is detected on both the p and q arms of the same chromosome
(possibly in different clones), check whether those clones share the
same base genotype.  When they do, merge the two into a single clone
covering the whole chromosome.

Call after Phase 1:

    from phase2 import run_phase2
    results = run_phase2(state)

Requires ``state.results`` to be populated (Phase 1 output).
"""
from __future__ import annotations

import re
from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .loh_config import LOHConfig, PipelineState


# ── internal helpers ───────────────────────────────────────────────

def _loh_arms_by_chromosome(results: dict) -> dict[str, list[str]]:
    """
    Group result keys by chromosome number.

    Returns only chromosomes that have *both* p and q in results,
    e.g. ``{"3": ["chr3p", "chr3q"]}``.
    """
    from .loh_config import get_chr_number

    chr_map: dict[str, list[str]] = defaultdict(list)
    for arm in results:
        chr_num = get_chr_number(arm)
        if chr_num is not None:
            chr_map[chr_num].append(arm)

    return {c: sorted(arms) for c, arms in chr_map.items() if len(arms) >= 2}


def _merge_result_entries(
    results: dict, arm_p: str, arm_q: str, merged_key: str
) -> None:
    """
    Combine ``results[arm_p]`` and ``results[arm_q]`` into
    ``results[merged_key]``.  Removes the two original keys.

    Each value is ``(df_segments, df_loh, df_correlations, df_homogeneity)``.
    """
    seg_p, loh_p, corr_p, hom_p = results[arm_p]
    seg_q, loh_q, corr_q, hom_q = results[arm_q]

    def _safe_concat(a, b):
        parts = [x for x in (a, b) if x is not None and len(x) > 0]
        if not parts:
            return pd.DataFrame()
        return pd.concat(parts, ignore_index=True)

    def _concat_loh(a, b):
        parts = [x for x in (a, b) if x is not None and len(x) > 0]
        if not parts:
            return pd.DataFrame()
        merged = pd.concat(parts)
        return merged[~merged.index.duplicated(keep="first")]

    results[merged_key] = (
        _safe_concat(seg_p, seg_q),
        _concat_loh(loh_p, loh_q),
        _safe_concat(corr_p, corr_q),
        _safe_concat(hom_p, hom_q),
    )

    del results[arm_p]
    del results[arm_q]


# ── main entry point ───────────────────────────────────────────────

def run_phase2(
    state: "PipelineState",
    cfg: "LOHConfig | None" = None,
) -> dict:
    """
    Merge p+q arms whose LOH clones share the same base genotype
    into a single whole-chromosome clone.

    Parameters
    ----------
    state : PipelineState
        Must contain ``state.results`` from Phase 1.
    cfg : LOHConfig, optional
        If *None*, uses ``state.config``.

    Returns
    -------
    dict
        The (possibly updated) ``state.results``.
    """
    from .loh_config import (
        LOH_TAG,
        extract_chr_arms,
        get_base_clone,
        vprint,
    )

    if cfg is None:
        cfg = state.config

    results = state.results

    vprint("\n" + "=" * 60, cfg)
    vprint("PHASE 2 — p/q Arm Merging", cfg)
    vprint("=" * 60, cfg)

    paired = _loh_arms_by_chromosome(results)
    if not paired:
        vprint("  No chromosomes with both p and q arms — nothing to merge.", cfg)
        return results

    vprint(f"  Chromosomes with both arms: {sorted(paired.keys())}", cfg)

    merges_done = 0

    for chr_num, arms in sorted(
        paired.items(),
        key=lambda kv: int(kv[0]) if kv[0].isdigit() else 99,
    ):
        arm_p = next((a for a in arms if a.endswith("p")), None)
        arm_q = next((a for a in arms if a.endswith("q")), None)
        if arm_p is None or arm_q is None:
            continue

        vprint(f"\n  ── chr{chr_num}: {arm_p} + {arm_q}", cfg)

        # ── Collect LOH clones on each arm, grouped by base genotype ──
        clones_on_p: dict[str, list[str]] = defaultdict(list)
        clones_on_q: dict[str, list[str]] = defaultdict(list)

        for lbl in state.loh_clone_labels():
            lbl_arms = extract_chr_arms(lbl)
            base = get_base_clone(lbl)
            if arm_p in lbl_arms:
                clones_on_p[base].append(lbl)
            if arm_q in lbl_arms:
                clones_on_q[base].append(lbl)

        # Find base genotypes present on both arms
        shared_bases = set(clones_on_p.keys()) & set(clones_on_q.keys())
        if not shared_bases:
            vprint(f"     No shared base genotypes across p and q — skipping.", cfg)
            continue

        for base in sorted(shared_bases):
            p_labels = clones_on_p[base]
            q_labels = clones_on_q[base]

            # Check if any clone already carries both arms
            already_merged = [
                lbl for lbl in state.loh_clone_labels()
                if get_base_clone(lbl) == base
                and arm_p in extract_chr_arms(lbl)
                and arm_q in extract_chr_arms(lbl)
            ]
            if already_merged:
                vprint(f"     {base}: already has both arms in {already_merged} — skipping.", cfg)
                continue

            # Merge all pairs of p-clone × q-clone with same base
            for p_lbl in p_labels:
                for q_lbl in q_labels:
                    if p_lbl == q_lbl:
                        continue

                    merges_done += 1

                    # Extract LOH tag from p-clone label
                    tag_re = re.compile(r"LOH-C\d+")
                    p_tag_match = tag_re.search(p_lbl)
                    p_tag = p_tag_match.group(0) if p_tag_match else "LOH-C0"

                    # Extract arm tokens with cytoband suffixes if present
                    def _arm_token(lbl, arm, tag):
                        pat = re.compile(
                            rf"{re.escape(tag)} ({re.escape(arm)}(?::\S+)?)"
                        )
                        m = pat.search(lbl)
                        return m.group(1) if m else arm

                    q_tag_match = tag_re.search(q_lbl)
                    q_tag = q_tag_match.group(0) if q_tag_match else "LOH-C1"

                    p_arm_token = _arm_token(p_lbl, arm_p, p_tag)
                    q_arm_token = _arm_token(q_lbl, arm_q, q_tag)

                    # Build merged label: base + tag + p-arm+q-arm
                    merged_label = f"{base} {p_tag} {p_arm_token}+{q_arm_token}"

                    vprint(f"     {base}: merging {p_lbl} + {q_lbl} → {merged_label}", cfg)

                    # Update all cells from both clones
                    current = state.current_clones.copy()
                    n_modified = 0
                    for i in range(len(current)):
                        old = str(current.iloc[i])
                        if old == p_lbl or old == q_lbl:
                            current.iloc[i] = merged_label
                            n_modified += 1
                    if n_modified > 0:
                        state.set_clone_labels(current)
                    vprint(f"       → {merged_label}  ({n_modified} cells)", cfg)

                    # Update "label" column in df_loh — check both
                    # original arm keys and the merged key
                    merged_key = f"{arm_p}+{arm_q}"
                    for arm_key in (arm_p, arm_q, merged_key):
                        if arm_key not in results:
                            continue
                        _, df_loh, _, _ = results[arm_key]
                        if "label" in df_loh.columns:
                            df_loh["label"] = df_loh["label"].apply(
                                lambda lbl, _p=p_lbl, _q=q_lbl, _m=merged_label:
                                    _m if str(lbl) in (_p, _q) else str(lbl)
                            )

                    # Merge result entries (only if not already merged)
                    if arm_p in results and arm_q in results:
                        _merge_result_entries(results, arm_p, arm_q, merged_key)
                        vprint(f"       Results merged → '{merged_key}'", cfg)
                    elif merged_key in results:
                        vprint(f"       Results already merged under '{merged_key}'", cfg)

    # Refresh state mappings after all label changes
    state.refresh()

    vprint(f"\n  Phase 2 merges performed: {merges_done}", cfg)
    vprint(f"  Result keys: {sorted(results.keys())}", cfg)
    vprint(f"  LOH labels:  {state.loh_clone_labels()}", cfg)
    vprint(f"  LOH cells:   {state.loh_cell_count()}", cfg)
    vprint("=" * 60 + "\n", cfg)

    return results