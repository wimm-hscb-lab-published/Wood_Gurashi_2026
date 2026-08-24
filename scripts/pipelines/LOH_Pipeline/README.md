# LOH Pipeline — Conservative Residual Validation

This package contains the full multi-chromosome LOH pipeline.

## Final residual base-clone validation

The final residual scan runs once, immediately before Phase 5. It may create a
new active LOH subgroup only when all of the following hold:

- at least `min_final_clone_size` cells (default 5);
- at least 4 contiguous high-extreme informative variants per represented arm;
- at least 3 independent informative amplicons across the candidate block;
- independent per-arm component homogeneity;
- a shared, variant-wise LOW/HIGH directional signature per arm;
- ADO probability classified as `ADO_UNLIKELY` under the configured amplicon model.

Residual 2–4 cell patterns remain in their base population. The scan is not
iterative and never creates de novo `small` labels.

## Small labels

During final Phase 5 cleanup, every terminal label below the final size threshold
is collapsed to the single label `small`.

## Optional final-label subclone refinement inside `run(...)`

To inspect a residual final parent clone for further LOH children without a
second function call, enable the targeted Phase-6 refinement directly in the
original wrapper invocation:

```python
out = LOH_Pipeline.run(
    ... ,
    post_phase5_subclone_scan_labels=["TP53_R248W_Hem"],
    # Optional restrictions; defaults to all Phase-0 eligible arms.
    post_phase5_subclone_scan_chr_arms=None,
    post_phase5_subclone_scan_min_parent_cells=None,
)
```

The wrapper freezes the completed Phase-5 labels, scans only the named final
non-LOH parent clone(s), validates any new LOH children, refreshes Phase 5,
and then produces the ordinary final accounting and plots. It does not rescan
WT by default, active LOH/SUB/FP/small/ADO labels are ineligible, and Phase-4
broad merging is not re-run. Audit output is available as
`out["phase6_results"]["audit_df"]`.

## Phase 6 primary-family refinement (v7)

Phase 6 discovers a strict LOH core before any expansion. A residual cell can
join that primary family only when it is highly similar to the core AF
representative, retains substantial extreme support over the accepted block,
and agrees with the core's allelic direction. The core itself remains the LOH
call; minor dropout in an expanded member does not invalidate the established
core.

When a secondary arm is found in only a subset of the primary family, the
primary `Clone` label is intentionally preserved. The secondary arm is stored
in `Sample.dna.row_attrs['phase6_secondary_loh_arms']` rather than creating a
flat-label split that could prune the rest of the primary block. Primary-family
audit attributes are `phase6_primary_loh_child`, `phase6_primary_loh_arm`, and
`phase6_primary_loh_member_role`.
