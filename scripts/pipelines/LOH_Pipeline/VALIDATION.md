# Validation

Validated in this build:

- all package Python modules compile with `python -m py_compile`;
- fresh package import and `build_config()` succeed;
- Phase 4e legacy residual discovery defaults to disabled and is not called by `run_phase4`;
- Phase 4f performs one conservative pass only and contains no de novo small-clone creation path;
- synthetic 5-cell, 4-amplicon residual component is accepted as an active LOH subgroup only when ADO support is `ADO_UNLIKELY`;
- synthetic 4-cell residual component is left in the base clone;
- final small-label helper and Phase 5 pruning collapse labels to exactly `small`.
