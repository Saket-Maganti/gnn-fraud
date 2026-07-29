# ContractShift leakage threat model

Status: `PASS_PRE_RUN_SENTINELS`

Protected channels are labels, prediction IDs, target membership, future
nodes/edges, target covariates, scaler/calibrator/threshold state, router
training state, temporal priors, and resource availability. Adversaries are
accidental code paths, stale artifacts, row-order alignment, misleading
metadata, and target-derived hyperparameter choices.

Controls:

1. Separate train, validation, and target `GraphView`s with causal edge
   cutoffs and explicit transductive exceptions.
2. `nextafter(cutoff,+inf)` boundaries prevent floating-point overlap.
3. Disjoint supervised masks and unknown-label exclusion.
4. Central access records for scaler, threshold, calibrator, router, target
   labels, and target IDs.
5. Identifier-column detection and typed-ID prediction alignment.
6. Diagnostic registry declaring level, graph view, target access, and label
   requirement; label-dependent target diagnostics are rejected.
7. Source/target contract split disjointness and unseen atomic-ID guards.
8. Config/code/data/schema/result/prediction checksum validation on resume and
   import.
9. Target labels absent from the saved-output CoReRouter fit signature and used
   only for offline scoring.

Declared transductive structure is not automatically called leakage; it is a
different contract. Any result must state that visibility contract.
