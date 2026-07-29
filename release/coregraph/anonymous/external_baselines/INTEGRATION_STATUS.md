# Baseline integration status

No upstream repository is vendored and no provider dataset has been downloaded.
`coregraph.experts.official_adapters.OfficialProcessAdapter` validates checkout
commit, entrypoint, and result boundary before any smoke command.

## Ready in-tree

- Group DRO objective
- VREx objective
- optional IRMv1 penalty
- GraphSafe saved-output baseline
- simple average
- source-validation best expert
- feasible oracle evaluation ceiling

These are labelled `VALIDATED_REIMPLEMENTATION` or diagnostic as appropriate.

## Prepared, not parity-complete

- Mowst (MIT, archived official repository)
- CIGA (MIT, graph-level scope only)
- faithful event-stream TGN (Apache-2.0)
- GOOD external benchmark bridge (GPL-3.0 process boundary)

Their pins, acquisition commands, task scopes, and checkout variables are in
`BASELINE_REGISTRY.yaml`. They remain `PENDING_INTEGRATION` until the pinned
checkout is installed, adapted to the common prediction schema, and its tiny
upstream smoke test passes. A local adapter smoke test is not claimed as model
parity.

## Genuine blockers

GraphMETRO is a strong direct competitor, but its official repository has no
licence declaration. EERM is a suitable node-level graph-OOD method with an
Elliptic surface, but its official repository likewise has no licence. Both are
`UNAVAILABLE_LICENSE`. They are not replaced by loose imitations and cannot be
counted toward the baseline-readiness gate.
