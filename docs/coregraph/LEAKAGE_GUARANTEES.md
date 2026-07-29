# Leakage guarantees

Temporal training, validation, and target graph views are constructed
separately. Non-transductive training views reject future nodes and edges.
Validation views reject target-period nodes/edges unless the declared
visibility contract permits them.

The central audit checks mask disjointness, label availability, graph cutoffs,
target-label access, target identity access, and fitting split for scalers,
calibrators, thresholds, and routers. Identifier-like feature columns are
blocked unless allowlisted. Target labels are legal only under the explicit
few-label access regime.

The target-label array may exist for final offline evaluation, but it is not
passed to fitting or label-free diagnostics. Tests deliberately inject future
edges, mask overlap, target labels, and identifier columns and require a
failure.
