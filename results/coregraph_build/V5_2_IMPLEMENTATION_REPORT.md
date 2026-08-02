# V5.2 Numerical Implementation Report

V5.2 repairs only the confirmed numerical realization defect. Target/source score assembly, method routing post-processing, routed-score calculation, expected compute, Brier/oracle/regret evaluation, stored NPZ arrays, manifests, resume identities, and package validation now bind float64 numerical semantics.

Every primary method passes through the same feasible-simplex and hull layer. The package validator independently rejects old schemas, old numerical implementation identities, float32 scientific arrays, unavailable nonzero weights, and regret below tolerance.

Unchanged: primary methods, experts, datasets, protocols, seeds, scenarios, source sampling, review fraction, matched expert-or-abstain oracle, abstention cost, metric names, and gate thresholds.

No V5.1 coordinate is reusable because V5.2 binds a new code SHA, preregistration hash, config hash, effective execution hash, implementation version, and output schema.
