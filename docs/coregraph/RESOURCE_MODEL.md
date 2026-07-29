# Resource model

Each expert declares minimum memory, expected latency, device class, graph and
edge-feature needs, full-graph limits, and cost provenance. Contract resource
axes can exclude experts before scoring. The mask is mathematical: excluded
experts receive exactly zero routing mass.

Large graph work uses deterministic neighbor sampling; temporal models use
chronological event mini-batches. Mixed precision and gradient accumulation
are run-config fields. Full-graph execution estimates memory and raises before
allocation when the cap is exceeded.

Kaggle notebooks target T4 x2 only as a scheduling envelope. They do not claim
multi-GPU training unless the model runner explicitly implements it. Runtime
fields in future matrices remain `TBD_PROFILE` until measured.
