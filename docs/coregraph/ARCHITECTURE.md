# Architecture

`DeploymentContract` validates and hashes the six-axis environment. Dataset
adapters emit typed task batches and separate train, validation, and target
graph views. Experts expose task support, contract support, resource needs, and
official-code status through one API.

CoReGraph encodes each axis independently, optionally adds pairwise
interactions, appends label-free deployment diagnostics, masks infeasible
experts, and routes with a linear, MLP, or attention scorer. CoReRouter exports
weights, selection, abstention probability, entropy, expected compute, and a
per-example explanation record. If no expert is feasible, it emits no selected
expert and abstains.

Experiment manifests bind canonical configuration, source commit, dependency
lock, dataset manifest, and output schema. Atomic writes and content checksums
make interrupted and stale runs distinguishable from valid completed runs.
