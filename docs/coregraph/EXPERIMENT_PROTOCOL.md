# Experiment protocol

The screening campaign uses five seeds and the frozen master matrix. Final
confirmatory cells use ten seeds. Source contracts and target contracts are
disjoint. Required split families are leave-one-contract-out, observed-axis
unseen combinations, axis holdouts, and controlled synthetic mechanisms.

Primary reporting uses per-contract predictions, macro contract summaries,
matched-contract worst-case performance, regret against one best feasible
expert for the whole contract, CVaR formed within expert-prediction seed,
budget curves, calibration, compute, abstention, and routing stability. The
instance-clairvoyant oracle is diagnostic only. Dataset/task families and
target contracts are never treated as independent seed replications.

Hyperparameters, source-contract budgets and abstention capacities, metrics,
statistical families, multiplicity method, exclusions, expert-prediction seeds
and deterministic router-training seeds are frozen before final execution.
Target metadata may constrain the final label-free decision; target labels
cannot select a method or threshold. Failed and resource-blocked runs remain
in manifests, but sentinel predictions never enter ranking metrics. Fallback
outputs remain distinct from abstention. Heavy execution is only permitted
after all applicable local, data, baseline, licence, and hardware gates pass.
