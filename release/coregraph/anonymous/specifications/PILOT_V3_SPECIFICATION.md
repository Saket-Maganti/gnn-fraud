# CoReGraph saved-output pilot V3 specification

Status: `SPECIFICATION_AND_DETERMINISTIC_IMPLEMENTATION_ONLY`.

No provider prediction was connected and the pilot was not executed during
this closure pass. This specification governs the later manifest-conversion
review and dry-run completeness audit; it does not establish execution
readiness.

## Provenance and completeness

Every manifest declares dataset, task, prediction unit, deployment coordinate
and environment, expert, expert-prediction seed, fold, config/code/checksum,
contract role, availability reasons, compute cost and score type. The exact
pilot surface is:

- Elliptic and DGraphFin;
- strict-inductive, isolated-inductive and transductive target contracts;
- expert-prediction seeds 1--10;
- feature MLP, GCN and GraphSAGE saved outputs;
- every predeclared baseline, all eight ablations and every required metric.

Aliases, duplicates, missing seeds, missing target cells, unexpected cells,
misaligned IDs/labels/splits, checksum mismatches and non-deterministic router
seeds fail closed. The expert-prediction seed is the inferential block.
Router-training seed is derived deterministically from expert seed plus method
or ablation and is secondary provenance only.

## Information and decision boundary

Source `train` rows fit the router. Balanced source `validation` groups control
early stopping, Mowst-inspired confidence routing and abstention threshold
selection. Target contract metadata, including operational capacity, may be
known. Target labels may not influence fitting, thresholding, routing,
abstention or baseline selection.

`BaselinePrediction` stores the frozen abstention vector, probability,
threshold, threshold provenance, forced-abstention vector, capacity, cost and
execution status. Evaluation consumes that stored decision exactly; it never
recomputes a `0.5` cutoff. A target capacity can remove threshold-selected
abstentions after freezing but cannot alter source parameters or fill capacity
with new abstentions. Zero coverage yields undefined selective risk, retains
its abstention cost and blocks a go/no-go win.

## Contract-local operational constraints

Review-budget loss is computed separately inside each source contract from
that contract's fraction or fixed K, then balanced across constrained groups.
Unconstrained contracts receive no invented K. Mixing fractional and fixed-K
constrained groups is unsupported and fails closed.

Abstention-capacity penalties are also group-local. Threshold search balances
source-validation groups and satisfies each source contract independently.
The held-out target capacity does not enter source training or calibration.

## Availability and comparators

Each method-contract result declares one of `EXECUTABLE`,
`EXECUTABLE_WITH_FALLBACK`, `ABSTAIN_ONLY`, `RESOURCE_BLOCKED` or
`NOT_APPLICABLE`. Fallback means an allowed alternative generated an
executable output; abstention means no prediction was accepted. Blocked,
not-applicable and abstain-only sentinel scores never enter ranking metrics.

Predeclared comparators are individual experts, feasible averaging,
source-validation best, source-validation convex mixing, the existing
graph-feature gate, learned no-contract and atomic-contract routers,
`MOWST_INSPIRED_REIMPLEMENTATION`, and
`graphsafe_confidence_abstention_component`. The GraphSafe-named component is
only a validation-fitted confidence/abstention component and is not presented
as full GraphSafe parity. The Mowst-inspired routing threshold is selected
across balanced source contracts with availability-aware fallback and frozen
before target scoring.

## Oracle and regret semantics

Headline regret uses `contract_feasible_oracle`: each expert's loss is first
aggregated over the target contract, infeasible whole-contract experts are
masked, and one best feasible expert is selected for the entire contract.
`instance_clairvoyant_oracle_ceiling` may select per example but is a
diagnostic only. It is excluded from deployable methods, significance,
headline regret, gates and paper comparisons.

Mean, maximum and CVaR regret are derived from contract regret within each
expert-prediction seed. Worst-case improvement is formed by subtracting the
method and baseline inside identical target contracts, then taking the minimum
paired improvement within seed. Independent method and baseline minima are
prohibited.

## Frozen go/no-go rule

`PILOT_GATE_FROZEN_SPEC.json` requires complete two-dataset/ten-seed coverage;
positive matched worst-contract regret improvement over feasible average and
source-validation best; Holm-corrected support with a positive effect and
confidence interval on at least one robust primary outcome; positive effects
on both datasets; no material mean AUPRC harm; no one-contract-only effect;
meaningful corrected contributions from factorised contracts, robust regret,
resource masks and budget loss; routing diversity and stability; positive
full-method coverage; and no target-label selection.

All eight full-versus-ablation effects are computed from paired expert-seed
blocks. Their declared effect thresholds, bootstrap intervals and Holm
decisions affect the verdict. Merely providing ablation names never passes.

## Stop boundary

The entry point remains plan-only without `--execute`. Manifest conversion,
dry-run completeness checking and a third independent review are required
before any pilot authorization. This closure does not claim provider-data,
official-baseline, empirical or pilot execution readiness.
