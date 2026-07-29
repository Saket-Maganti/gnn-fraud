# CoReGraph saved-output pilot V2 specification

Status: `SPECIFICATION_AND_DETERMINISTIC_IMPLEMENTATION_ONLY`

The pilot has not been executed in this repair pass. No prediction manifests
were connected, no official baseline was installed, and no target result was
measured. This document defines the second-review surface; it is not an
execution-readiness claim.

## Unit of provenance and pairing

Every prediction manifest must declare:

- dataset, task, and prediction unit;
- contract coordinate hash and environment ID;
- expert ID and optional alias provenance;
- seed and fold;
- config hash, code hash, prediction checksum, and deployment contract;
- contract role, availability state and reason codes, compute cost, and score
  type.

The loader verifies the checksum, the V3 contract coordinate hash, environment
ID, role, and required fields. Probability-pilot rows must be finite and lie in
`[0,1]`. Artifacts group by
`dataset × task × coordinate × environment × seed × fold`, with exactly one
canonical artifact per expert. Environment is retained because two deployment
environments may share coordinates without being the same evaluation unit.
Missing seeds, duplicate expert-seed artifacts, aliases counted as independent
experts, and ID/label/split disagreements fail closed.

## Information boundary

Source artifacts are divided by their declared `train` and `validation` rows.
All learned methods, mixture weights, early-stopping decisions, and the
abstention threshold are fit using source contracts only. Each target
prediction is produced after those choices are frozen. Target labels enter
only in the final offline scoring function and the explicitly named offline
feasible-oracle ceiling.

The entry point remains plan-only unless `--execute` is supplied. Supplying it
is outside this repair pass and requires a separate execution review.

## CoReGraph training

Source contracts are explicit groups. The implementation balances group
contributions and optimises the end-to-end composite objective from example
predictions, labels, group indices, expert predictions, feasibility masks, and
costs. The objective includes prediction risk, feasible-oracle regret, CVaR,
review-budget loss, compute, perturbation stability, calibration, and
abstention. Availability masks and costs come from the saved artifact
manifests; they are not replaced with all-one masks.

The abstention operating threshold is selected on source validation under the
declared capacity and cost. No-feasible-expert rows force abstention. Source
validation also controls early stopping.

## Predeclared comparators

The pilot emits distinct implementations for:

1. every single expert;
2. the average of all feasible experts;
3. the source-validation best expert;
4. a source-validation convex mixture over all experts;
5. the actual GraphSafe V2 compatibility implementation;
6. the existing graph-feature gate implementation;
7. a learned no-contract router;
8. a learned atomic-contract router, with unseen target IDs mapped to the
   unknown token;
9. `MOWST_INSPIRED_REIMPLEMENTATION`, never represented as official Mowst;
10. the offline feasible-oracle ceiling.

GraphSafe is not aliased to feature-only prediction. The learned router
baselines are not aliases for a fixed mixture or source-validation best expert.
All learned baselines use the same complete source-contract set.

## Required ablations

The result surface includes the full router and:

- no contract;
- atomic contract;
- no regret;
- no budget;
- no resource mask;
- no stability;
- no abstention;
- no diagnostics.

Each target result is long-form, with one row per dataset, target contract,
seed, fold, method, and metric.

## Outcomes and inference

Primary outcomes are AUPRC, Recall@0.5%, Recall@1%, Recall@2%,
budget-curve area, mean regret, maximum regret, CVaR regret, selective risk,
and compute. Coverage and AURC are also emitted for selective-prediction
diagnostics.

Exact dataset/task/target-contract/fold outcomes are paired first and then
averaged within seed. The seed—not a contract row—is the inferential block.
The analysis applies exact Wilcoxon, paired permutation, and seed-block
bootstrap intervals. Holm correction is applied within the frozen
`ranking_and_budget`, `robust_risk`, and `deployment` families against every
predeclared strong baseline. Worst-contract gain uses paired target-contract
outcomes; regret summaries use the paired contract-regret surface.

## Go/no-go interpretation

The V2 gate remains blocked until a separately authorized pilot supplies every
required expert, seed, baseline, ablation, outcome, and target pairing. A
passing deterministic implementation gate does not imply empirical success,
official-baseline parity, provider-data readiness, or execution readiness.
