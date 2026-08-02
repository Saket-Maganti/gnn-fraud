# V5 saved-output pilot specification v5.1

Status: `FROZEN_BEFORE_ANY_REAL_V5_PILOT_RESULT`

This v5.1 preregistration supersedes v5.0 before real pilot execution. The primary regret comparator was corrected before real pilot execution because the previous method and comparator used unequal feasible action spaces. No real V5 method was fit, no real target score, label, metric, or oracle was inspected, and no empirical paper result was populated before this correction.

## Immutable primary surface

- Datasets, in order: `elliptic`, `dgraphfin`.
- Target protocols, in order: `strict_inductive`, `isolated_inductive`, `transductive_structure`.
- Provider prediction seeds: integers 1 through 10.
- Experts, in order: `feature_mlp`, `gcn`, `graphsage`.
- Methods, in order: `coregraph`, `uniform_average`, `best_fixed_expert`, `source_logistic_gate`.
- Primary cells: 60 dataset × held-out-protocol × provider-seed scenarios.
- Primary coordinates: exactly 240 scenario-method pairs.
- Frozen review fraction: 0.01.

## Artifact, role, and label-access semantics

The 180 immutable prediction members are role-neutral. Each scenario assigns six source bindings and three target bindings. Reuse across scenarios is valid; dual source/target use inside one scenario is invalid. Source assembly admits label-known `train` and `validation` rows only. Target-unlabelled assembly admits `test` scores but exposes no labels. A single-use offline evaluator may open target labels only after policy, preprocessing, threshold, target-row, target-score, base-config, effective-config, code, dependency, schema, and preregistration identities are frozen and hashed.

## Bounded deterministic source assembly

Each source protocol is a separate environment. From each source split/environment, retain the 4,096 rows with the smallest SHA-256 rank of the composite row identity. Source train rows fit learned state. Source validation rows select hyperparameters, early stopping, the best fixed expert, and abstention thresholds. Target labels never participate.

## Frozen methods and operational parameters

`coregraph` uses the committed factorised contract encoder and resource-aware CoReRouter. `uniform_average` weights available experts equally. `best_fixed_expert` uses source-validation environment-balanced Brier risk. `source_logistic_gate` is trained and selected only on source data. All methods respect resource masks and force abstention when no expert is available.

Relative expert costs remain preregistered proxies: feature MLP 1.0, GCN 3.0, and GraphSAGE 4.0. Abstention capacity is 0.10 and abstention cost is 0.20. The canonical first pilot uses one worker, float32 target scores, strict deterministic algorithms, verified ZIP-member streaming without permanent extraction, stable SHA-256 source sampling, and bounded label-blind target inference.

## Corrected primary regret and oracle scope

For target row \(i\), feasible expert \(e\), label \(y_i\), expert score \(p_{i,e}\), and frozen abstention cost \(c_{\mathrm{abs}}\):

```text
expert_loss(i,e) = (y_i - p_i,e)^2
feasible_row_oracle_loss(i) = min(c_abs, min over feasible e of expert_loss(i,e))
method_loss(i) = c_abs if the method abstains, otherwise (y_i - p_i)^2
contract_regret_vs_feasible_row_oracle = mean(method_loss - feasible_row_oracle_loss)
```

The primary oracle is the row-wise feasible hindsight oracle including abstention. It uses the same feasible expert-or-abstain action space as the deployed method. Unavailable experts never enter it; an all-experts-unavailable row uses abstention. Primary regret must be nonnegative under exact arithmetic. Values below `-1e-12` fail closed; values within tolerance are normalized to zero.

The secondary diagnostic is the best fixed feasible non-abstaining expert over the whole contract. It is reported only as `best_fixed_nonabstaining_expert_brier` and `excess_cost_vs_best_fixed_nonabstaining_expert`; it is not the primary regret comparator or gate statistic.

## Metric schema and gate

The frozen metric schema is `coregraph_v5_metric_schema_v2`. It reports `global_target_auprc` over all target rows, `recall_at_frozen_review_fraction` with the exact fraction recorded, `selective_risk`, `coverage`, `contract_brier_risk`, `feasible_row_oracle_loss_with_abstention`, `contract_regret_vs_feasible_row_oracle`, the separate best-fixed diagnostics, relative compute, and boolean `resource_feasible`. No selective AUPRC is introduced.

Methods pair only within exact dataset × target protocol × provider seed cells. Missing, failed, stale, duplicated, mixed-schema, mixed-preregistration, mixed-effective-config, or malformed cells force `INCONCLUSIVE`. The gate compares CoReGraph with each baseline using the corrected primary regret and global target AUPRC. The frozen minimum worst-cell regret improvement is 0.001 and the allowed mean global-AUPRC harm floor is -0.002.

## Effective execution identity and resume

Every coordinate binds a canonical `coregraph_v5_effective_execution_config_v1` hash covering the base config, preregistration, configured and effective chunk rows, one-worker policy, real/synthetic mode, dtype, deterministic algorithms, output and metric schemas, method registry, archive streaming, source sampling, target inference, dependency lock, and code SHA. A change to any bound field invalidates coordinate identity, checkpoint reuse, policy freeze, evaluation, completion, package eligibility, and gate aggregation.

## Exact package closure

Packaging compares the exact coordinate keys in `PILOT_PLAN.csv`, `RUN_MANIFEST.json`, method directories, checkpoints, evaluations, and `COMPLETE` identities. Counts alone are insufficient. Missing, extra, duplicated, stale, mixed, failed, or partial coordinates fail. Every coordinate's identities and file checksums are validated. The package is validated before ZIP creation, checked by ZIP CRC, extracted to a temporary directory, checksum-verified, and validated again before the temporary extraction is deleted.

## Authorization

Authoritative real execution has no dirty-tree bypass and requires a clean tree, compatible safe output root, sufficient disk, one worker, exact remote/local provenance checks by the operator, and token `AUTHORIZE_COREGRAPH_V5_PILOT_RUN`. Plan, validate-only, packaging, and synthetic execution cannot satisfy or bypass this guard. The real pilot remains unexecuted during this repair.
