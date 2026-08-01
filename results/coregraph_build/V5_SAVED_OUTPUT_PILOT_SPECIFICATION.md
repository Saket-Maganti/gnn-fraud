# V5 saved-output pilot specification

Status: `FROZEN_BEFORE_ANY_V5_PILOT_RESULT`

This is the explicit preregistration amendment required because the prior V5 file was a readiness-only manifest and did not define an executable policy, review budget, bounded source assembly, output protocol, or authorization mechanism. No real V5 method was fit, no real target score was produced, and no target metric or oracle was inspected before this amendment.

## Immutable primary surface

- Datasets, in order: `elliptic`, `dgraphfin`.
- Target protocols, in order: `strict_inductive`, `isolated_inductive`, `transductive_structure`.
- Provider prediction seeds: integers 1 through 10, stratified by dataset.
- Experts, in order: `feature_mlp`, `gcn`, `graphsage`.
- Methods, in order: `coregraph`, `uniform_average`, `best_fixed_expert`, `source_logistic_gate`.
- Primary cells: 60 dataset × held-out-protocol × provider-seed scenarios.
- Primary coordinates: exactly 240 scenario-method pairs.
- Review fraction: 0.01.

## Artifact and role semantics

The 180 immutable prediction members are role-neutral. A scenario assigns six source bindings (two protocols × three experts) and three target bindings (one held-out protocol × three experts). Reuse across scenarios is valid; dual source/target use inside one scenario is invalid. Identity includes dataset, provider seed, protocol, normalized split, and node ID. Provider `val` and `validation` both normalize explicitly to `validation`; unknown split tokens fail closed.

Source assembly admits label-known `train` and `validation` rows only. Target-unlabelled assembly admits label-known `test` score rows but has no label attribute or label-returning API. Provider-unknown rows are excluded. Target labels are opened by a single-use offline evaluator only after the policy, preprocessing, threshold, row-key, and target-score hashes are written to `POLICY_FREEZE_MANIFEST.json`.

## Bounded deterministic source assembly

Each source protocol is a separate environment. From each source split/environment, retain the 4,096 rows with the smallest SHA-256 rank of the composite row identity. This is a deterministic, order-independent, bounded sample fixed before real execution. Source train rows fit learned state; source validation rows select hyperparameters, early stopping, the best fixed expert, and abstention thresholds. Target labels never participate.

## Methods

`coregraph` uses the committed factorised contract encoder and resource-aware CoReRouter with score mean, score standard deviation, score range, confidence, expert disagreement, and normalized relative-cost diagnostics. It uses the committed balanced composite objective weights: average 1.0, ranking 0.1, robust regret 1.0, budget 0.1, stability 0.1, compute 0.1, calibration 0.1, and abstention 0.2; CVaR alpha 0.8; Adam learning rate 0.003; at most 100 steps; deterministic source-validation early stopping; axis dropout 0.05; contract noise 0.0; source-only abstention thresholding; resource masks; and all-unavailable forced abstention.

`uniform_average` assigns equal weight to each available expert, zero weight to unavailable experts, and forced abstention when none are available.

`best_fixed_expert` minimizes mean environment-balanced Brier risk on source validation. Ties use the frozen expert order. Resource infeasibility is respected.

`source_logistic_gate` predicts the source per-row lowest-error feasible expert from the three expert probabilities and their row-wise standard deviation. Standardization is fit on source train only. Logistic `C` is selected from `[0.1, 1.0, 10.0]` on mean source-validation Brier risk, ties choosing the smaller `C`; maximum iterations 1,000; deterministic random state derived separately from the provider seed. Target protocol identity is not a feature.

Relative expert costs are preregistered proxies, not measured runtime: feature MLP 1.0, GCN 3.0, GraphSAGE 4.0. The canonical primary resource profile makes all three experts available. Abstention capacity is 0.10 and abstention cost is 0.20. These proxy values must not be reported as measured resources.

## Offline evaluation and gate

Offline evaluation computes AUPRC, recall at the frozen 1% review fraction, selective risk, coverage, Brier risk, contract regret relative to the best whole-contract feasible expert, and relative-compute summaries. The best feasible expert and any instance oracle are non-deployable offline diagnostics and cannot influence fitting, hyperparameters, thresholds, early stopping, or method selection.

Methods pair only within exact dataset × held-out target protocol × provider seed cells. Missing, failed, stale, duplicated, resource-infeasible, mixed-preregistration, or malformed cells remain nonnumeric and force `INCONCLUSIVE`. The gate compares CoReGraph with each of the other three methods. The frozen minimum worst-cell contract-regret improvement is 0.001 and the allowed mean AUPRC harm floor is -0.002. A complete family that passes all three comparisons returns `GO`; a complete family that fails any frozen effect condition returns `NO_GO`.

## Determinism, output, and authorization

Provider prediction seed, router/model seed, data-order seed, bootstrap seed, and notebook shard are distinct manifest fields. Strict deterministic mode is required. Outputs use float32 scores, compressed NPZ, atomic JSON/checkpoint writes, SHA-256 manifests, and identity-gated resume. A complete coordinate is reusable only when code, config, preregistration, evidence, scenario, dependency-lock, method, output-schema, and file hashes match.

Real execution requires a clean Git tree plus the exact token `AUTHORIZE_COREGRAPH_V5_PILOT_RUN`. Plan, validation, dry-run, and synthetic execution never satisfy or bypass that guard. The real pilot is not executed during executor closure.
