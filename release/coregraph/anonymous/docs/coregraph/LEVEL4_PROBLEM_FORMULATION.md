# Level-4 problem formulation

Let \(\mathcal E\) be a finite expert set and \(\mathcal A(c)\subseteq\mathcal E\) the set feasible under contract \(c\). Source contracts \(\mathcal C_s\) expose labelled training and validation rows. A target contract \(c_\star\) may be a seen contract, an unseen combination, an unseen axis value, partially observed, noisy, subject to a dynamic resource change, or associated with universal expert failure/unavailability.

For instance \(x\), label-free diagnostics \(d_e(x,c)\), and expert score \(s_e(x,c)\), the policy emits \(\pi_\theta(e\mid x,c,d)\). Infeasible mass is zero before normalization. The hierarchical policy factorizes as a contract prior plus a bounded instance correction. Its routed score is the feasible weighted combination of expert scores; an empty feasible set triggers explicit abstention or a declared non-expert fallback.

## Loss, regret, and robustness

For loss \(\ell\), contract risk is

\[
R_c(\pi)=\mathbb E_{(X,Y)\sim P_c}[\ell(\pi(X,c),Y)].
\]

The feasible whole-contract oracle is \(R_c^\star=\min_{e\in\mathcal A(c)}R_c(e)\), and regret is \(R_c(\pi)-R_c^\star\). We report mean, maximum, and CVaR regret. The oracle selects one expert after risk aggregation across the contract; an instance-changing oracle is a diagnostic ceiling only.

The source objective combines task risk, contract regret, CVaR/worst-contract risk, review-budget penalty, routing stability, abstention, uncertainty, and sparsity. Each coefficient is frozen from source information only. Resource feasibility is a hard pre-selection constraint, not a soft target-evaluated penalty.

## Selection and budgets

The selection contract records domain generalisation with no target access, label-free target adaptation, or few-label adaptation. These regimes are never pooled. Review budgets may be a fraction, fixed `K`, latency allowance, or cost envelope. Recall at a review budget and selective risk/coverage are separate operational outcomes.

## Selective prediction

An abstention threshold is selected on balanced source validation and frozen before target scoring. Coverage is the accepted fraction. Selective risk is undefined at zero coverage and is reported `NOT_APPLICABLE_ZERO_COVERAGE`. All-unavailable rows abstain by construction.

## Composition gap and stability

The composition generalisation gap compares performance on unseen combinations with matched seen combinations under common expert and resource sets. Routing stability measures policy change under metadata noise, score perturbation, availability changes, and nearby budget thresholds. Counterfactual tests hold expert scores fixed while varying only contract/resource inputs.

## Access invariants

- Target labels are unavailable during router/baseline fitting and threshold selection.
- Provider labels, model scores, split, `label_known`, dataset, unit ID, fold, protocol, expert, and seed must align.
- Unknown labels are excluded from scoring.
- No artifact is both source and target inside one scenario.
- Same-numbered seeds across datasets are not paired observations.
