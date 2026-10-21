# Regret decomposition

Status: `PROVED_INTERNAL_REVIEW_PENDING` as a triangle-inequality upper bound.

Insert successive policies that use: the true contract representation; the estimated representation; true diagnostics; estimated diagnostics; the best router in the class; the learned router; the unconstrained feasible set; the deployed resource set; the unconstrained budget; the deployed budget; and the finite-sample decision. Adding and subtracting their risks yields an exact telescoping identity. Applying the triangle inequality gives

\[
\mathrm{Reg}_c(\hat\pi)\leq
\epsilon_{\mathrm{repr}}+\epsilon_{\mathrm{diag}}+
\epsilon_{\mathrm{route}}+\epsilon_{\mathrm{resource}}+
\epsilon_{\mathrm{budget}}+\epsilon_{\mathrm{abstain}}+
\epsilon_{\mathrm{sample}}.
\]

The individual nonnegative terms are definitions relative to the intermediate policies, not automatically identifiable causal effects. Resource, budget, and abstention penalties can be estimated by label-preserving offline counterfactuals after target labels are unlocked for evaluation. Representation error is generally theoretical, except in controlled mechanisms. Diagnostic and routing errors are source-validation estimable. Finite-sample error requires an inference bound or resampling plan.
