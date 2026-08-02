# CoReGraph theory strengthening plan

Status: `PROOF_SKETCH_INCOMPLETE`

The current proved results are exact but elementary. They establish a
fixed-mixture minimax-regret lower bound, an axis-additive approximation bound,
and feasible-oracle monotonicity. None is a finite-sample guarantee for the
learned router.

## Candidate finite-sample result

**Conjecture (group-robust compositional router generalisation).** Let source
examples be partitioned into \(G\) deployment-contract groups. Assume bounded
per-example loss, a factorised contract encoder with bounded norm, a
Lipschitz expert-aware router, a finite feasible expert set under every source
group, and an interaction residual bounded by
\(\epsilon_{\mathrm{int}}\). For empirical CVaR level \(\alpha\), a
regularised empirical minimiser should satisfy, with probability at least
\(1-\delta\),

\[
\operatorname{CVaR}_{\alpha}(R(\widehat r))
- \inf_{r\in\mathcal R}\operatorname{CVaR}_{\alpha}(R(r))
\;\lesssim\;
\frac{\mathfrak R_n(\mathcal R\circ\mathcal E)}
     {1-\alpha}
+ \sqrt{\frac{\log(G/\delta)}{n_{\min}(1-\alpha)^2}}
+ 2\epsilon_{\mathrm{int}},
\]

where \(n_{\min}\) is the smallest source-group sample count and
\(\mathfrak R_n\) is a group-aware Rademacher complexity for the composed
encoder/router class.

## Required proof work

1. State the sampling model explicitly: independent examples within groups
   versus graph-dependent observations need different concentration tools.
2. Prove uniform convergence of group risks for the composed contract encoder
   and masked router.
3. Lift group-risk deviations through empirical CVaR, retaining the
   \(1/(1-\alpha)\) dependence.
4. Treat learned abstention as an extra feasible action with declared cost and
   capacity, rather than assuming every group has an available expert.
5. Add the compositional approximation term without double-counting the
   interaction residual already present in the risk class.
6. Construct lower bounds showing the dependence on \(n_{\min}\), \(G\), and
   \(1-\alpha\) cannot all be removed.

No theorem, corollary, or empirical guarantee may cite this plan until the
sampling assumptions, constants, graph-dependence treatment, and proof are
complete. The only permitted label is `PROOF_SKETCH_INCOMPLETE`.
