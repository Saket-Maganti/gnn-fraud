# Compositional generalisation

Status: `PROVED_INTERNAL_REVIEW_PENDING`.

For action \(a\), suppose

\[
R(a,c)=b(a)+\sum_{j=1}^{6}f_j(a,c_j)+h(a,c),
\qquad |h(a,c)|\leq\epsilon_{\mathrm{int}}.
\]

For source-observed axis values, assume \(|\hat f_j-f_j|\leq\epsilon_j\) uniformly, and let the router's optimization error under the estimated additive risk be at most \(\epsilon_{\mathrm{route}}\). On an unseen combination made only from these observed values, its excess true risk is at most

\[
2\sum_j\epsilon_j+2\epsilon_{\mathrm{int}}+\epsilon_{\mathrm{route}}.
\]

The proof applies the uniform approximation bound to the learned and true-optimal actions and inserts the optimization inequality. Bounded pairwise or low-rank interactions are covered by placing their unmodeled part in \(h\). The theorem gives no guarantee for an unseen axis value. XOR-type interactions with zero marginal effects and a large joint effect demonstrate non-identifiability and make a purely additive representation fail.
