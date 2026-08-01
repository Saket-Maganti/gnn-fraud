# Selective-risk transfer

Status: `PROVED_INTERNAL_REVIEW_PENDING`.

Let \(A\) be the acceptance event chosen without target labels and let loss lie in \([0,L]\). Assume \(dP_t/dP_s\leq\kappa\) on accepted examples and target coverage \(P_t(A)\geq\gamma>0\). Then

\[
R_t^{\mathrm{sel}}=
\frac{\mathbb E_t[\ell\mathbf 1_A]}{P_t(A)}
\leq
\frac{\kappa P_s(A)R_s^{\mathrm{sel}}}{\gamma}
\leq L.
\]

The first inequality is change of measure plus the target-coverage lower bound. The result does not give a distribution-free target guarantee. If the density ratio is unbounded, the acceptance event uses target labels, or target coverage can approach zero, the bound is invalid or vacuous. Zero coverage is reported as `NOT_APPLICABLE_ZERO_COVERAGE`, not zero risk.
