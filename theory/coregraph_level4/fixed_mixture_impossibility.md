# Fixed-mixture impossibility

Status: `PROVED_INTERNAL_REVIEW_PENDING`.

Let contract 1 have expert risks \((0,\delta_1)\) and contract 2 have risks \((\delta_2,0)\), with \(\delta_1,\delta_2>0\). A contract-independent randomized selector uses expert 1 with probability \(w\). Its regrets are \((1-w)\delta_1\) and \(w\delta_2\). Their maximum is minimized at \(w=\delta_1/(\delta_1+\delta_2)\), giving the strictly positive value

\[
\frac{\delta_1\delta_2}{\delta_1+\delta_2}.
\]

A contract-aware selector chooses expert 1 for the first contract and expert 2 for the second, with zero regret. If a resource mask removes the contract-optimal expert, replace its risk by the minimum risk in the corresponding feasible set; the positive conclusion holds only when feasible-set optima still reverse.

The result fails when either gap is zero, one feasible expert is optimal everywhere, or the deployed mechanism is a nonlinear prediction-space mixture whose risk is not affine in selection mass.
