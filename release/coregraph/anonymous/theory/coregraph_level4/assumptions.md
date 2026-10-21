# Assumptions and proof obligations

1. Risks are defined under one declared loss and common row scope.
2. Expert feasibility is determined before routing and does not depend on target labels.
3. The fixed-mixture result concerns randomized expert selection, for which risk is affine in mixture mass. Nonlinear prediction averaging needs additional calibration/loss assumptions.
4. The compositional result covers unseen combinations of source-observed axis values. Axis effects have uniform estimation error, interactions have a bounded residual, and the learned action has bounded optimization error.
5. Resource costs used for feasibility have compatible units and conservative measurement status. An empty feasible set produces explicit abstention.
6. The selective-risk result assumes bounded loss, label-free acceptance, a finite target/source density-ratio bound on the accepted set, and a positive target-coverage lower bound.
7. Statistical statements pair dataset, target contract, and seed; equal seed numbers across datasets are not exchangeable observations.

Human review remains required for assumption-to-deployment correspondence. None of these assumptions follows from code execution alone.
