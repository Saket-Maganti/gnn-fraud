# Level-4 notation

Status: `PROVED_INTERNAL_REVIEW_PENDING` unless a result below says otherwise.

A deployment contract is (c=(c_t,c_v,c_g,c_s,c_b,c_r)), corresponding to time, visibility, construction, selection, budget, and resource. The source set is \(\mathcal C_s\); a target contract \(c_\star\) may contain observed axis values in a combination absent from \(\mathcal C_s\). Expert \(e\in\mathcal E\) is feasible only when \(e\in\mathcal A(c)\). A routing policy \(\pi_\theta(e\mid x,c,d)\) may use instance \(x\), declared contract metadata, and label-free diagnostics \(d\), but never target labels during fitting or deployment.

Risk is \(R_c(\pi)=\mathbb E_c[\ell(\pi(X),Y)]\). The whole-contract feasible oracle is \(R_c^\star=\min_{e\in\mathcal A(c)}R_c(e)\). Contract regret is \(R_c(\pi)-R_c^\star\). Worst-contract and CVaR regret aggregate this quantity over contracts. Selective risk conditions on acceptance; coverage is the acceptance probability. An instance-clairvoyant oracle is a diagnostic ceiling and is not the regret reference.
