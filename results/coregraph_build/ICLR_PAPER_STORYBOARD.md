# ICLR paper storyboard

The paper asks one question: can a graph system route among heterogeneous
experts under an unseen composition of deployment constraints without using
target labels and while respecting review and compute limits?

1. Motivate expert-order crossing across contracts and show why an atomic
   leaderboard or fixed mixture is insufficient.
2. Define the six-axis contract, task unit, admissible graph view, access regime,
   feasible expert set, and regret.
3. Present factorised contract encoding, label-free diagnostics, masks,
   CoReRouter, fallback, abstention, and the composite objective.
4. Prove the fixed-mixture lower bound and the axis-additive approximation bound.
5. Evaluate progressively: controlled mechanisms, saved-output pilot, five-seed
   screening, then frozen ten-seed fraud/GOOD/resource grids.
6. Lead results with worst-contract AUPRC and Recall@1%, regret, and average
   utility guardrails. Show compute and routing stability rather than only mean
   accuracy.

Pre-result figures 1--3 are illustrative and synthetic. Figures 4--6 and all
result tables remain data-driven placeholders until validated final manifests
exist.
