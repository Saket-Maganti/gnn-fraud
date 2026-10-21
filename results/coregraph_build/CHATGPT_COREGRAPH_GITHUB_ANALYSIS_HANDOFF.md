# Independent CoReGraph GitHub analysis handoff

Please independently analyse the CoReGraph build in:

Repository:
Saket-Maganti/gnn-fraud

Base branch:
codex/curated-fraudshiftbench-2026

CoReGraph branch:
codex/coregraph-iclr-buildout-2026

Commit:
7167d9b1604d1c896559704ffb8e8e244bc89113

Draft PR:
#2 — https://github.com/Saket-Maganti/gnn-fraud/pull/2

Start with:
1. results/coregraph_build/FINAL_COREGRAPH_BUILD_REPORT.md
2. results/coregraph_build/FINAL_GATE_STATUS.json
3. results/coregraph_build/PREBUILD_CODE_RISK_REGISTER.csv
4. results/coregraph_build/POSTBUILD_RISK_CLOSURE.csv
5. docs/coregraph/ARCHITECTURE.md
6. docs/coregraph/METHOD_CARD.md
7. docs/coregraph/LEAKAGE_GUARANTEES.md
8. docs/coregraph/STATISTICAL_ANALYSIS_PLAN.md
9. external_baselines/BASELINE_REGISTRY.yaml
10. coregraph/contracts/
11. coregraph/data/
12. coregraph/routing/
13. coregraph/objectives/
14. coregraph/experiments/pilot.py
15. tests/coregraph/
16. paper_iclr/

Audit whether:
- implementations are substantive rather than decorative scaffolds;
- tests match claimed guarantees;
- held-out-contract logic is leakage-safe;
- regret/CVaR and budget objectives are correct;
- resource masks behave correctly;
- V2 temporal semantics are defensible;
- official baseline labels are honest;
- theory statements match proofs;
- the pilot can test the intended ICLR claim;
- the project is genuinely distinct from FraudShiftBench;
- any defect must be repaired before baseline integration or runs.

Publication note: the commit above is the immutable implementation and
preflight SHA on which both GitHub workflows passed. The branch may advance by
one documentation-only commit containing this handoff and the final push
report; the exact current tip is available as the draft PR head.
