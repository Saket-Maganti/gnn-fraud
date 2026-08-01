# Level-4 novelty ledger

Status: `SCHOLARLY_REVIEW_REQUIRED`; novelty is not established by implementation alone.

| Proposed contribution | Closest category | Distinguishing test | Promotion gate |
|---|---|---|---|
| Compositional deployment contracts | graph OOD/domain shift | unseen combinations of time, visibility, construction, selection, budget, resource | formal split audit plus protocol one-hot comparison |
| Expert-aware hierarchical routing | graph mixture of experts | contract prior with bounded per-instance correction over heterogeneous pretrained experts | contract-only and instance-only ablations |
| Hard resource-feasible routing | resource-aware MoE | exact mask, zero unavailable mass, explicit empty-set abstention, measurement provenance | resource counterfactual and profiler completeness |
| Robust contract regret objective | GroupDRO/VREx/CVaR | regret relative to a whole-contract feasible oracle under held-out composition | mean-risk, GroupDRO, VREx, CVaR comparisons |
| Controlled contract abstention | selective prediction | source-frozen threshold under metadata/resource uncertainty and review capacity | risk-coverage and zero-coverage audits |
| Three-layer evaluation | graph OOD and fraud benchmarks | real fraud, controlled mechanisms, recognized non-fraud OOD under one typed contract API | official adapter parity and licence gates |

FraudShiftBench/TKDE contributes evaluation and evidence semantics. CoReGraph/ICLR contributes a learning problem, router, objective, theory, and compositional evaluation. Shared datasets and infrastructure are disclosed; prior tables, figures, wording, and empirical claims are not reused as CoReGraph evidence.

The latent contract-discovery encoder is an optional extension. It remains secondary unless noisy/missing-metadata experiments beat both supplied-contract and no-contract controls under the frozen claim gate.
