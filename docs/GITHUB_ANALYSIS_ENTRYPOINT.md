# GitHub analysis entry point

## Read first

1. `README.md`
2. `docs/PROJECT_OVERVIEW.md`
3. `docs/REPOSITORY_MAP.md`
4. `docs/EVIDENCE_AND_CLAIM_MAP.md`
5. `paper/pdf/FraudShiftBench_TKDE_main.pdf`
6. `paper/pdf/FraudShiftBench_TKDE_supplement.pdf`
7. `results/tkde_visual_rebuild/FINAL_VISUAL_AND_EDITORIAL_READINESS_REPORT.md`
8. `results/tkde_visual_rebuild/FINAL_VISUAL_QA_REPORT.md`
9. `results/tkde_rebuild/FINAL_TKDE_READINESS_REPORT.md`
10. `results/tkde_rebuild/CLAIM_EVIDENCE_LEDGER.csv`
11. `results/tkde_rebuild/EVIDENCE_INVENTORY.csv`
12. `results/tkde_rebuild/NUMBER_PROVENANCE_MAP.csv`

## Canonical implementation surfaces

- contracts and metrics: `fraudshiftbench/protocols.py`,
  `fraudshiftbench/metrics.py`;
- evidence and claims: `fraudshiftbench/evidence.py`,
  `fraudshiftbench/claims.py`;
- model factory and extensions: `models/registry.py`,
  `models/temporal_calibration.py`, `models/graphsafe_v2.py`,
  `models/protocol_theory.py`;
- real/multi-dataset runners: `experiments/_multi_harness.py`,
  `experiments/run_multi_dataset.py`;
- evidence and statistics: `scripts/tkde_rebuild/build_evidence_inventory.py`,
  `build_claim_ledger.py`, `compute_analysis.py`,
  `validate_support_relation.py`;
- publication regeneration: `scripts/tkde_rebuild/make_figures.py`,
  `scripts/tkde_visual_rebuild/build_main_tables.py`,
  `build_curated_supplement_tables.py`.

## Main experiment families and boundaries

The evidence inventory covers real-graph visibility comparisons, IBM AML
baseline/construction/scale analyses, bounded GraphSafe saved-score analyses,
theoretical contract statements, and support-validator mutations. Raw
predictions, datasets, checkpoints, provider workspaces, and resource-blocked
outputs are intentionally absent.

Resource-blocked cells are listed in `docs/RESOURCE_BOUNDARIES.md`. They are not
predictive evidence. Scientific limitations are in
`docs/KNOWN_LIMITATIONS.md`.

The current open question is whether this foundation can support a distinct
ICLR-style method on contract-robust graph learning. The benchmark paper cannot
simply be relabeled as a method paper. GraphSafe is currently a bounded case,
and “ContractGuard” should be treated as a possible future extension point, not
an implemented or validated contribution.

## Analysis checklist

1. Verify paper novelty against current graph and benchmark literature.
2. Inspect deployment-contract formalism.
3. Inspect support-relation implementation.
4. Inspect evidence/claim mappings.
5. Inspect protocol and visibility implementations.
6. Inspect GraphSafe-TTA and ContractGuard extension points.
7. Inspect dataset/model coverage.
8. Identify a minimal held-out-contract pilot.
9. Assess reusable infrastructure for an ICLR submission.
10. Separate reusable methodology from fraud-specific assumptions.

