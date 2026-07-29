# TKDE Evidence Inventory

This inventory is generated from canonical locks, imported indexes, saved-prediction manifests, and validation reports. One row is one claimable experimental cell over a complete seed set, or one explicit blocked/diagnostic/governance cell. It does not treat filenames, dry runs, missing outputs, or blocked lanes as performance evidence.

Total rows: **247**.

- `diagnostic-only`: 18 rows
- `excluded`: 4 rows
- `main case-study eligible`: 2 rows
- `main-paper eligible`: 102 rows
- `resource-blocked`: 7 rows
- `supplement-only`: 114 rows

The canonical machine-readable version is `EVIDENCE_INVENTORY.csv`; the compact table below omits long provenance fields only for readability.

## Conflict resolutions

- V22 evidence is under `kaggle_workspace/`, not `results/v22*`; the canonical V22 lock is recorded there.
- The legacy `P100` token in one V26 lane identifier is an alias, not hardware evidence. Imported records show `cuda:0`, and the normalized execution environment is Tesla T4.
- V24's memory-reduced DGraphFin GAT h32/l1 diagnostic does not replace the fixed h64/l2 T4-OOM cell.
- IBM AML Large, Medium GINE, DGraphFin RB30 extra architectures, and RB18 remain resource-blocked/blocked. They carry no performance metrics.
- V29-V39 are downstream analysis, contract, evaluator, or release surfaces. They do not supersede V26-V28 empirical locks.

## Compact inventory

| evidence_id | dataset | dataset_variant | protocol | model_or_configuration | seed_set | validation_status | eligibility |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RB09::dgraphfin::inductive_isolated::gcn | dgraphfin | canonical | inductive_isolated | GCN | 1-10 | PASS artifact family: 180 results and 180 prediction references | main-paper eligible |
| RB09::dgraphfin::inductive_isolated::mlp | dgraphfin | canonical | inductive_isolated | MLP | 1-10 | PASS artifact family: 180 results and 180 prediction references | main-paper eligible |
| RB09::dgraphfin::inductive_isolated::sage | dgraphfin | canonical | inductive_isolated | GraphSAGE | 1-10 | PASS artifact family: 180 results and 180 prediction references | main-paper eligible |
| RB09::dgraphfin::strict_inductive::gcn | dgraphfin | canonical | strict_inductive | GCN | 1-10 | PASS artifact family: 180 results and 180 prediction references | main-paper eligible |
| RB09::dgraphfin::strict_inductive::mlp | dgraphfin | canonical | strict_inductive | MLP | 1-10 | PASS artifact family: 180 results and 180 prediction references | main-paper eligible |
| RB09::dgraphfin::strict_inductive::sage | dgraphfin | canonical | strict_inductive | GraphSAGE | 1-10 | PASS artifact family: 180 results and 180 prediction references | main-paper eligible |
| RB09::dgraphfin::transductive::gcn | dgraphfin | canonical | transductive | GCN | 1-10 | PASS artifact family: 180 results and 180 prediction references | main-paper eligible |
| RB09::dgraphfin::transductive::mlp | dgraphfin | canonical | transductive | MLP | 1-10 | PASS artifact family: 180 results and 180 prediction references | main-paper eligible |
| RB09::dgraphfin::transductive::sage | dgraphfin | canonical | transductive | GraphSAGE | 1-10 | PASS artifact family: 180 results and 180 prediction references | main-paper eligible |
| RB09::elliptic::inductive_isolated::gcn | elliptic | canonical | inductive_isolated | GCN | 1-10 | PASS artifact family: 180 results and 180 prediction references | main-paper eligible |
| RB09::elliptic::inductive_isolated::mlp | elliptic | canonical | inductive_isolated | MLP | 1-10 | PASS artifact family: 180 results and 180 prediction references | main-paper eligible |
| RB09::elliptic::inductive_isolated::sage | elliptic | canonical | inductive_isolated | GraphSAGE | 1-10 | PASS artifact family: 180 results and 180 prediction references | main-paper eligible |
| RB09::elliptic::strict_inductive::gcn | elliptic | canonical | strict_inductive | GCN | 1-10 | PASS artifact family: 180 results and 180 prediction references | main-paper eligible |
| RB09::elliptic::strict_inductive::mlp | elliptic | canonical | strict_inductive | MLP | 1-10 | PASS artifact family: 180 results and 180 prediction references | main-paper eligible |
| RB09::elliptic::strict_inductive::sage | elliptic | canonical | strict_inductive | GraphSAGE | 1-10 | PASS artifact family: 180 results and 180 prediction references | main-paper eligible |
| RB09::elliptic::transductive::gcn | elliptic | canonical | transductive | GCN | 1-10 | PASS artifact family: 180 results and 180 prediction references | main-paper eligible |
| RB09::elliptic::transductive::mlp | elliptic | canonical | transductive | MLP | 1-10 | PASS artifact family: 180 results and 180 prediction references | main-paper eligible |
| RB09::elliptic::transductive::sage | elliptic | canonical | transductive | GraphSAGE | 1-10 | PASS artifact family: 180 results and 180 prediction references | main-paper eligible |
| RB11::elliptic::inductive_isolated::mlp | elliptic | canonical | inductive_isolated | MLP | 1-5 | aggregate and analysis manifests present | diagnostic-only |
| RB11::elliptic::inductive_isolated::sage | elliptic | canonical | inductive_isolated | GraphSAGE mean | 1-5 | aggregate and analysis manifests present | diagnostic-only |
| RB11::elliptic::inductive_isolated::sage_maxpool | elliptic | canonical | inductive_isolated | GraphSAGE max-pool | 1-5 | aggregate and analysis manifests present | diagnostic-only |
| RB11::elliptic::strict_inductive::mlp | elliptic | canonical | strict_inductive | MLP | 1-5 | aggregate and analysis manifests present | diagnostic-only |
| RB11::elliptic::strict_inductive::sage | elliptic | canonical | strict_inductive | GraphSAGE mean | 1-5 | aggregate and analysis manifests present | diagnostic-only |
| RB11::elliptic::strict_inductive::sage_maxpool | elliptic | canonical | strict_inductive | GraphSAGE max-pool | 1-5 | aggregate and analysis manifests present | diagnostic-only |
| RB15::saved_prediction_analysis | Elliptic; DGraphFin | canonical | saved-output policy evaluation | GraphSafe-TTA saved-output analysis | 1-10 | PASS | main case-study eligible |
| RB15b::saved_prediction_analysis | Elliptic; DGraphFin | canonical | saved-output policy evaluation | validation-selected conservative GraphSafe policy | 1-10 | PASS | main case-study eligible |
| RB16::saved_prediction_analysis | Elliptic; DGraphFin | canonical | saved-output policy evaluation | GraphSafe best-branch strengthening | 1-10 | PASS | supplement-only |
| RB17::saved_prediction_analysis | Elliptic; DGraphFin | canonical | saved-output policy evaluation | review-budget/worst-block/cost sensitivity | 1-10 | PASS | supplement-only |
| RB18::dgraphfin::sage_maxpool::BLOCKED | dgraphfin | canonical | strict_inductive and inductive_isolated planned | GraphSAGE max-pool | none | BLOCKED_WAITING_FOR_GPU | resource-blocked |
| V22::RB28::dgraphfin::inductive_isolated::gcn::default | dgraphfin | canonical | inductive_isolated | gcn; loss=default | 1-10 | PASS_FULL10_MERGED_WITH_COMPATIBLE_ALIASES | supplement-only |
| V22::RB28::dgraphfin::inductive_isolated::gcn::focal | dgraphfin | canonical | inductive_isolated | gcn; loss=focal | 1-10 | PASS_FULL10_MERGED_WITH_COMPATIBLE_ALIASES | supplement-only |
| V22::RB28::dgraphfin::inductive_isolated::gcn::weighted_bce | dgraphfin | canonical | inductive_isolated | gcn; loss=weighted_bce | 1-10 | PASS_FULL10_MERGED_WITH_COMPATIBLE_ALIASES | supplement-only |
| V22::RB28::dgraphfin::inductive_isolated::mlp::default | dgraphfin | canonical | inductive_isolated | mlp; loss=default | 1-10 | PASS_FULL10_MERGED_WITH_COMPATIBLE_ALIASES | supplement-only |
| V22::RB28::dgraphfin::inductive_isolated::mlp::focal | dgraphfin | canonical | inductive_isolated | mlp; loss=focal | 1-10 | PASS_FULL10_MERGED_WITH_COMPATIBLE_ALIASES | supplement-only |
| V22::RB28::dgraphfin::inductive_isolated::mlp::weighted_bce | dgraphfin | canonical | inductive_isolated | mlp; loss=weighted_bce | 1-10 | PASS_FULL10_MERGED_WITH_COMPATIBLE_ALIASES | supplement-only |
| V22::RB28::dgraphfin::inductive_isolated::sage::default | dgraphfin | canonical | inductive_isolated | sage; loss=default | 1-10 | PASS_FULL10_MERGED_WITH_COMPATIBLE_ALIASES | supplement-only |
| V22::RB28::dgraphfin::inductive_isolated::sage::focal | dgraphfin | canonical | inductive_isolated | sage; loss=focal | 1-10 | PASS_FULL10_MERGED_WITH_COMPATIBLE_ALIASES | supplement-only |
| V22::RB28::dgraphfin::inductive_isolated::sage::weighted_bce | dgraphfin | canonical | inductive_isolated | sage; loss=weighted_bce | 1-10 | PASS_FULL10_MERGED_WITH_COMPATIBLE_ALIASES | supplement-only |
| V22::RB28::dgraphfin::strict_inductive::gcn::default | dgraphfin | canonical | strict_inductive | gcn; loss=default | 1-10 | PASS_FULL10_MERGED_WITH_COMPATIBLE_ALIASES | supplement-only |
| V22::RB28::dgraphfin::strict_inductive::gcn::focal | dgraphfin | canonical | strict_inductive | gcn; loss=focal | 1-10 | PASS_FULL10_MERGED_WITH_COMPATIBLE_ALIASES | supplement-only |
| V22::RB28::dgraphfin::strict_inductive::gcn::weighted_bce | dgraphfin | canonical | strict_inductive | gcn; loss=weighted_bce | 1-10 | PASS_FULL10_MERGED_WITH_COMPATIBLE_ALIASES | supplement-only |
| V22::RB28::dgraphfin::strict_inductive::mlp::default | dgraphfin | canonical | strict_inductive | mlp; loss=default | 1-10 | PASS_FULL10_MERGED_WITH_COMPATIBLE_ALIASES | supplement-only |
| V22::RB28::dgraphfin::strict_inductive::mlp::focal | dgraphfin | canonical | strict_inductive | mlp; loss=focal | 1-10 | PASS_FULL10_MERGED_WITH_COMPATIBLE_ALIASES | supplement-only |
| V22::RB28::dgraphfin::strict_inductive::mlp::weighted_bce | dgraphfin | canonical | strict_inductive | mlp; loss=weighted_bce | 1-10 | PASS_FULL10_MERGED_WITH_COMPATIBLE_ALIASES | supplement-only |
| V22::RB28::dgraphfin::strict_inductive::sage::default | dgraphfin | canonical | strict_inductive | sage; loss=default | 1-10 | PASS_FULL10_MERGED_WITH_COMPATIBLE_ALIASES | supplement-only |
| V22::RB28::dgraphfin::strict_inductive::sage::focal | dgraphfin | canonical | strict_inductive | sage; loss=focal | 1-10 | PASS_FULL10_MERGED_WITH_COMPATIBLE_ALIASES | supplement-only |
| V22::RB28::dgraphfin::strict_inductive::sage::weighted_bce | dgraphfin | canonical | strict_inductive | sage; loss=weighted_bce | 1-10 | PASS_FULL10_MERGED_WITH_COMPATIBLE_ALIASES | supplement-only |
| V22::RB28::elliptic::inductive_isolated::gcn::default | elliptic | canonical | inductive_isolated | gcn; loss=default | 1-10 | PASS_FULL10_MERGED_WITH_COMPATIBLE_ALIASES | supplement-only |
| V22::RB28::elliptic::inductive_isolated::gcn::focal | elliptic | canonical | inductive_isolated | gcn; loss=focal | 1-10 | PASS_FULL10_MERGED_WITH_COMPATIBLE_ALIASES | supplement-only |
| V22::RB28::elliptic::inductive_isolated::gcn::weighted_bce | elliptic | canonical | inductive_isolated | gcn; loss=weighted_bce | 1-10 | PASS_FULL10_MERGED_WITH_COMPATIBLE_ALIASES | supplement-only |
| V22::RB28::elliptic::inductive_isolated::mlp::default | elliptic | canonical | inductive_isolated | mlp; loss=default | 1-10 | PASS_FULL10_MERGED_WITH_COMPATIBLE_ALIASES | supplement-only |
| V22::RB28::elliptic::inductive_isolated::mlp::focal | elliptic | canonical | inductive_isolated | mlp; loss=focal | 1-10 | PASS_FULL10_MERGED_WITH_COMPATIBLE_ALIASES | supplement-only |
| V22::RB28::elliptic::inductive_isolated::mlp::weighted_bce | elliptic | canonical | inductive_isolated | mlp; loss=weighted_bce | 1-10 | PASS_FULL10_MERGED_WITH_COMPATIBLE_ALIASES | supplement-only |
| V22::RB28::elliptic::inductive_isolated::sage::default | elliptic | canonical | inductive_isolated | sage; loss=default | 1-10 | PASS_FULL10_MERGED_WITH_COMPATIBLE_ALIASES | supplement-only |
| V22::RB28::elliptic::inductive_isolated::sage::focal | elliptic | canonical | inductive_isolated | sage; loss=focal | 1-10 | PASS_FULL10_MERGED_WITH_COMPATIBLE_ALIASES | supplement-only |
| V22::RB28::elliptic::inductive_isolated::sage::weighted_bce | elliptic | canonical | inductive_isolated | sage; loss=weighted_bce | 1-10 | PASS_FULL10_MERGED_WITH_COMPATIBLE_ALIASES | supplement-only |
| V22::RB28::elliptic::strict_inductive::gcn::default | elliptic | canonical | strict_inductive | gcn; loss=default | 1-10 | PASS_FULL10_MERGED_WITH_COMPATIBLE_ALIASES | supplement-only |
| V22::RB28::elliptic::strict_inductive::gcn::focal | elliptic | canonical | strict_inductive | gcn; loss=focal | 1-10 | PASS_FULL10_MERGED_WITH_COMPATIBLE_ALIASES | supplement-only |
| V22::RB28::elliptic::strict_inductive::gcn::weighted_bce | elliptic | canonical | strict_inductive | gcn; loss=weighted_bce | 1-10 | PASS_FULL10_MERGED_WITH_COMPATIBLE_ALIASES | supplement-only |
| V22::RB28::elliptic::strict_inductive::mlp::default | elliptic | canonical | strict_inductive | mlp; loss=default | 1-10 | PASS_FULL10_MERGED_WITH_COMPATIBLE_ALIASES | supplement-only |
| V22::RB28::elliptic::strict_inductive::mlp::focal | elliptic | canonical | strict_inductive | mlp; loss=focal | 1-10 | PASS_FULL10_MERGED_WITH_COMPATIBLE_ALIASES | supplement-only |
| V22::RB28::elliptic::strict_inductive::mlp::weighted_bce | elliptic | canonical | strict_inductive | mlp; loss=weighted_bce | 1-10 | PASS_FULL10_MERGED_WITH_COMPATIBLE_ALIASES | supplement-only |
| V22::RB28::elliptic::strict_inductive::sage::default | elliptic | canonical | strict_inductive | sage; loss=default | 1-10 | PASS_FULL10_MERGED_WITH_COMPATIBLE_ALIASES | supplement-only |
| V22::RB28::elliptic::strict_inductive::sage::focal | elliptic | canonical | strict_inductive | sage; loss=focal | 1-10 | PASS_FULL10_MERGED_WITH_COMPATIBLE_ALIASES | supplement-only |
| V22::RB28::elliptic::strict_inductive::sage::weighted_bce | elliptic | canonical | strict_inductive | sage; loss=weighted_bce | 1-10 | PASS_FULL10_MERGED_WITH_COMPATIBLE_ALIASES | supplement-only |
| V22::RB29::dgraphfin::inductive_isolated::mlp::feature_shuffle | dgraphfin | canonical | inductive_isolated | mlp; loss=weighted_bce; control=feature_shuffle | 1-10 | PASS_FULL10 | supplement-only |
| V22::RB29::dgraphfin::inductive_isolated::mlp::label_shuffle | dgraphfin | canonical | inductive_isolated | mlp; loss=weighted_bce; control=label_shuffle | 1-10 | PASS_FULL10 | supplement-only |
| V22::RB29::dgraphfin::inductive_isolated::sage::feature_shuffle | dgraphfin | canonical | inductive_isolated | sage; loss=weighted_bce; control=feature_shuffle | 1-10 | PASS_FULL10 | supplement-only |
| V22::RB29::dgraphfin::inductive_isolated::sage::label_shuffle | dgraphfin | canonical | inductive_isolated | sage; loss=weighted_bce; control=label_shuffle | 1-10 | PASS_FULL10 | supplement-only |
| V22::RB29::dgraphfin::strict_inductive::mlp::feature_shuffle | dgraphfin | canonical | strict_inductive | mlp; loss=weighted_bce; control=feature_shuffle | 1-10 | PASS_FULL10 | supplement-only |
| V22::RB29::dgraphfin::strict_inductive::mlp::label_shuffle | dgraphfin | canonical | strict_inductive | mlp; loss=weighted_bce; control=label_shuffle | 1-10 | PASS_FULL10 | supplement-only |
| V22::RB29::dgraphfin::strict_inductive::sage::feature_shuffle | dgraphfin | canonical | strict_inductive | sage; loss=weighted_bce; control=feature_shuffle | 1-10 | PASS_FULL10 | supplement-only |
| V22::RB29::dgraphfin::strict_inductive::sage::label_shuffle | dgraphfin | canonical | strict_inductive | sage; loss=weighted_bce; control=label_shuffle | 1-10 | PASS_FULL10 | supplement-only |
| V22::RB29::elliptic::inductive_isolated::mlp::edge_drop | elliptic | canonical | inductive_isolated | mlp; loss=weighted_bce; control=edge_drop | 1-10 | PASS_FULL10 | supplement-only |
| V22::RB29::elliptic::inductive_isolated::mlp::feature_shuffle | elliptic | canonical | inductive_isolated | mlp; loss=weighted_bce; control=feature_shuffle | 1-10 | PASS_FULL10 | supplement-only |
| V22::RB29::elliptic::inductive_isolated::mlp::label_shuffle | elliptic | canonical | inductive_isolated | mlp; loss=weighted_bce; control=label_shuffle | 1-10 | PASS_FULL10 | supplement-only |
| V22::RB29::elliptic::inductive_isolated::mlp::time_shuffle | elliptic | canonical | inductive_isolated | mlp; loss=weighted_bce; control=time_shuffle | 1-10 | PASS_FULL10 | supplement-only |
| V22::RB29::elliptic::inductive_isolated::sage::edge_drop | elliptic | canonical | inductive_isolated | sage; loss=weighted_bce; control=edge_drop | 1-10 | PASS_FULL10 | supplement-only |
| V22::RB29::elliptic::inductive_isolated::sage::feature_shuffle | elliptic | canonical | inductive_isolated | sage; loss=weighted_bce; control=feature_shuffle | 1-10 | PASS_FULL10 | supplement-only |
| V22::RB29::elliptic::inductive_isolated::sage::label_shuffle | elliptic | canonical | inductive_isolated | sage; loss=weighted_bce; control=label_shuffle | 1-10 | PASS_FULL10 | supplement-only |
| V22::RB29::elliptic::inductive_isolated::sage::time_shuffle | elliptic | canonical | inductive_isolated | sage; loss=weighted_bce; control=time_shuffle | 1-10 | PASS_FULL10 | supplement-only |
| V22::RB29::elliptic::strict_inductive::mlp::edge_drop | elliptic | canonical | strict_inductive | mlp; loss=weighted_bce; control=edge_drop | 1-10 | PASS_FULL10 | supplement-only |
| V22::RB29::elliptic::strict_inductive::mlp::feature_shuffle | elliptic | canonical | strict_inductive | mlp; loss=weighted_bce; control=feature_shuffle | 1-10 | PASS_FULL10 | supplement-only |
| V22::RB29::elliptic::strict_inductive::mlp::label_shuffle | elliptic | canonical | strict_inductive | mlp; loss=weighted_bce; control=label_shuffle | 1-10 | PASS_FULL10 | supplement-only |
| V22::RB29::elliptic::strict_inductive::mlp::time_shuffle | elliptic | canonical | strict_inductive | mlp; loss=weighted_bce; control=time_shuffle | 1-10 | PASS_FULL10 | supplement-only |
| V22::RB29::elliptic::strict_inductive::sage::edge_drop | elliptic | canonical | strict_inductive | sage; loss=weighted_bce; control=edge_drop | 1-10 | PASS_FULL10 | supplement-only |
| V22::RB29::elliptic::strict_inductive::sage::feature_shuffle | elliptic | canonical | strict_inductive | sage; loss=weighted_bce; control=feature_shuffle | 1-10 | PASS_FULL10 | supplement-only |
| V22::RB29::elliptic::strict_inductive::sage::label_shuffle | elliptic | canonical | strict_inductive | sage; loss=weighted_bce; control=label_shuffle | 1-10 | PASS_FULL10 | supplement-only |
| V22::RB29::elliptic::strict_inductive::sage::time_shuffle | elliptic | canonical | strict_inductive | sage; loss=weighted_bce; control=time_shuffle | 1-10 | PASS_FULL10 | supplement-only |
| V22::RB30::dgraphfin::BLOCKED | dgraphfin | canonical | not measured | GAT/GIN extra-architecture lane | none | BLOCKED_T4_OOM | resource-blocked |
| V22::RB30::elliptic::inductive_isolated::gat::weighted_bce | elliptic | canonical | inductive_isolated | gat; loss=weighted_bce | 1-10 | PASS_FULL10 | supplement-only |
| V22::RB30::elliptic::inductive_isolated::gin::weighted_bce | elliptic | canonical | inductive_isolated | gin; loss=weighted_bce | 1-10 | PASS_FULL10 | supplement-only |
| V22::RB30::elliptic::strict_inductive::gat::weighted_bce | elliptic | canonical | strict_inductive | gat; loss=weighted_bce | 1-10 | PASS_FULL10 | supplement-only |
| V22::RB30::elliptic::strict_inductive::gin::weighted_bce | elliptic | canonical | strict_inductive | gin; loss=weighted_bce | 1-10 | PASS_FULL10 | supplement-only |
| V22::excluded::RB28_ELLIPTIC_LOSS_ROBUSTNESS_SEEDS_1_2_DUALGPU_V22 | not inferable from directory name | partial/noncomparable | not promoted | not promoted | 1 | FAIL_MISSING_SEED | excluded |
| V22::excluded::RB28_ELLIPTIC_LOSS_ROBUSTNESS_SMOKE_2SEED_V20 | not inferable from directory name | partial/noncomparable | not promoted | not promoted | 1;2 | EXCLUDED_NOT_COMPARABLE | excluded |
| V22::excluded::RB30_ELLIPTIC_EXTRA_ARCH_SEEDS_1_2_DUALGPU_V22 | not inferable from directory name | partial/noncomparable | not promoted | not promoted | 1 | FAIL_DUPLICATE | excluded |
| V24::DGRAPHFIN_FIXED_CONFIG_GAT_HIDDEN64_LAYERS2::BLOCKED | dgraphfin | canonical | not measured | GAT hidden=64 layers=2 | none | BLOCKED_T4_OOM | resource-blocked |
| V24::RB41::dgraphfin::early_to_late_transfer::inductive_isolated::gcn | dgraphfin | canonical | inductive_isolated | gcn; hidden=64; layers=2 | 1-10 | PASS | supplement-only |
| V24::RB41::dgraphfin::early_to_late_transfer::inductive_isolated::mlp | dgraphfin | canonical | inductive_isolated | mlp; hidden=64; layers=2 | 1-10 | PASS | supplement-only |
| V24::RB41::dgraphfin::early_to_late_transfer::inductive_isolated::sage | dgraphfin | canonical | inductive_isolated | sage; hidden=64; layers=2 | 1-10 | PASS | supplement-only |
| V24::RB41::dgraphfin::early_to_late_transfer::strict_inductive::gcn | dgraphfin | canonical | strict_inductive | gcn; hidden=64; layers=2 | 1-10 | PASS | supplement-only |
| V24::RB41::dgraphfin::early_to_late_transfer::strict_inductive::mlp | dgraphfin | canonical | strict_inductive | mlp; hidden=64; layers=2 | 1-10 | PASS | supplement-only |
| V24::RB41::dgraphfin::early_to_late_transfer::strict_inductive::sage | dgraphfin | canonical | strict_inductive | sage; hidden=64; layers=2 | 1-10 | PASS | supplement-only |
| V24::RB41::dgraphfin::late_window_holdout::inductive_isolated::gcn | dgraphfin | canonical | inductive_isolated | gcn; hidden=64; layers=2 | 1-10 | PASS | supplement-only |
| V24::RB41::dgraphfin::late_window_holdout::inductive_isolated::mlp | dgraphfin | canonical | inductive_isolated | mlp; hidden=64; layers=2 | 1-10 | PASS | supplement-only |
| V24::RB41::dgraphfin::late_window_holdout::inductive_isolated::sage | dgraphfin | canonical | inductive_isolated | sage; hidden=64; layers=2 | 1-10 | PASS | supplement-only |
| V24::RB41::dgraphfin::late_window_holdout::strict_inductive::gcn | dgraphfin | canonical | strict_inductive | gcn; hidden=64; layers=2 | 1-10 | PASS | supplement-only |
| V24::RB41::dgraphfin::late_window_holdout::strict_inductive::mlp | dgraphfin | canonical | strict_inductive | mlp; hidden=64; layers=2 | 1-10 | PASS | supplement-only |
| V24::RB41::dgraphfin::late_window_holdout::strict_inductive::sage | dgraphfin | canonical | strict_inductive | sage; hidden=64; layers=2 | 1-10 | PASS | supplement-only |
| V24::RB41::dgraphfin::rolling_window_stress::inductive_isolated::gcn | dgraphfin | canonical | inductive_isolated | gcn; hidden=64; layers=2 | 1-10 | PASS | supplement-only |
| V24::RB41::dgraphfin::rolling_window_stress::inductive_isolated::mlp | dgraphfin | canonical | inductive_isolated | mlp; hidden=64; layers=2 | 1-10 | PASS | supplement-only |
| V24::RB41::dgraphfin::rolling_window_stress::inductive_isolated::sage | dgraphfin | canonical | inductive_isolated | sage; hidden=64; layers=2 | 1-10 | PASS | supplement-only |
| V24::RB41::dgraphfin::rolling_window_stress::strict_inductive::gcn | dgraphfin | canonical | strict_inductive | gcn; hidden=64; layers=2 | 1-10 | PASS | supplement-only |
| V24::RB41::dgraphfin::rolling_window_stress::strict_inductive::mlp | dgraphfin | canonical | strict_inductive | mlp; hidden=64; layers=2 | 1-10 | PASS | supplement-only |
| V24::RB41::dgraphfin::rolling_window_stress::strict_inductive::sage | dgraphfin | canonical | strict_inductive | sage; hidden=64; layers=2 | 1-10 | PASS | supplement-only |
| V24::RB41::elliptic::early_to_late_transfer::inductive_isolated::gcn | elliptic | canonical | inductive_isolated | gcn; hidden=128; layers=2 | 1-10 | PASS | supplement-only |
| V24::RB41::elliptic::early_to_late_transfer::inductive_isolated::mlp | elliptic | canonical | inductive_isolated | mlp; hidden=128; layers=2 | 1-10 | PASS | supplement-only |
| V24::RB41::elliptic::early_to_late_transfer::inductive_isolated::sage | elliptic | canonical | inductive_isolated | sage; hidden=128; layers=2 | 1-10 | PASS | supplement-only |
| V24::RB41::elliptic::early_to_late_transfer::strict_inductive::gcn | elliptic | canonical | strict_inductive | gcn; hidden=128; layers=2 | 1-10 | PASS | supplement-only |
| V24::RB41::elliptic::early_to_late_transfer::strict_inductive::mlp | elliptic | canonical | strict_inductive | mlp; hidden=128; layers=2 | 1-10 | PASS | supplement-only |
| V24::RB41::elliptic::early_to_late_transfer::strict_inductive::sage | elliptic | canonical | strict_inductive | sage; hidden=128; layers=2 | 1-10 | PASS | supplement-only |
| V24::RB41::elliptic::late_window_holdout::inductive_isolated::gcn | elliptic | canonical | inductive_isolated | gcn; hidden=128; layers=2 | 1-10 | PASS | supplement-only |
| V24::RB41::elliptic::late_window_holdout::inductive_isolated::mlp | elliptic | canonical | inductive_isolated | mlp; hidden=128; layers=2 | 1-10 | PASS | supplement-only |
| V24::RB41::elliptic::late_window_holdout::inductive_isolated::sage | elliptic | canonical | inductive_isolated | sage; hidden=128; layers=2 | 1-10 | PASS | supplement-only |
| V24::RB41::elliptic::late_window_holdout::strict_inductive::gcn | elliptic | canonical | strict_inductive | gcn; hidden=128; layers=2 | 1-10 | PASS | supplement-only |
| V24::RB41::elliptic::late_window_holdout::strict_inductive::mlp | elliptic | canonical | strict_inductive | mlp; hidden=128; layers=2 | 1-10 | PASS | supplement-only |
| V24::RB41::elliptic::late_window_holdout::strict_inductive::sage | elliptic | canonical | strict_inductive | sage; hidden=128; layers=2 | 1-10 | PASS | supplement-only |
| V24::RB41::elliptic::rolling_window_stress::inductive_isolated::gcn | elliptic | canonical | inductive_isolated | gcn; hidden=128; layers=2 | 1-10 | PASS | supplement-only |
| V24::RB41::elliptic::rolling_window_stress::inductive_isolated::mlp | elliptic | canonical | inductive_isolated | mlp; hidden=128; layers=2 | 1-10 | PASS | supplement-only |
| V24::RB41::elliptic::rolling_window_stress::inductive_isolated::sage | elliptic | canonical | inductive_isolated | sage; hidden=128; layers=2 | 1-10 | PASS | supplement-only |
| V24::RB41::elliptic::rolling_window_stress::strict_inductive::gcn | elliptic | canonical | strict_inductive | gcn; hidden=128; layers=2 | 1-10 | PASS | supplement-only |
| V24::RB41::elliptic::rolling_window_stress::strict_inductive::mlp | elliptic | canonical | strict_inductive | mlp; hidden=128; layers=2 | 1-10 | PASS | supplement-only |
| V24::RB41::elliptic::rolling_window_stress::strict_inductive::sage | elliptic | canonical | strict_inductive | sage; hidden=128; layers=2 | 1-10 | PASS | supplement-only |
| V24::RB44::elliptic::perturbation_edge_drop::inductive_isolated::gcn | elliptic | canonical | inductive_isolated | gcn; hidden=128; layers=2 | 1-10 | PASS | supplement-only |
| V24::RB44::elliptic::perturbation_edge_drop::inductive_isolated::mlp | elliptic | canonical | inductive_isolated | mlp; hidden=128; layers=2 | 1-10 | PASS | supplement-only |
| V24::RB44::elliptic::perturbation_edge_drop::inductive_isolated::sage | elliptic | canonical | inductive_isolated | sage; hidden=128; layers=2 | 1-10 | PASS | supplement-only |
| V24::RB44::elliptic::perturbation_edge_drop::strict_inductive::gcn | elliptic | canonical | strict_inductive | gcn; hidden=128; layers=2 | 1-10 | PASS | supplement-only |
| V24::RB44::elliptic::perturbation_edge_drop::strict_inductive::mlp | elliptic | canonical | strict_inductive | mlp; hidden=128; layers=2 | 1-10 | PASS | supplement-only |
| V24::RB44::elliptic::perturbation_edge_drop::strict_inductive::sage | elliptic | canonical | strict_inductive | sage; hidden=128; layers=2 | 1-10 | PASS | supplement-only |
| V24::RB44::elliptic::perturbation_feature_shuffle::inductive_isolated::gcn | elliptic | canonical | inductive_isolated | gcn; hidden=128; layers=2 | 1-10 | PASS | supplement-only |
| V24::RB44::elliptic::perturbation_feature_shuffle::inductive_isolated::mlp | elliptic | canonical | inductive_isolated | mlp; hidden=128; layers=2 | 1-10 | PASS | supplement-only |
| V24::RB44::elliptic::perturbation_feature_shuffle::inductive_isolated::sage | elliptic | canonical | inductive_isolated | sage; hidden=128; layers=2 | 1-10 | PASS | supplement-only |
| V24::RB44::elliptic::perturbation_feature_shuffle::strict_inductive::gcn | elliptic | canonical | strict_inductive | gcn; hidden=128; layers=2 | 1-10 | PASS | supplement-only |
| V24::RB44::elliptic::perturbation_feature_shuffle::strict_inductive::mlp | elliptic | canonical | strict_inductive | mlp; hidden=128; layers=2 | 1-10 | PASS | supplement-only |
| V24::RB44::elliptic::perturbation_feature_shuffle::strict_inductive::sage | elliptic | canonical | strict_inductive | sage; hidden=128; layers=2 | 1-10 | PASS | supplement-only |
| V24::RB45::dgraphfin::memory_reduced_exploratory_not_comparable::inductive_isolated::gat | dgraphfin | canonical | inductive_isolated | gat; hidden=32; layers=1 | 1-10 | EXPLORATORY_PASS | diagnostic-only |
| V24::RB45::dgraphfin::memory_reduced_exploratory_not_comparable::strict_inductive::gat | dgraphfin | canonical | strict_inductive | gat; hidden=32; layers=1 | 1-10 | EXPLORATORY_PASS | diagnostic-only |
| V26::ibm_aml::hi-large::BLOCKED | IBM AML-Data | hi-large | not measured | baseline grid not executed | none | SAFE_RESOURCE_BLOCKED | resource-blocked |
| V26::ibm_aml::hi-medium::early_to_late_transfer::graphsage_edge_minibatch_h32 | IBM AML-Data | hi-medium | early_to_late_transfer | one-hop GraphSAGE-derived edge classifier h32 | 1-10 | PASS | main-paper eligible |
| V26::ibm_aml::hi-medium::early_to_late_transfer::hist_gradient_boosting_edge_features | IBM AML-Data | hi-medium | early_to_late_transfer | histogram gradient boosting on transaction features | 1-10 | PASS | main-paper eligible |
| V26::ibm_aml::hi-medium::early_to_late_transfer::logistic_regression_edge_features | IBM AML-Data | hi-medium | early_to_late_transfer | logistic regression on transaction features | 1-10 | PASS | main-paper eligible |
| V26::ibm_aml::hi-medium::late_window_holdout::graphsage_edge_minibatch_h32 | IBM AML-Data | hi-medium | late_window_holdout | one-hop GraphSAGE-derived edge classifier h32 | 1-10 | PASS | main-paper eligible |
| V26::ibm_aml::hi-medium::late_window_holdout::hist_gradient_boosting_edge_features | IBM AML-Data | hi-medium | late_window_holdout | histogram gradient boosting on transaction features | 1-10 | PASS | main-paper eligible |
| V26::ibm_aml::hi-medium::late_window_holdout::logistic_regression_edge_features | IBM AML-Data | hi-medium | late_window_holdout | logistic regression on transaction features | 1-10 | PASS | main-paper eligible |
| V26::ibm_aml::hi-small::early_to_late_transfer::graphsage_edge_minibatch_h32 | IBM AML-Data | hi-small | early_to_late_transfer | one-hop GraphSAGE-derived edge classifier h32 | 1-10 | PASS | main-paper eligible |
| V26::ibm_aml::hi-small::early_to_late_transfer::hist_gradient_boosting_edge_features | IBM AML-Data | hi-small | early_to_late_transfer | histogram gradient boosting on transaction features | 1-10 | PASS | main-paper eligible |
| V26::ibm_aml::hi-small::early_to_late_transfer::logistic_regression_edge_features | IBM AML-Data | hi-small | early_to_late_transfer | logistic regression on transaction features | 1-10 | PASS | main-paper eligible |
| V26::ibm_aml::hi-small::late_window_holdout::graphsage_edge_minibatch_h32 | IBM AML-Data | hi-small | late_window_holdout | one-hop GraphSAGE-derived edge classifier h32 | 1-10 | PASS | main-paper eligible |
| V26::ibm_aml::hi-small::late_window_holdout::hist_gradient_boosting_edge_features | IBM AML-Data | hi-small | late_window_holdout | histogram gradient boosting on transaction features | 1-10 | PASS | main-paper eligible |
| V26::ibm_aml::hi-small::late_window_holdout::logistic_regression_edge_features | IBM AML-Data | hi-small | late_window_holdout | logistic regression on transaction features | 1-10 | PASS | main-paper eligible |
| V26::ibm_aml::li-large::BLOCKED | IBM AML-Data | li-large | not measured | baseline grid not executed | none | SAFE_RESOURCE_BLOCKED | resource-blocked |
| V26::ibm_aml::li-medium::early_to_late_transfer::graphsage_edge_minibatch_h32 | IBM AML-Data | li-medium | early_to_late_transfer | one-hop GraphSAGE-derived edge classifier h32 | 1-10 | PASS | main-paper eligible |
| V26::ibm_aml::li-medium::early_to_late_transfer::hist_gradient_boosting_edge_features | IBM AML-Data | li-medium | early_to_late_transfer | histogram gradient boosting on transaction features | 1-10 | PASS | main-paper eligible |
| V26::ibm_aml::li-medium::early_to_late_transfer::logistic_regression_edge_features | IBM AML-Data | li-medium | early_to_late_transfer | logistic regression on transaction features | 1-10 | PASS | main-paper eligible |
| V26::ibm_aml::li-medium::late_window_holdout::graphsage_edge_minibatch_h32 | IBM AML-Data | li-medium | late_window_holdout | one-hop GraphSAGE-derived edge classifier h32 | 1-10 | PASS | main-paper eligible |
| V26::ibm_aml::li-medium::late_window_holdout::hist_gradient_boosting_edge_features | IBM AML-Data | li-medium | late_window_holdout | histogram gradient boosting on transaction features | 1-10 | PASS | main-paper eligible |
| V26::ibm_aml::li-medium::late_window_holdout::logistic_regression_edge_features | IBM AML-Data | li-medium | late_window_holdout | logistic regression on transaction features | 1-10 | PASS | main-paper eligible |
| V26::ibm_aml::li-small::early_to_late_transfer::graphsage_edge_minibatch_h32 | IBM AML-Data | li-small | early_to_late_transfer | one-hop GraphSAGE-derived edge classifier h32 | 1-10 | PASS | main-paper eligible |
| V26::ibm_aml::li-small::early_to_late_transfer::hist_gradient_boosting_edge_features | IBM AML-Data | li-small | early_to_late_transfer | histogram gradient boosting on transaction features | 1-10 | PASS | main-paper eligible |
| V26::ibm_aml::li-small::early_to_late_transfer::logistic_regression_edge_features | IBM AML-Data | li-small | early_to_late_transfer | logistic regression on transaction features | 1-10 | PASS | main-paper eligible |
| V26::ibm_aml::li-small::late_window_holdout::graphsage_edge_minibatch_h32 | IBM AML-Data | li-small | late_window_holdout | one-hop GraphSAGE-derived edge classifier h32 | 1-10 | PASS | main-paper eligible |
| V26::ibm_aml::li-small::late_window_holdout::hist_gradient_boosting_edge_features | IBM AML-Data | li-small | late_window_holdout | histogram gradient boosting on transaction features | 1-10 | PASS | main-paper eligible |
| V26::ibm_aml::li-small::late_window_holdout::logistic_regression_edge_features | IBM AML-Data | li-small | late_window_holdout | logistic regression on transaction features | 1-10 | PASS | main-paper eligible |
| V27::ibm_aml::hi-medium::early_to_late_transfer::edge_aware_graphsage_h64 | IBM AML-Data | hi-medium | early_to_late_transfer | edge-aware one-hop GraphSAGE-derived edge classifier h64 | 1-10 | PASS | main-paper eligible |
| V27::ibm_aml::hi-medium::late_window_holdout::edge_aware_graphsage_h64 | IBM AML-Data | hi-medium | late_window_holdout | edge-aware one-hop GraphSAGE-derived edge classifier h64 | 1-10 | PASS | main-paper eligible |
| V27::ibm_aml::hi-small::early_to_late_transfer::edge_aware_graphsage_h64 | IBM AML-Data | hi-small | early_to_late_transfer | edge-aware one-hop GraphSAGE-derived edge classifier h64 | 1-10 | PASS | main-paper eligible |
| V27::ibm_aml::hi-small::late_window_holdout::edge_aware_graphsage_h64 | IBM AML-Data | hi-small | late_window_holdout | edge-aware one-hop GraphSAGE-derived edge classifier h64 | 1-10 | PASS | main-paper eligible |
| V27::ibm_aml::li-medium::early_to_late_transfer::edge_aware_graphsage_h64 | IBM AML-Data | li-medium | early_to_late_transfer | edge-aware one-hop GraphSAGE-derived edge classifier h64 | 1-10 | PASS | main-paper eligible |
| V27::ibm_aml::li-medium::late_window_holdout::edge_aware_graphsage_h64 | IBM AML-Data | li-medium | late_window_holdout | edge-aware one-hop GraphSAGE-derived edge classifier h64 | 1-10 | PASS | main-paper eligible |
| V27::ibm_aml::li-small::early_to_late_transfer::edge_aware_graphsage_h64 | IBM AML-Data | li-small | early_to_late_transfer | edge-aware one-hop GraphSAGE-derived edge classifier h64 | 1-10 | PASS | main-paper eligible |
| V27::ibm_aml::li-small::late_window_holdout::edge_aware_graphsage_h64 | IBM AML-Data | li-small | late_window_holdout | edge-aware one-hop GraphSAGE-derived edge classifier h64 | 1-10 | PASS | main-paper eligible |
| V28::ibm_aml::hi-medium::early_to_late_transfer::account_account_sender_receiver | IBM AML-Data | hi-medium | early_to_late_transfer | edge-aware GraphSAGE h64, sender-receiver construction | 1-10 | PASS | main-paper eligible |
| V28::ibm_aml::hi-medium::early_to_late_transfer::degree_capped_bipartite | IBM AML-Data | hi-medium | early_to_late_transfer | edge-aware GraphSAGE h64, DegreeCap | 1-10 | PASS | main-paper eligible |
| V28::ibm_aml::hi-medium::early_to_late_transfer::edge_aware_graphsage_h64_degree_only | IBM AML-Data | hi-medium | early_to_late_transfer | edge-aware GraphSAGE h64, DegreeOnly | 1-10 | PASS | main-paper eligible |
| V28::ibm_aml::hi-medium::early_to_late_transfer::edge_aware_graphsage_h64_no_edge_features | IBM AML-Data | hi-medium | early_to_late_transfer | edge-aware GraphSAGE h64, NoEdge | 1-10 | PASS | main-paper eligible |
| V28::ibm_aml::hi-medium::early_to_late_transfer::edge_aware_graphsage_h64_shuffled_edge_features | IBM AML-Data | hi-medium | early_to_late_transfer | edge-aware GraphSAGE h64, ShuffledEdge | 1-10 | PASS | main-paper eligible |
| V28::ibm_aml::hi-medium::early_to_late_transfer::recent_window_only_graph | IBM AML-Data | hi-medium | early_to_late_transfer | edge-aware GraphSAGE h64, RecentWindow | 1-10 | PASS | main-paper eligible |
| V28::ibm_aml::hi-medium::gine_light_h64::BLOCKED | IBM AML-Data | hi-medium | late-window holdout and early-to-late transfer planned | one-layer GINE edge classifier h64 | none | RESOURCE_BLOCKED_T4_CUDA_OOM | resource-blocked |
| V28::ibm_aml::hi-medium::late_window_holdout::account_account_sender_receiver | IBM AML-Data | hi-medium | late_window_holdout | edge-aware GraphSAGE h64, sender-receiver construction | 1-10 | PASS | main-paper eligible |
| V28::ibm_aml::hi-medium::late_window_holdout::degree_capped_bipartite | IBM AML-Data | hi-medium | late_window_holdout | edge-aware GraphSAGE h64, DegreeCap | 1-10 | PASS | main-paper eligible |
| V28::ibm_aml::hi-medium::late_window_holdout::edge_aware_graphsage_h64_degree_only | IBM AML-Data | hi-medium | late_window_holdout | edge-aware GraphSAGE h64, DegreeOnly | 1-10 | PASS | main-paper eligible |
| V28::ibm_aml::hi-medium::late_window_holdout::edge_aware_graphsage_h64_no_edge_features | IBM AML-Data | hi-medium | late_window_holdout | edge-aware GraphSAGE h64, NoEdge | 1-10 | PASS | main-paper eligible |
| V28::ibm_aml::hi-medium::late_window_holdout::edge_aware_graphsage_h64_shuffled_edge_features | IBM AML-Data | hi-medium | late_window_holdout | edge-aware GraphSAGE h64, ShuffledEdge | 1-10 | PASS | main-paper eligible |
| V28::ibm_aml::hi-medium::late_window_holdout::recent_window_only_graph | IBM AML-Data | hi-medium | late_window_holdout | edge-aware GraphSAGE h64, RecentWindow | 1-10 | PASS | main-paper eligible |
| V28::ibm_aml::hi-small::early_to_late_transfer::account_account_sender_receiver | IBM AML-Data | hi-small | early_to_late_transfer | edge-aware GraphSAGE h64, sender-receiver construction | 1-10 | PASS | main-paper eligible |
| V28::ibm_aml::hi-small::early_to_late_transfer::degree_capped_bipartite | IBM AML-Data | hi-small | early_to_late_transfer | edge-aware GraphSAGE h64, DegreeCap | 1-10 | PASS | main-paper eligible |
| V28::ibm_aml::hi-small::early_to_late_transfer::edge_aware_graphsage_h64_degree_only | IBM AML-Data | hi-small | early_to_late_transfer | edge-aware GraphSAGE h64, DegreeOnly | 1-10 | PASS | main-paper eligible |
| V28::ibm_aml::hi-small::early_to_late_transfer::edge_aware_graphsage_h64_no_edge_features | IBM AML-Data | hi-small | early_to_late_transfer | edge-aware GraphSAGE h64, NoEdge | 1-10 | PASS | main-paper eligible |
| V28::ibm_aml::hi-small::early_to_late_transfer::edge_aware_graphsage_h64_shuffled_edge_features | IBM AML-Data | hi-small | early_to_late_transfer | edge-aware GraphSAGE h64, ShuffledEdge | 1-10 | PASS | main-paper eligible |
| V28::ibm_aml::hi-small::early_to_late_transfer::gine_light_h64 | IBM AML-Data | hi-small | early_to_late_transfer | one-layer GINE edge classifier h64 | 1-10 | PASS | main-paper eligible |
| V28::ibm_aml::hi-small::early_to_late_transfer::recent_window_only_graph | IBM AML-Data | hi-small | early_to_late_transfer | edge-aware GraphSAGE h64, RecentWindow | 1-10 | PASS | main-paper eligible |
| V28::ibm_aml::hi-small::late_window_holdout::account_account_sender_receiver | IBM AML-Data | hi-small | late_window_holdout | edge-aware GraphSAGE h64, sender-receiver construction | 1-10 | PASS | main-paper eligible |
| V28::ibm_aml::hi-small::late_window_holdout::degree_capped_bipartite | IBM AML-Data | hi-small | late_window_holdout | edge-aware GraphSAGE h64, DegreeCap | 1-10 | PASS | main-paper eligible |
| V28::ibm_aml::hi-small::late_window_holdout::edge_aware_graphsage_h64_degree_only | IBM AML-Data | hi-small | late_window_holdout | edge-aware GraphSAGE h64, DegreeOnly | 1-10 | PASS | main-paper eligible |
| V28::ibm_aml::hi-small::late_window_holdout::edge_aware_graphsage_h64_no_edge_features | IBM AML-Data | hi-small | late_window_holdout | edge-aware GraphSAGE h64, NoEdge | 1-10 | PASS | main-paper eligible |
| V28::ibm_aml::hi-small::late_window_holdout::edge_aware_graphsage_h64_shuffled_edge_features | IBM AML-Data | hi-small | late_window_holdout | edge-aware GraphSAGE h64, ShuffledEdge | 1-10 | PASS | main-paper eligible |
| V28::ibm_aml::hi-small::late_window_holdout::gine_light_h64 | IBM AML-Data | hi-small | late_window_holdout | one-layer GINE edge classifier h64 | 1-10 | PASS | main-paper eligible |
| V28::ibm_aml::hi-small::late_window_holdout::recent_window_only_graph | IBM AML-Data | hi-small | late_window_holdout | edge-aware GraphSAGE h64, RecentWindow | 1-10 | PASS | main-paper eligible |
| V28::ibm_aml::li-medium::early_to_late_transfer::account_account_sender_receiver | IBM AML-Data | li-medium | early_to_late_transfer | edge-aware GraphSAGE h64, sender-receiver construction | 1-10 | PASS | main-paper eligible |
| V28::ibm_aml::li-medium::early_to_late_transfer::degree_capped_bipartite | IBM AML-Data | li-medium | early_to_late_transfer | edge-aware GraphSAGE h64, DegreeCap | 1-10 | PASS | main-paper eligible |
| V28::ibm_aml::li-medium::early_to_late_transfer::edge_aware_graphsage_h64_degree_only | IBM AML-Data | li-medium | early_to_late_transfer | edge-aware GraphSAGE h64, DegreeOnly | 1-10 | PASS | main-paper eligible |
| V28::ibm_aml::li-medium::early_to_late_transfer::edge_aware_graphsage_h64_no_edge_features | IBM AML-Data | li-medium | early_to_late_transfer | edge-aware GraphSAGE h64, NoEdge | 1-10 | PASS | main-paper eligible |
| V28::ibm_aml::li-medium::early_to_late_transfer::edge_aware_graphsage_h64_shuffled_edge_features | IBM AML-Data | li-medium | early_to_late_transfer | edge-aware GraphSAGE h64, ShuffledEdge | 1-10 | PASS | main-paper eligible |
| V28::ibm_aml::li-medium::early_to_late_transfer::recent_window_only_graph | IBM AML-Data | li-medium | early_to_late_transfer | edge-aware GraphSAGE h64, RecentWindow | 1-10 | PASS | main-paper eligible |
| V28::ibm_aml::li-medium::gine_light_h64::BLOCKED | IBM AML-Data | li-medium | late-window holdout and early-to-late transfer planned | one-layer GINE edge classifier h64 | none | RESOURCE_BLOCKED_T4_CUDA_OOM | resource-blocked |
| V28::ibm_aml::li-medium::late_window_holdout::account_account_sender_receiver | IBM AML-Data | li-medium | late_window_holdout | edge-aware GraphSAGE h64, sender-receiver construction | 1-10 | PASS | main-paper eligible |
| V28::ibm_aml::li-medium::late_window_holdout::degree_capped_bipartite | IBM AML-Data | li-medium | late_window_holdout | edge-aware GraphSAGE h64, DegreeCap | 1-10 | PASS | main-paper eligible |
| V28::ibm_aml::li-medium::late_window_holdout::edge_aware_graphsage_h64_degree_only | IBM AML-Data | li-medium | late_window_holdout | edge-aware GraphSAGE h64, DegreeOnly | 1-10 | PASS | main-paper eligible |
| V28::ibm_aml::li-medium::late_window_holdout::edge_aware_graphsage_h64_no_edge_features | IBM AML-Data | li-medium | late_window_holdout | edge-aware GraphSAGE h64, NoEdge | 1-10 | PASS | main-paper eligible |
| V28::ibm_aml::li-medium::late_window_holdout::edge_aware_graphsage_h64_shuffled_edge_features | IBM AML-Data | li-medium | late_window_holdout | edge-aware GraphSAGE h64, ShuffledEdge | 1-10 | PASS | main-paper eligible |
| V28::ibm_aml::li-medium::late_window_holdout::recent_window_only_graph | IBM AML-Data | li-medium | late_window_holdout | edge-aware GraphSAGE h64, RecentWindow | 1-10 | PASS | main-paper eligible |
| V28::ibm_aml::li-small::early_to_late_transfer::account_account_sender_receiver | IBM AML-Data | li-small | early_to_late_transfer | edge-aware GraphSAGE h64, sender-receiver construction | 1-10 | PASS | main-paper eligible |
| V28::ibm_aml::li-small::early_to_late_transfer::degree_capped_bipartite | IBM AML-Data | li-small | early_to_late_transfer | edge-aware GraphSAGE h64, DegreeCap | 1-10 | PASS | main-paper eligible |
| V28::ibm_aml::li-small::early_to_late_transfer::edge_aware_graphsage_h64_degree_only | IBM AML-Data | li-small | early_to_late_transfer | edge-aware GraphSAGE h64, DegreeOnly | 1-10 | PASS | main-paper eligible |
| V28::ibm_aml::li-small::early_to_late_transfer::edge_aware_graphsage_h64_no_edge_features | IBM AML-Data | li-small | early_to_late_transfer | edge-aware GraphSAGE h64, NoEdge | 1-10 | PASS | main-paper eligible |
| V28::ibm_aml::li-small::early_to_late_transfer::edge_aware_graphsage_h64_shuffled_edge_features | IBM AML-Data | li-small | early_to_late_transfer | edge-aware GraphSAGE h64, ShuffledEdge | 1-10 | PASS | main-paper eligible |
| V28::ibm_aml::li-small::early_to_late_transfer::gine_light_h64 | IBM AML-Data | li-small | early_to_late_transfer | one-layer GINE edge classifier h64 | 1-10 | PASS | main-paper eligible |
| V28::ibm_aml::li-small::early_to_late_transfer::recent_window_only_graph | IBM AML-Data | li-small | early_to_late_transfer | edge-aware GraphSAGE h64, RecentWindow | 1-10 | PASS | main-paper eligible |
| V28::ibm_aml::li-small::late_window_holdout::account_account_sender_receiver | IBM AML-Data | li-small | late_window_holdout | edge-aware GraphSAGE h64, sender-receiver construction | 1-10 | PASS | main-paper eligible |
| V28::ibm_aml::li-small::late_window_holdout::degree_capped_bipartite | IBM AML-Data | li-small | late_window_holdout | edge-aware GraphSAGE h64, DegreeCap | 1-10 | PASS | main-paper eligible |
| V28::ibm_aml::li-small::late_window_holdout::edge_aware_graphsage_h64_degree_only | IBM AML-Data | li-small | late_window_holdout | edge-aware GraphSAGE h64, DegreeOnly | 1-10 | PASS | main-paper eligible |
| V28::ibm_aml::li-small::late_window_holdout::edge_aware_graphsage_h64_no_edge_features | IBM AML-Data | li-small | late_window_holdout | edge-aware GraphSAGE h64, NoEdge | 1-10 | PASS | main-paper eligible |
| V28::ibm_aml::li-small::late_window_holdout::edge_aware_graphsage_h64_shuffled_edge_features | IBM AML-Data | li-small | late_window_holdout | edge-aware GraphSAGE h64, ShuffledEdge | 1-10 | PASS | main-paper eligible |
| V28::ibm_aml::li-small::late_window_holdout::gine_light_h64 | IBM AML-Data | li-small | late_window_holdout | one-layer GINE edge classifier h64 | 1-10 | PASS | main-paper eligible |
| V28::ibm_aml::li-small::late_window_holdout::recent_window_only_graph | IBM AML-Data | li-small | late_window_holdout | edge-aware GraphSAGE h64, RecentWindow | 1-10 | PASS | main-paper eligible |
| V29::governance_surface | inherits V26-V28/RB families | not an independent experiment | audit/tooling | not applicable | inherits sources | present | diagnostic-only |
| V30::governance_surface | none | not an independent experiment | audit/tooling | not applicable | none | MISSING_DIRECTORY | excluded |
| V31::governance_surface | inherits V26-V28/RB families | not an independent experiment | audit/tooling | not applicable | inherits sources | present | diagnostic-only |
| V32::governance_surface | inherits V26-V28/RB families | not an independent experiment | audit/tooling | not applicable | inherits sources | present | diagnostic-only |
| V33::governance_surface | inherits V26-V28/RB families | not an independent experiment | audit/tooling | not applicable | inherits sources | present | diagnostic-only |
| V34::governance_surface | inherits V26-V28/RB families | not an independent experiment | audit/tooling | not applicable | inherits sources | present | diagnostic-only |
| V35::governance_surface | inherits V26-V28/RB families | not an independent experiment | audit/tooling | not applicable | inherits sources | present | diagnostic-only |
| V36::governance_surface | inherits V26-V28/RB families | not an independent experiment | audit/tooling | not applicable | inherits sources | present | diagnostic-only |
| V37::governance_surface | inherits V26-V28/RB families | not an independent experiment | audit/tooling | not applicable | inherits sources | present | diagnostic-only |
| V38::governance_surface | inherits V26-V28/RB families | not an independent experiment | audit/tooling | not applicable | inherits sources | present | diagnostic-only |
| V39::governance_surface | inherits V26-V28/RB families | not an independent experiment | audit/tooling | not applicable | inherits sources | present | diagnostic-only |

