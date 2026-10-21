# Raw table artifact index

The visual rebuild removes four exhaustive row dumps from the rendered supplement. The scientific rows are unchanged and remain the machine-readable source of truth. The generator verifies each frozen row count and SHA-256 before producing any curated table.

| Object | Rows | Columns | Artifact path | SHA-256 | Rendered replacement |
| --- | ---: | ---: | --- | --- | --- |
| RB09_SEED_GRID | 180 | 32 | results/runs_rb09v3/runs.csv | d9f77bfd14ccaf157780858e35ecfee96af1e2fbf60f3559ad888d14c0ace2e9 | Tables S14--S16: matched protocol effects by metric |
| V22_PAIRED_TESTS | 198 | 18 | manuscript_assets/tables/V22_STAT_TESTS_FULL10.csv | 513f1fb5c2dbae2f63255fd384c62fe86394a15746c31792a92a1f287b8ddadd | Table S19: correction-family aggregate |
| IBM_SEED_GRID | 840 | 19 | results/tkde_rebuild/IBM_IMPORTED_SEED_ROWS.csv | dae5d51178b9a746946c9fc05bf31e74840b6663d69031597ef2be45d5e766b6 | Tables S20 and S21--S25: baseline and matched aggregates |
| IBM_CONTEXT_EFFECTS | 208 | 14 | results/tkde_rebuild/IBM_MATCHED_ABLATION_CONTEXT_EFFECTS.csv | 626f0e014a0b8d339fe344bb736d513ed63a6ec435be6104acad1f53f7bbe202 | Tables S21--S23: ten-seed effects after fixed-context aggregation |

## Schemas

### RB09_SEED_GRID

- Purpose: complete node-protocol seed grid
- Path: results/runs_rb09v3/runs.csv
- Data rows: 180
- SHA-256: d9f77bfd14ccaf157780858e35ecfee96af1e2fbf60f3559ad888d14c0ace2e9
- Columns in order: dataset, protocol, split_name, model, seed, early_stopping_metric, early_stopping_split, scaler_mode, graph_mode, train_nodes, val_nodes, test_nodes, positives_train, positives_val, positives_test, f1, precision, recall, auroc, auprc, precision_at_100, recall_at_100, precision_at_500, recall_at_500, precision_at_1000, recall_at_1000, precision_at_1pct, recall_at_1pct, runtime_seconds, command, git_commit, artifact_created_at_utc
- Regenerate/verify curated representation: gnn_env/bin/python scripts/tkde_visual_rebuild/build_curated_supplement_tables.py

### V22_PAIRED_TESTS

- Purpose: complete legacy V22 paired-test family
- Path: manuscript_assets/tables/V22_STAT_TESTS_FULL10.csv
- Data rows: 198
- SHA-256: 513f1fb5c2dbae2f63255fd384c62fe86394a15746c31792a92a1f287b8ddadd
- Columns in order: family, dataset, protocol, fixed_model, comparison_axis, left, right, metric, n, paired, mean_left, mean_right, mean_diff_left_minus_right, bootstrap_ci95_low, bootstrap_ci95_high, effect_size_standardized, p_value, p_value_bh
- Regenerate/verify curated representation: gnn_env/bin/python scripts/tkde_visual_rebuild/build_curated_supplement_tables.py

### IBM_SEED_GRID

- Purpose: complete IBM seed-level imported rows
- Path: results/tkde_rebuild/IBM_IMPORTED_SEED_ROWS.csv
- Data rows: 840
- SHA-256: dae5d51178b9a746946c9fc05bf31e74840b6663d69031597ef2be45d5e766b6
- Columns in order: version, variant, size, regime, protocol, config, seed, f1, precision, recall, balanced_accuracy, auroc, auprc, runtime_seconds, positive_rate, positive_count, test_count, actual_backend, source_path
- Regenerate/verify curated representation: gnn_env/bin/python scripts/tkde_visual_rebuild/build_curated_supplement_tables.py

### IBM_CONTEXT_EFFECTS

- Purpose: complete context-specific IBM sensitivity rows
- Path: results/tkde_rebuild/IBM_MATCHED_ABLATION_CONTEXT_EFFECTS.csv
- Data rows: 208
- SHA-256: 626f0e014a0b8d339fe344bb736d513ed63a6ec435be6104acad1f53f7bbe202
- Columns in order: config, size, metric, variant, protocol, n_seed_pairs, reference_mean, candidate_mean, mean_delta, delta_ci95_low, delta_ci95_high, cohen_dz, wilcoxon_p_descriptive, source_paths
- Regenerate/verify curated representation: gnn_env/bin/python scripts/tkde_visual_rebuild/build_curated_supplement_tables.py

## Allocation rule

The PDF is the human review interface: definitions, dataset/model/protocol cards, aggregate effects, uncertainty, correction status, feasibility exclusions, interpretation, and limitations. The artifact is the exhaustive interface: one row per seed, context, or legacy test, with full paths and provenance. Moving these rows is a publication-design change only and does not change a value, test, claim status, or evidence scope.
