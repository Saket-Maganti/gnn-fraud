# Baselines

Internal baselines include logistic regression, MLP, histogram gradient
boosting, random forest, GroupDRO, VREx, and IRM. Legacy GNN wrappers preserve
diagnostic status and cannot be promoted to official baselines.

Official adapter targets are pinned in
`external_baselines/BASELINE_REGISTRY.yaml`. Mowst, CIGA, TGN, and GOOD are
pending isolated installation/parity checks. GraphMETRO and EERM have no
verified reusable licence at the pinned revisions and are unavailable until
written permission or a licence-bearing revision is supplied. CIGA also has a
graph-classification task mismatch and cannot stand in for a node/event result.

No baseline is reported as official merely because a local approximation has a
similar name. Process-boundary adapters verify commit, entry point, task,
prediction schema, and checksums.
