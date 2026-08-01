"""Explicit FraudShiftBench protocol-to-contract-axis mappings."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FraudContractMapping:
    dataset: str
    protocol: str
    time: str
    visibility: str
    construction: str
    selection: str
    budget: str
    resource: str
    label_access: str = "NO_TARGET_LABELS_DURING_FITTING"


def canonical_fraud_contracts() -> tuple[FraudContractMapping, ...]:
    records = []
    for dataset in ("elliptic", "dgraphfin"):
        for protocol, visibility, construction in (
            ("strict_inductive", "historical_by_cutoff", "source_graph_only"),
            ("isolated_inductive", "target_nodes_no_edges", "no_target_edges"),
            ("transductive_structure", "all_structure_label_free", "full_visible_graph"),
        ):
            records.append(
                FraudContractMapping(
                    dataset=dataset,
                    protocol=protocol,
                    time="chronological_holdout",
                    visibility=visibility,
                    construction=construction,
                    selection="source_validation_only",
                    budget="declared_review_capacity",
                    resource="declared_feasible_expert_set",
                )
            )
    return tuple(records)
