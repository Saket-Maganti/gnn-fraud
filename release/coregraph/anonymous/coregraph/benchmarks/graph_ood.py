"""No-download registry and tiny stand-ins for established graph-OOD families."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GraphOODCandidate:
    family: str
    role: str
    scientific_relevance: str
    licence_status: str
    dataset_size: str
    disk_use: str
    gpu_memory: str
    runtime: str
    compatible_experts: tuple[str, ...]
    contract_axis_mapping: str
    official_splits: str
    target_label_policy: str
    implementation_complexity: str
    status: str


def candidate_registry() -> tuple[GraphOODCandidate, ...]:
    return (
        GraphOODCandidate(
            family="GOOD",
            role="PRIMARY",
            scientific_relevance="Recognized graph OOD covariate/concept environments",
            licence_status="GPL-3.0_VERIFIED_IN_LOCAL_PIN_AUDIT",
            dataset_size="UNKNOWN_UNTIL_OFFICIAL_ACQUISITION",
            disk_use="BLOCKED_RESOURCE_UNMEASURED",
            gpu_memory="BLOCKED_RESOURCE_UNMEASURED",
            runtime="BLOCKED_RESOURCE_UNMEASURED",
            compatible_experts=("feature_mlp", "gcn", "graphsage"),
            contract_axis_mapping="domain/shift -> time,visibility,construction",
            official_splits="OFFICIAL_AVAILABLE_NOT_INSTALLED",
            target_label_policy="NO_TARGET_LABELS_DURING_FITTING",
            implementation_complexity="MEDIUM",
            status="OFFICIAL_AVAILABLE_NOT_INSTALLED",
        ),
        GraphOODCandidate(
            family="OGB molecular scaffold/time split",
            role="FALLBACK",
            scientific_relevance="Non-fraud graph-level distribution shift",
            licence_status="BLOCKED_PENDING_LICENSE_REVIEW_PER_DATASET",
            dataset_size="UNKNOWN_UNTIL_DATASET_SELECTION",
            disk_use="BLOCKED_RESOURCE_UNMEASURED",
            gpu_memory="BLOCKED_RESOURCE_UNMEASURED",
            runtime="BLOCKED_RESOURCE_UNMEASURED",
            compatible_experts=("gin", "gine", "feature_mlp"),
            contract_axis_mapping="scaffold/time -> construction,time",
            official_splits="OFFICIAL_AVAILABLE_NOT_INSTALLED",
            target_label_policy="NO_TARGET_LABELS_DURING_FITTING",
            implementation_complexity="MEDIUM_HIGH",
            status="OFFICIAL_PENDING_LICENSE",
        ),
    )


def tiny_stand_in(family: str) -> dict[str, object]:
    if family not in {candidate.family for candidate in candidate_registry()}:
        raise KeyError(f"unknown graph-OOD family: {family}")
    return {
        "family": family,
        "synthetic_stand_in": True,
        "official_result_claim": "PROHIBITED",
        "target_labels_during_fitting": False,
        "status": "TINY_ADAPTER_FIXTURE_ONLY",
    }
