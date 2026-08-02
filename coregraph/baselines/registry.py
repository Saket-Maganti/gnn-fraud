"""Level-4 baseline registry with exact non-promotional statuses."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class BaselineStatus(str, Enum):
    IMPLEMENTED_INTERNAL = "IMPLEMENTED_INTERNAL"
    OFFICIAL_AVAILABLE_NOT_INSTALLED = "OFFICIAL_AVAILABLE_NOT_INSTALLED"
    OFFICIAL_INTEGRATED = "OFFICIAL_INTEGRATED"
    OFFICIAL_PENDING_LICENSE = "OFFICIAL_PENDING_LICENSE"
    UNAVAILABLE_LICENSE = "UNAVAILABLE_LICENSE"
    UNAVAILABLE_DEPENDENCY = "UNAVAILABLE_DEPENDENCY"
    RESOURCE_BLOCKED = "RESOURCE_BLOCKED"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    INTERNAL_APPROXIMATION_ONLY = "INTERNAL_APPROXIMATION_ONLY"


@dataclass(frozen=True)
class BaselineRecord:
    baseline_id: str
    category: str
    status: BaselineStatus
    deployable: bool
    target_label_access: str
    official_repository: str
    official_commit: str
    licence: str
    protocol_validity: str
    blocker: str


def level4_baselines() -> tuple[BaselineRecord, ...]:
    simple = (
        "best_fixed_expert",
        "uniform_average",
        "validation_weighted_average",
        "confidence_selection",
        "entropy_selection",
        "margin_selection",
        "source_logistic_gate",
        "source_mlp_gate",
        "resource_aware_heuristic",
        "cheapest_feasible_expert",
        "random_feasible_expert",
    )
    records = [
        BaselineRecord(
            name,
            "simple",
            BaselineStatus.IMPLEMENTED_INTERNAL,
            True,
            "SOURCE_VALIDATION_ONLY",
            "NOT_APPLICABLE",
            "NOT_APPLICABLE",
            "REPOSITORY_LICENSE",
            "VALID_WHEN_COMMON_ACCESS_CONTRACT_IS_ENFORCED",
            "NONE",
        )
        for name in simple
    ]
    records.extend(
        [
            BaselineRecord(
                "contract_oracle_diagnostic",
                "oracle_diagnostic",
                BaselineStatus.IMPLEMENTED_INTERNAL,
                False,
                "TARGET_LABELS_OFFLINE_EVALUATION_ONLY",
                "NOT_APPLICABLE",
                "NOT_APPLICABLE",
                "REPOSITORY_LICENSE",
                "DIAGNOSTIC_ONLY",
                "NON_DEPLOYABLE_ORACLE",
            ),
            BaselineRecord(
                "instance_oracle_diagnostic",
                "oracle_diagnostic",
                BaselineStatus.IMPLEMENTED_INTERNAL,
                False,
                "TARGET_LABELS_OFFLINE_EVALUATION_ONLY",
                "NOT_APPLICABLE",
                "NOT_APPLICABLE",
                "REPOSITORY_LICENSE",
                "DIAGNOSTIC_ONLY",
                "NON_DEPLOYABLE_CLAIRVOYANT_ORACLE",
            ),
        ]
    )
    external = (
        ("mowst", "graph_moe", BaselineStatus.OFFICIAL_AVAILABLE_NOT_INSTALLED, "https://github.com/facebookresearch/mowst-gnn", "2e3569962d2388bfda4535cdd1fc0b6eaec88a28", "MIT", "TASK_BRIDGE_AND_PARITY_PENDING"),
        ("graphmetro", "graph_ood", BaselineStatus.UNAVAILABLE_LICENSE, "https://github.com/Wuyxin/GraphMETRO", "e2b6ab62c6d7a3d72b6508db9bfce49336a9b129", "NO_LICENCE_FILE_FOUND_AT_LOCAL_AUDIT", "WRITTEN_PERMISSION_OR_LICENSED_REVISION_REQUIRED"),
        ("ciga", "graph_ood", BaselineStatus.OFFICIAL_AVAILABLE_NOT_INSTALLED, "https://github.com/LFhase/CIGA", "454801108737ff8855ac2be947201dd9338dff37", "MIT", "GRAPH_CLASSIFICATION_ONLY_AT_PIN"),
        ("eerm", "graph_ood", BaselineStatus.UNAVAILABLE_LICENSE, "https://github.com/qitianwu/GraphOOD-EERM", "ffdc4a11161976fac7dd71e2aa1dcd72db6e44e9", "NO_LICENCE_FILE_FOUND_AT_LOCAL_AUDIT", "WRITTEN_PERMISSION_OR_LICENSED_REVISION_REQUIRED"),
        ("groupdro", "robust", BaselineStatus.IMPLEMENTED_INTERNAL, "NOT_APPLICABLE", "NOT_APPLICABLE", "REPOSITORY_LICENSE", "INTERNAL_SOURCE_ENVIRONMENT_OBJECTIVE"),
        ("vrex", "robust", BaselineStatus.IMPLEMENTED_INTERNAL, "NOT_APPLICABLE", "NOT_APPLICABLE", "REPOSITORY_LICENSE", "INTERNAL_SOURCE_ENVIRONMENT_OBJECTIVE"),
        ("good", "benchmark", BaselineStatus.OFFICIAL_AVAILABLE_NOT_INSTALLED, "https://github.com/divelab/GOOD", "b53566c9297bc65b90a7f2213fb9ffa930f5b6e5", "GPL-3.0", "OFFICIAL_CHECKOUT_AND_ADAPTER_PARITY_PENDING"),
    )
    records.extend(
        BaselineRecord(name, category, status, True, "NO_TARGET_LABELS_DURING_FITTING", repo, commit, licence, "DECLARED_PER_ADAPTER", blocker)
        for name, category, status, repo, commit, licence, blocker in external
    )
    return tuple(records)
