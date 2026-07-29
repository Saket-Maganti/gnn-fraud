"""Central leakage audits for temporal and contract-shift experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np

from coregraph.contracts.axes import AccessRegime
from coregraph.data.graph_views import GraphViewBundle, audit_view_bundle


class LeakageError(RuntimeError):
    """Raised before fitting whenever an information-flow invariant fails."""


@dataclass(frozen=True)
class FitAccessRecord:
    component: str
    split: str
    fields: tuple[str, ...]
    used_labels: bool
    used_target_ids: bool


def audit_split_masks(
    *,
    train: np.ndarray,
    validation: np.ndarray,
    target: np.ndarray,
    label_known: np.ndarray,
) -> tuple[str, ...]:
    masks = [np.asarray(value, dtype=bool) for value in (train, validation, target)]
    if len({len(mask) for mask in masks}) != 1 or len(label_known) != len(masks[0]):
        return ("mask_length_mismatch",)
    violations: list[str] = []
    if np.any(masks[0] & masks[1]):
        violations.append("train_validation_overlap")
    if np.any(masks[0] & masks[2]):
        violations.append("train_target_overlap")
    if np.any(masks[1] & masks[2]):
        violations.append("validation_target_overlap")
    if np.any((masks[0] | masks[1] | masks[2]) & ~np.asarray(label_known, dtype=bool)):
        violations.append("unknown_label_in_supervised_mask")
    return tuple(violations)


def audit_component_access(
    records: Iterable[FitAccessRecord],
    *,
    regime: AccessRegime,
) -> tuple[str, ...]:
    """Check scalers, thresholds, calibrators and routers uniformly."""

    violations: list[str] = []
    for record in records:
        if record.split == "target" and record.used_labels:
            if regime is not AccessRegime.FEW_LABEL_TARGET:
                violations.append(f"{record.component}:target_label_access")
        if record.split == "target" and record.used_target_ids:
            violations.append(f"{record.component}:target_identity_access")
        if record.component in {"scaler", "threshold", "calibrator"}:
            if record.split not in {"train", "validation"}:
                violations.append(f"{record.component}:fit_on_{record.split}")
        if record.component == "router" and record.used_labels and record.split == "target":
            violations.append("router:target_label_access")
    return tuple(sorted(set(violations)))


def audit_identifier_features(
    feature_names: Sequence[str],
    *,
    allowlisted_identifiers: Sequence[str] = (),
) -> tuple[str, ...]:
    allow = {name.lower() for name in allowlisted_identifiers}
    suspicious = ("id", "index", "row_number", "account_number", "transaction_id")
    return tuple(
        f"identifier_feature:{name}"
        for name in feature_names
        if name.lower() not in allow
        and (
            name.lower() in suspicious
            or name.lower().endswith("_id")
            or name.lower().startswith("id_")
        )
    )


def audit_temporal_experiment(
    *,
    graph_views: GraphViewBundle,
    test_node_ids: np.ndarray,
    train_cutoff: float,
    validation_cutoff: float,
    masks: Mapping[str, np.ndarray],
    label_known: np.ndarray,
    accesses: Iterable[FitAccessRecord],
    access_regime: AccessRegime,
) -> None:
    violations = list(
        audit_view_bundle(
            graph_views,
            test_node_ids=test_node_ids,
            train_cutoff=train_cutoff,
            validation_cutoff=validation_cutoff,
        )
    )
    violations.extend(
        audit_split_masks(
            train=masks["train"],
            validation=masks["validation"],
            target=masks["target"],
            label_known=label_known,
        )
    )
    violations.extend(audit_component_access(accesses, regime=access_regime))
    if violations:
        raise LeakageError(";".join(sorted(set(violations))))
