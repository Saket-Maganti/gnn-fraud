"""Central leakage audits for temporal and contract-shift experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
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


@dataclass(frozen=True)
class PredictionArtifactScope:
    """Read-only row/identity evidence passed to cross-role leakage audits."""

    dataset: str
    expert_id: str
    protocol_id: str
    expert_prediction_seed: int
    fold: str
    role: str
    contract_coordinate_hash: str
    contract_id: str
    path: str
    checksum: str
    original_checksum: str
    identifiers: tuple[str, ...]
    splits: tuple[str, ...]
    label_known: tuple[bool, ...]
    timestamps: tuple[float | None, ...] = ()
    selection_metadata_fields: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.role not in {"source", "target"}:
            raise ValueError("prediction artifact scope role must be source or target")
        if len(self.identifiers) != len(self.splits):
            raise ValueError("prediction artifact scope IDs and splits must align")
        if len(self.label_known) != len(self.identifiers):
            raise ValueError("prediction artifact scope label-known rows must align")
        if self.timestamps and len(self.timestamps) != len(self.identifiers):
            raise ValueError("prediction artifact scope timestamps must align")


@dataclass(frozen=True)
class LeakageFinding:
    code: str
    severity: str
    detail: str


@dataclass(frozen=True)
class PredictionLeakageReport:
    dataset: str
    target_protocol_id: str
    expert_prediction_seed: int
    fold: str
    target_contract_coordinate_hash: str
    target_contract_id: str
    findings: tuple[LeakageFinding, ...]

    @property
    def passed(self) -> bool:
        return not any(finding.severity == "ATOMIC" for finding in self.findings)

    def to_dict(self) -> dict[str, object]:
        return {
            "dataset": self.dataset,
            "target_protocol_id": self.target_protocol_id,
            "expert_prediction_seed": self.expert_prediction_seed,
            "fold": self.fold,
            "target_contract_coordinate_hash": (
                self.target_contract_coordinate_hash
            ),
            "target_contract_id": self.target_contract_id,
            "passed": self.passed,
            "findings": [asdict(finding) for finding in self.findings],
        }


def _scoped_ids(
    scope: PredictionArtifactScope,
    allowed_splits: set[str],
) -> set[str]:
    return {
        identifier
        for identifier, split in zip(
            scope.identifiers,
            scope.splits,
            strict=True,
        )
        if split in allowed_splits
    }


def _known_times(
    scope: PredictionArtifactScope,
    allowed_splits: set[str],
) -> list[float]:
    if not scope.timestamps:
        return []
    return [
        float(timestamp)
        for timestamp, split in zip(
            scope.timestamps,
            scope.splits,
            strict=True,
        )
        if timestamp is not None and split in allowed_splits
    ]


def audit_cross_role_prediction_scopes(
    scopes: Sequence[PredictionArtifactScope],
    *,
    held_out_contract_required: bool = True,
) -> tuple[PredictionLeakageReport, ...]:
    """Audit every target against same-dataset/seed/fold source artifacts."""

    reports: list[PredictionLeakageReport] = []
    target_groups: dict[
        tuple[str, str, int, str, str, str],
        list[PredictionArtifactScope],
    ] = {}
    for scope in scopes:
        if scope.role == "target":
            target_groups.setdefault(
                (
                    scope.dataset,
                    scope.protocol_id,
                    scope.expert_prediction_seed,
                    scope.fold,
                    scope.contract_coordinate_hash,
                    scope.contract_id,
                ),
                [],
            ).append(scope)
    contract_metadata: dict[str, set[tuple[str, str, str]]] = {}
    coordinate_aliases: dict[str, set[str]] = {}
    for scope in scopes:
        contract_metadata.setdefault(scope.contract_id, set()).add(
            (
                scope.role,
                scope.contract_coordinate_hash,
                scope.protocol_id,
            )
        )
        coordinate_aliases.setdefault(
            scope.contract_coordinate_hash,
            set(),
        ).add(scope.protocol_id)
    for target_key, targets in sorted(target_groups.items()):
        (
            dataset,
            protocol_id,
            expert_seed,
            fold,
            target_coordinate_hash,
            target_contract_id,
        ) = target_key
        sources = [
            scope
            for scope in scopes
            if scope.role == "source"
            and scope.dataset == dataset
            and scope.expert_prediction_seed == expert_seed
            and scope.fold == fold
        ]
        findings: list[LeakageFinding] = []
        if not sources:
            findings.append(
                LeakageFinding(
                    "MISSING_SOURCE_SCOPE",
                    "ATOMIC",
                    "target has no same-dataset/seed/fold source artifact",
                )
            )
        for target in targets:
            target_test_ids = _scoped_ids(target, {"test"})
            if len(set(target.identifiers)) != len(target.identifiers):
                findings.append(
                    LeakageFinding(
                        "DUPLICATED_PREDICTION_ROWS",
                        "ATOMIC",
                        f"duplicate identifiers in target file {target.path}",
                    )
                )
            for source in sources:
                source_ids = _scoped_ids(source, {"train", "validation"})
                overlap = sorted(source_ids & target_test_ids)
                if overlap:
                    detail = (
                        f"{len(overlap)} source train/validation identifiers "
                        f"overlap target test rows; sample={overlap[:5]}"
                    )
                    findings.extend(
                        (
                            LeakageFinding(
                                "ATOMIC_ID_OVERLAP",
                                "ATOMIC",
                                detail,
                            ),
                            LeakageFinding(
                                "SOURCE_TARGET_SPLIT_ID_OVERLAP",
                                "ATOMIC",
                                detail,
                            ),
                        )
                    )
                if source.path == target.path:
                    findings.append(
                        LeakageFinding(
                            "REUSED_FILE_CONFLICTING_ROLES",
                            "ATOMIC",
                            f"file reused as source and target: {source.path}",
                        )
                    )
                incompatible = (
                    source.role != target.role
                    or source.contract_id != target.contract_id
                    or source.protocol_id != target.protocol_id
                )
                if source.checksum == target.checksum and incompatible:
                    findings.append(
                        LeakageFinding(
                            "REUSED_CHECKSUM_INCOMPATIBLE_METADATA",
                            "ATOMIC",
                            "identical prediction bytes have incompatible "
                            "role/contract metadata",
                        )
                    )
                if (
                    source.original_checksum
                    and source.original_checksum == target.original_checksum
                    and incompatible
                ):
                    findings.append(
                        LeakageFinding(
                            "REUSED_ORIGINAL_CHECKSUM_INCOMPATIBLE_METADATA",
                            "ATOMIC",
                            "converted artifacts share original bytes under "
                            "incompatible role/contract metadata",
                        )
                    )
                if (
                    held_out_contract_required
                    and source.contract_coordinate_hash
                    == target.contract_coordinate_hash
                ):
                    findings.append(
                        LeakageFinding(
                            "HELD_OUT_COORDINATE_EQUIVALENCE",
                            "ATOMIC",
                            "source and target share the held-out scientific "
                            "coordinate",
                        )
                    )
                source_times = _known_times(source, {"train", "validation"})
                target_times = _known_times(target, {"test"})
                if (
                    source_times
                    and target_times
                    and max(source_times) >= min(target_times)
                ):
                    findings.append(
                        LeakageFinding(
                            "TIMESTAMP_ORDER_VIOLATION",
                            "ATOMIC",
                            "source train/validation time is not strictly "
                            "before target test",
                        )
                    )
        for source in sources:
            if len(set(source.identifiers)) != len(source.identifiers):
                findings.append(
                    LeakageFinding(
                        "DUPLICATED_PREDICTION_ROWS",
                        "ATOMIC",
                        f"duplicate identifiers in source file {source.path}",
                    )
                )
            forbidden_metadata = {
                field.lower()
                for field in source.selection_metadata_fields
            } & {"target_label", "target_labels", "y_true", "label"}
            if forbidden_metadata:
                findings.append(
                    LeakageFinding(
                        "TARGET_LABEL_IN_SOURCE_SELECTION_METADATA",
                        "ATOMIC",
                        f"forbidden metadata fields: {sorted(forbidden_metadata)}",
                    )
                )
        if len(contract_metadata.get(target_contract_id, ())) != 1:
            findings.append(
                LeakageFinding(
                    "CONTRACT_ROLE_HASH_DISAGREEMENT",
                    "ATOMIC",
                    "target contract ID is reused under incompatible role, "
                    "coordinate or protocol metadata",
                )
            )
        if len(coordinate_aliases.get(target_coordinate_hash, ())) != 1:
            findings.append(
                LeakageFinding(
                    "COORDINATE_ALIAS_COLLISION",
                    "ATOMIC",
                    "target coordinate hash maps to incompatible aliases",
                )
            )
        deduplicated = {
            (finding.code, finding.severity, finding.detail): finding
            for finding in findings
        }
        reports.append(
            PredictionLeakageReport(
                dataset=dataset,
                target_protocol_id=protocol_id,
                expert_prediction_seed=expert_seed,
                fold=fold,
                target_contract_coordinate_hash=target_coordinate_hash,
                target_contract_id=target_contract_id,
                findings=tuple(
                    deduplicated[key] for key in sorted(deduplicated)
                ),
            )
        )
    return tuple(reports)


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
