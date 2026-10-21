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


@dataclass(frozen=True)
class ScenarioPredictionScope:
    """Selected rows for one base artifact inside one evaluation scenario."""

    scenario_id: str
    dataset: str
    base_artifact_hash: str
    expert_id: str
    base_protocol_id: str
    bound_protocol_id: str
    expert_prediction_seed: int
    fold: str
    role: str
    contract_coordinate_hash: str
    path: str
    checksum: str
    selected_identifiers: tuple[str, ...]
    selected_splits: tuple[str, ...]
    selected_label_known: tuple[bool, ...]
    selected_timestamps: tuple[float | None, ...] = ()
    target_labels_used_for_fitting: bool = False
    selection_metadata_fields: tuple[str, ...] = ()
    registry_consistent: bool = True

    def __post_init__(self) -> None:
        if self.role not in {"source", "target"}:
            raise ValueError("scenario prediction scope role must be source or target")
        if len(self.selected_identifiers) != len(self.selected_splits):
            raise ValueError("scenario scope identifiers and splits must align")
        if len(self.selected_label_known) != len(self.selected_identifiers):
            raise ValueError("scenario scope label-known rows must align")
        if self.selected_timestamps and len(self.selected_timestamps) != len(
            self.selected_identifiers
        ):
            raise ValueError("scenario scope timestamps must align")


@dataclass(frozen=True)
class ScenarioLeakageReport:
    scenario_id: str
    dataset: str
    target_protocol_id: str
    source_protocol_ids: tuple[str, ...]
    expert_prediction_seed: int
    fold: str
    findings: tuple[LeakageFinding, ...]

    @property
    def passed(self) -> bool:
        return not any(finding.severity == "ATOMIC" for finding in self.findings)

    def to_dict(self) -> dict[str, object]:
        return {
            "scenario_id": self.scenario_id,
            "dataset": self.dataset,
            "target_protocol_id": self.target_protocol_id,
            "source_protocol_ids": list(self.source_protocol_ids),
            "expert_prediction_seed": self.expert_prediction_seed,
            "fold": self.fold,
            "passed": self.passed,
            "findings": [asdict(finding) for finding in self.findings],
        }


def audit_evaluation_scenario_scopes(
    scopes: Sequence[ScenarioPredictionScope],
    *,
    scenario_id: str,
    dataset: str,
    target_protocol_id: str,
    source_protocol_ids: Sequence[str],
    expert_prediction_seed: int,
    fold: str,
) -> ScenarioLeakageReport:
    """Audit role bindings only within one held-out-protocol scenario.

    Reusing the same immutable artifact or checksum in a different scenario is
    intentionally outside this function's scope and is not leakage.
    """

    findings: list[LeakageFinding] = []
    source_protocols = tuple(sorted(set(source_protocol_ids)))
    if target_protocol_id in source_protocols:
        findings.append(
            LeakageFinding(
                "TARGET_PROTOCOL_IN_SOURCE_SET",
                "ATOMIC",
                f"target protocol {target_protocol_id!r} also appears as a source",
            )
        )
    if any(scope.scenario_id != scenario_id for scope in scopes):
        findings.append(
            LeakageFinding(
                "CROSS_SCENARIO_SCOPE_MIX",
                "ATOMIC",
                "the audit received bindings from more than one scenario",
            )
        )
    for scope in scopes:
        if (
            scope.dataset != dataset
            or scope.expert_prediction_seed != expert_prediction_seed
            or scope.fold != fold
        ):
            findings.append(
                LeakageFinding(
                    "SCENARIO_IDENTITY_MISMATCH",
                    "ATOMIC",
                    "binding dataset, seed, or fold disagrees with its scenario",
                )
            )
        if scope.base_protocol_id != scope.bound_protocol_id:
            findings.append(
                LeakageFinding(
                    "ARTIFACT_PROTOCOL_REBINDING",
                    "ATOMIC",
                    f"base protocol {scope.base_protocol_id!r} was rebound as "
                    f"{scope.bound_protocol_id!r}",
                )
            )
        if not scope.registry_consistent:
            findings.append(
                LeakageFinding(
                    "PROTOCOL_REGISTRY_CONFLICT",
                    "ATOMIC",
                    f"binding for {scope.base_protocol_id!r} conflicts with the registry",
                )
            )
        if len(set(scope.selected_identifiers)) != len(scope.selected_identifiers):
            findings.append(
                LeakageFinding(
                    "DUPLICATED_PREDICTION_ROWS",
                    "ATOMIC",
                    f"duplicate selected identifiers in {scope.path}",
                )
            )
        if scope.role == "source":
            forbidden_splits = sorted(
                set(scope.selected_splits) - {"train", "validation"}
            )
            if forbidden_splits:
                findings.append(
                    LeakageFinding(
                        "TEST_ROWS_ENTER_SOURCE_SCOPE",
                        "ATOMIC",
                        f"source binding selected forbidden splits {forbidden_splits}",
                    )
                )
            if not all(scope.selected_label_known):
                findings.append(
                    LeakageFinding(
                        "UNKNOWN_LABEL_ENTERS_SOURCE_FITTING",
                        "ATOMIC",
                        "source fitting/selection contains provider-unknown labels",
                    )
                )
            forbidden_metadata = {
                field.lower() for field in scope.selection_metadata_fields
            } & {"target_label", "target_labels", "y_true", "label"}
            if scope.target_labels_used_for_fitting or forbidden_metadata:
                findings.append(
                    LeakageFinding(
                        "TARGET_LABEL_IN_SOURCE_SELECTION",
                        "ATOMIC",
                        "target labels are reachable from source fitting or selection",
                    )
                )
        else:
            if set(scope.selected_splits) - {"test"}:
                findings.append(
                    LeakageFinding(
                        "TRAIN_VALIDATION_ROWS_ENTER_TARGET_SCORING",
                        "ATOMIC",
                        "target scoring selected train or validation rows",
                    )
                )
            if not all(scope.selected_label_known):
                findings.append(
                    LeakageFinding(
                        "UNKNOWN_PROVIDER_LABEL_ENTERS_SCORING",
                        "ATOMIC",
                        "target scoring selected a provider-unknown label",
                    )
                )
            if scope.bound_protocol_id != target_protocol_id:
                findings.append(
                    LeakageFinding(
                        "WRONG_TARGET_PROTOCOL_BINDING",
                        "ATOMIC",
                        f"target binding uses {scope.bound_protocol_id!r}, expected "
                        f"{target_protocol_id!r}",
                    )
                )

    roles_by_artifact: dict[str, set[str]] = {}
    for scope in scopes:
        roles_by_artifact.setdefault(scope.base_artifact_hash, set()).add(scope.role)
    for base_hash, roles in sorted(roles_by_artifact.items()):
        if roles == {"source", "target"}:
            findings.append(
                LeakageFinding(
                    "SAME_ARTIFACT_BOUND_TO_BOTH_ROLES",
                    "ATOMIC",
                    f"base artifact {base_hash} has both roles in one scenario",
                )
            )

    sources = [scope for scope in scopes if scope.role == "source"]
    targets = [scope for scope in scopes if scope.role == "target"]
    for source in sources:
        source_ids = set(source.selected_identifiers)
        source_times = [
            float(value)
            for value in source.selected_timestamps
            if value is not None
        ]
        for target in targets:
            overlap = sorted(source_ids & set(target.selected_identifiers))
            if overlap:
                findings.append(
                    LeakageFinding(
                        "SOURCE_TARGET_SPLIT_ID_OVERLAP",
                        "ATOMIC",
                        f"{len(overlap)} source train/validation IDs overlap target "
                        f"test IDs; sample={overlap[:5]}",
                    )
                )
            if source.contract_coordinate_hash == target.contract_coordinate_hash:
                findings.append(
                    LeakageFinding(
                        "HELD_OUT_COORDINATE_EQUIVALENCE",
                        "ATOMIC",
                        "source and target scientific coordinates are equal",
                    )
                )
            target_times = [
                float(value)
                for value in target.selected_timestamps
                if value is not None
            ]
            if source_times and target_times and max(source_times) >= min(target_times):
                findings.append(
                    LeakageFinding(
                        "TIMESTAMP_ORDER_VIOLATION",
                        "ATOMIC",
                        "source train/validation time is not strictly before target test",
                    )
                )

    deduplicated = {
        (finding.code, finding.severity, finding.detail): finding
        for finding in findings
    }
    return ScenarioLeakageReport(
        scenario_id=scenario_id,
        dataset=dataset,
        target_protocol_id=target_protocol_id,
        source_protocol_ids=source_protocols,
        expert_prediction_seed=expert_prediction_seed,
        fold=fold,
        findings=tuple(deduplicated[key] for key in sorted(deduplicated)),
    )


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
    """Audit a legacy V4 scope set treated as one logical scenario.

    V5 callers must use :func:`audit_evaluation_scenario_scopes`, which binds
    roles inside an explicit scenario.  File/checksum reuse is deliberately
    not a global violation: one immutable base artifact may legitimately be a
    source in one scenario and a target in another.
    """

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
