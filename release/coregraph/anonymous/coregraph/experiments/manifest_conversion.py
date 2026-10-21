"""Read-only discovery and evidence-bound V4 prediction-manifest conversion."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from coregraph.contracts.contract import DeploymentContract
from coregraph.experiments.protocol_registry import validate_protocol_bindings

REQUIRED_DATASETS = ("elliptic", "dgraphfin")
REQUIRED_PROTOCOLS = (
    "strict_inductive",
    "isolated_inductive",
    "transductive_structure",
)
REQUIRED_EXPERTS = ("feature_mlp", "gcn", "graphsage")
REQUIRED_SEEDS = tuple(range(1, 11))
REQUIRED_ROLES = ("source", "target")
REQUIRED_COLUMNS = {
    "dataset",
    "protocol",
    "model",
    "seed",
    "split",
    "node_id",
    "y_true",
    "score",
    "label_known",
}
PROTOCOL_ALIASES = {
    "strict_inductive": "strict_inductive",
    "inductive_isolated": "isolated_inductive",
    "isolated_inductive": "isolated_inductive",
    "transductive": "transductive_structure",
    "transductive_structure": "transductive_structure",
}
EXPERT_ALIASES = {
    "mlp": "feature_mlp",
    "feature_mlp": "feature_mlp",
    "gcn": "gcn",
    "sage": "graphsage",
    "graphsage": "graphsage",
}
LABEL_MAPPING = {"0": "unknown", "1": "fraud", "2": "normal"}
LABEL_MAPPING_EVIDENCE = {
    "elliptic": "coregraph/data/elliptic_v2.py",
    "dgraphfin": "coregraph/data/dgraphfin_v2.py",
}
PROVIDER_SPLIT_MAPPING = {
    "train": "train",
    "val": "validation",
    "validation": "validation",
    "test": "test",
    "unscored": "unscored",
}
PROVIDER_SPLIT_MAPPING_EVIDENCE = {
    "elliptic": (
        "scripts/validate_prediction_exports.py and "
        "scripts/run_tpc_tta_eval.py"
    ),
    "dgraphfin": (
        "scripts/validate_prediction_exports.py and "
        "scripts/run_tpc_tta_eval.py"
    ),
}
_FILENAME = re.compile(
    r"^(?P<dataset>elliptic|dgraphfin)__"
    r"(?P<protocol>[a-z0-9_]+)__"
    r"(?P<expert>mlp|feature_mlp|gcn|sage|graphsage)__"
    r"seed(?P<seed>[1-9]|10)\.csv$"
)


@dataclass(frozen=True)
class HistoricalPredictionCandidate:
    path: Path
    dataset: str
    source_protocol: str
    protocol_id: str
    source_expert: str
    expert_id: str
    expert_prediction_seed: int
    fold: str = "fold0"

    @property
    def logical_key(self) -> tuple[str, str, str, int, str]:
        return (
            self.dataset,
            self.protocol_id,
            self.expert_id,
            self.expert_prediction_seed,
            self.fold,
        )


@dataclass(frozen=True)
class ValidationEvidence:
    report_path: str
    reported_path: str
    expected_checksum: str
    status: str


@dataclass(frozen=True)
class CandidateAudit:
    original_path: str
    original_checksum: str
    dataset: str
    source_protocol: str
    protocol_id: str
    source_expert: str
    expert_id: str
    expert_prediction_seed: int
    fold: str
    row_count: int
    split_counts: Mapping[str, int]
    label_known_counts: Mapping[str, int]
    excluded_unknown_label_counts: Mapping[str, int]
    timestamp_ranges: Mapping[str, Sequence[float]]
    duplicate_identifier_count: int
    validation_evidence: tuple[ValidationEvidence, ...]
    schema_errors: tuple[str, ...]
    content_errors: tuple[str, ...]

    @property
    def validated_export(self) -> bool:
        return bool(self.validation_evidence) and not self.schema_errors

    @property
    def structurally_usable(self) -> bool:
        return (
            self.validated_export
            and not self.content_errors
            and self.duplicate_identifier_count == 0
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["validation_evidence"] = [
            asdict(value) for value in self.validation_evidence
        ]
        payload["provider_split_mapping"] = dict(PROVIDER_SPLIT_MAPPING)
        payload["provider_split_mapping_evidence"] = (
            PROVIDER_SPLIT_MAPPING_EVIDENCE[self.dataset]
        )
        payload["validated_export"] = self.validated_export
        payload["structurally_usable"] = self.structurally_usable
        return payload


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover_historical_predictions(
    roots: Sequence[str | Path],
) -> tuple[HistoricalPredictionCandidate, ...]:
    """Discover only requested dataset/protocol/expert/seed CSV candidates."""

    candidates: list[HistoricalPredictionCandidate] = []
    seen: set[Path] = set()
    for root_value in roots:
        root = Path(root_value).expanduser().resolve()
        paths = (root,) if root.is_file() else root.rglob("*.csv")
        for path in paths:
            match = _FILENAME.fullmatch(path.name)
            if match is None:
                continue
            source_protocol = match.group("protocol")
            if source_protocol not in PROTOCOL_ALIASES:
                continue
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            candidates.append(
                HistoricalPredictionCandidate(
                    path=resolved,
                    dataset=match.group("dataset"),
                    source_protocol=source_protocol,
                    protocol_id=PROTOCOL_ALIASES[source_protocol],
                    source_expert=match.group("expert"),
                    expert_id=EXPERT_ALIASES[match.group("expert")],
                    expert_prediction_seed=int(match.group("seed")),
                )
            )
    return tuple(
        sorted(
            candidates,
            key=lambda value: (*value.logical_key, str(value.path)),
        )
    )


def discover_validation_evidence(
    roots: Sequence[str | Path],
) -> tuple[tuple[Path, Mapping[str, Any]], ...]:
    reports: list[tuple[Path, Mapping[str, Any]]] = []
    seen: set[Path] = set()
    for root_value in roots:
        root = Path(root_value).expanduser().resolve()
        if not root.exists():
            continue
        for path in root.rglob("prediction_validation_report.json"):
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(payload, Mapping):
                reports.append((resolved, payload))
    return tuple(sorted(reports, key=lambda value: str(value[0])))


def _path_matches(candidate: Path, reported_path: str) -> bool:
    normalized = reported_path.replace("\\", "/").lstrip("./")
    return candidate.as_posix().endswith(normalized)


def validation_evidence_for(
    candidate: HistoricalPredictionCandidate,
    checksum: str,
    reports: Sequence[tuple[Path, Mapping[str, Any]]],
) -> tuple[ValidationEvidence, ...]:
    evidence: list[ValidationEvidence] = []
    for report_path, payload in reports:
        files = payload.get("files")
        if isinstance(files, list):
            for item in files:
                if not isinstance(item, Mapping):
                    continue
                reported = str(item.get("path", ""))
                if not reported or not _path_matches(candidate.path, reported):
                    continue
                expected = str(item.get("sha256", ""))
                ok = item.get("ok") is True
                status = (
                    "VALIDATED_CHECKSUM_MATCH"
                    if ok and expected == checksum
                    else "VALIDATION_EVIDENCE_CONFLICT"
                )
                evidence.append(
                    ValidationEvidence(
                        str(report_path),
                        reported,
                        expected,
                        status,
                    )
                )
        inputs = payload.get("inputs")
        if (
            payload.get("ok") is True
            and isinstance(inputs, list)
            and not payload.get("issues")
        ):
            for reported_value in inputs:
                reported = str(reported_value)
                if _path_matches(candidate.path, reported):
                    evidence.append(
                        ValidationEvidence(
                            str(report_path),
                            reported,
                            "",
                            "VALIDATED_REPORT_PATH_MATCH_CHECKSUM_RECOMPUTED",
                        )
                    )
    unique = {
        (
            item.report_path,
            item.reported_path,
            item.expected_checksum,
            item.status,
        ): item
        for item in evidence
    }
    return tuple(unique[key] for key in sorted(unique))


def _parse_known(value: object) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    raise ValueError(f"invalid label_known value {value!r}")


def audit_historical_candidate(
    candidate: HistoricalPredictionCandidate,
    reports: Sequence[tuple[Path, Mapping[str, Any]]],
) -> CandidateAudit:
    """Audit rows, labels and splits without computing target metrics."""

    checksum = sha256_path(candidate.path)
    split_counts: dict[str, int] = {}
    known_counts = {"known": 0, "unknown": 0}
    unknown_by_split: dict[str, int] = {}
    timestamp_values: dict[str, list[float]] = {}
    schema_errors: list[str] = []
    content_errors: list[str] = []
    identifiers: set[str] = set()
    duplicates = 0
    row_count = 0
    with candidate.path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or ())
        missing = sorted(REQUIRED_COLUMNS - fieldnames)
        if missing:
            schema_errors.append(f"missing_columns:{','.join(missing)}")
        time_column = (
            "timestamp"
            if "timestamp" in fieldnames
            else "timestep"
            if "timestep" in fieldnames
            else ""
        )
        for row in reader:
            row_count += 1
            identifier = str(row.get("node_id", ""))
            if identifier in identifiers:
                duplicates += 1
            identifiers.add(identifier)
            split = str(row.get("split", ""))
            split_counts[split] = split_counts.get(split, 0) + 1
            try:
                known = _parse_known(row.get("label_known"))
            except ValueError as error:
                content_errors.append(str(error))
                known = False
            known_counts["known" if known else "unknown"] += 1
            if not known:
                unknown_by_split[split] = unknown_by_split.get(split, 0) + 1
            expected = {
                "dataset": candidate.dataset,
                "protocol": candidate.source_protocol,
                "model": candidate.source_expert,
                "seed": str(candidate.expert_prediction_seed),
            }
            for field, value in expected.items():
                if str(row.get(field, "")) != value:
                    content_errors.append(
                        f"row_{row_count}_{field}_mismatch"
                    )
            try:
                label = int(row.get("y_true", ""))
                if str(label) not in LABEL_MAPPING:
                    content_errors.append(
                        f"row_{row_count}_label_outside_provider_mapping"
                    )
                if not known and label != 0:
                    content_errors.append(
                        f"row_{row_count}_unknown_label_not_provider_unknown"
                    )
                if known and label == 0:
                    content_errors.append(
                        f"row_{row_count}_provider_unknown_declared_known"
                    )
                score = float(row.get("score", ""))
                if not np.isfinite(score) or not 0 <= score <= 1:
                    content_errors.append(
                        f"row_{row_count}_score_not_probability"
                    )
            except (TypeError, ValueError):
                content_errors.append(f"row_{row_count}_invalid_label_or_score")
            if time_column:
                try:
                    timestamp = float(row[time_column])
                    if not np.isfinite(timestamp):
                        raise ValueError
                    timestamp_values.setdefault(split, []).append(timestamp)
                except (TypeError, ValueError):
                    content_errors.append(
                        f"row_{row_count}_invalid_{time_column}"
                    )
    if not row_count:
        content_errors.append("empty_prediction_file")
    allowed_splits = set(PROVIDER_SPLIT_MAPPING)
    unexpected_splits = sorted(set(split_counts) - allowed_splits)
    if unexpected_splits:
        content_errors.append(
            f"unexpected_splits:{','.join(unexpected_splits)}"
        )
    timestamp_ranges = {
        split: (float(min(values)), float(max(values)))
        for split, values in sorted(timestamp_values.items())
        if values
    }
    evidence = validation_evidence_for(candidate, checksum, reports)
    if evidence and any(
        item.status == "VALIDATION_EVIDENCE_CONFLICT" for item in evidence
    ):
        content_errors.append("validation_evidence_conflict")
    return CandidateAudit(
        original_path=str(candidate.path),
        original_checksum=checksum,
        dataset=candidate.dataset,
        source_protocol=candidate.source_protocol,
        protocol_id=candidate.protocol_id,
        source_expert=candidate.source_expert,
        expert_id=candidate.expert_id,
        expert_prediction_seed=candidate.expert_prediction_seed,
        fold=candidate.fold,
        row_count=row_count,
        split_counts=dict(sorted(split_counts.items())),
        label_known_counts=known_counts,
        excluded_unknown_label_counts=dict(sorted(unknown_by_split.items())),
        timestamp_ranges=timestamp_ranges,
        duplicate_identifier_count=duplicates,
        validation_evidence=evidence,
        schema_errors=tuple(sorted(set(schema_errors))),
        content_errors=tuple(sorted(set(content_errors))),
    )


def audit_candidates(
    candidates: Sequence[HistoricalPredictionCandidate],
    reports: Sequence[tuple[Path, Mapping[str, Any]]],
) -> tuple[CandidateAudit, ...]:
    return tuple(audit_historical_candidate(value, reports) for value in candidates)


def _metadata_entry_by_path(
    evidence_map: Mapping[str, Any] | None,
) -> dict[str, Mapping[str, Any]]:
    if evidence_map is None:
        return {}
    if evidence_map.get("schema_version") != (
        "coregraph_manifest_conversion_evidence_v4"
    ):
        raise ValueError("conversion evidence map must use the V4 schema")
    entries = evidence_map.get("artifacts")
    if not isinstance(entries, list):
        raise ValueError("conversion evidence map artifacts must be a list")
    output: dict[str, Mapping[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ValueError("conversion evidence entries must be mappings")
        path = str(Path(str(entry["original_prediction_path"])).resolve())
        if path in output:
            raise ValueError(f"duplicate conversion evidence path {path}")
        output[path] = entry
    return output


def unresolved_metadata_fields(
    audit: CandidateAudit,
    evidence: Mapping[str, Any] | None,
) -> tuple[str, ...]:
    if evidence is None:
        return (
            "code_hash",
            "compute_cost",
            "compute_cost_provenance",
            "config_hash",
            "contract_role",
            "deployment_contract",
        )
    required = {
        "code_hash",
        "compute_cost",
        "compute_cost_provenance",
        "config_hash",
        "contract_role",
        "deployment_contract",
    }
    missing = sorted(
        key
        for key in required
        if key not in evidence
        or evidence[key] is None
        or evidence[key] == ""
    )
    if str(evidence.get("original_prediction_checksum", "")) != (
        audit.original_checksum
    ):
        missing.append("original_prediction_checksum_match")
    return tuple(sorted(set(missing)))


def build_v4_manifest_payload(
    audit: CandidateAudit,
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    unresolved = unresolved_metadata_fields(audit, evidence)
    if unresolved:
        raise ValueError(f"manifest metadata remains unresolved: {unresolved}")
    contract_payload = evidence["deployment_contract"]
    if not isinstance(contract_payload, Mapping):
        raise ValueError("deployment contract evidence must be a mapping")
    contract = DeploymentContract.from_dict(contract_payload)
    role = str(evidence["contract_role"])
    if role != contract.role.value:
        raise ValueError("evidenced contract role disagrees with contract")
    if contract.dataset_id != audit.dataset:
        raise ValueError("evidenced contract dataset disagrees with candidate")
    if role == "source":
        permitted_splits = ["train", "validation"]
        evaluation_split = "validation"
    elif role == "target":
        permitted_splits = ["test"]
        evaluation_split = "test"
    else:
        raise ValueError("converted manifest role must be source or target")
    return {
        "schema_version": "coregraph_prediction_manifest_v4",
        "expert_id": audit.expert_id,
        "dataset": audit.dataset,
        "task": "node_classification",
        "prediction_unit": "node",
        "protocol_id": audit.protocol_id,
        "contract_coordinate_hash": contract.coordinate_hash,
        "contract_id": contract.contract_id,
        "environment_id": contract.environment_id,
        "expert_prediction_seed": audit.expert_prediction_seed,
        "fold": audit.fold,
        "prediction_path": audit.original_path,
        "prediction_checksum": audit.original_checksum,
        "config_hash": str(evidence["config_hash"]),
        "code_hash": str(evidence["code_hash"]),
        "contract_role": role,
        "deployment_contract": contract.to_dict(),
        "expert_available": True,
        "availability_reason_codes": ["available"],
        "compute_cost": float(evidence["compute_cost"]),
        "compute_cost_provenance": str(evidence["compute_cost_provenance"]),
        "score_type": "PROBABILITY",
        "permitted_splits": permitted_splits,
        "evaluation_split": evaluation_split,
        "row_scope_policy": "filter_and_audit",
        "label_mapping": dict(LABEL_MAPPING),
        "positive_label_id": 1,
        "provider_split_mapping": dict(PROVIDER_SPLIT_MAPPING),
        "original_prediction_path": audit.original_path,
        "original_prediction_checksum": audit.original_checksum,
    }


def build_conversion_records(
    audits: Sequence[CandidateAudit],
    evidence_map: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], ...]:
    metadata = _metadata_entry_by_path(evidence_map)
    records: list[dict[str, Any]] = []
    for audit in audits:
        evidence = metadata.get(str(Path(audit.original_path).resolve()))
        unresolved = unresolved_metadata_fields(audit, evidence)
        reason_codes: list[str] = []
        status = "CONVERTED_V4"
        manifest: dict[str, Any] | None = None
        if not audit.validation_evidence:
            reason_codes.append("VALIDATION_EVIDENCE_MISSING")
        if audit.schema_errors:
            reason_codes.append("PREDICTION_SCHEMA_INVALID")
        if audit.content_errors or audit.duplicate_identifier_count:
            reason_codes.append("PREDICTION_CONTENT_INVALID")
        if unresolved:
            reason_codes.append("BLOCKED_METADATA_UNRESOLVED")
        if reason_codes:
            status = (
                "BLOCKED_METADATA_UNRESOLVED"
                if unresolved
                else "BLOCKED_PREDICTION_AUDIT"
            )
        else:
            assert evidence is not None
            manifest = build_v4_manifest_payload(audit, evidence)
        records.append(
            {
                "conversion_status": status,
                "availability": status == "CONVERTED_V4",
                "availability_reason_codes": reason_codes or ["available"],
                "unresolved_fields": list(unresolved),
                "audit": audit.to_dict(),
                "manifest": manifest,
            }
        )
    return tuple(records)


def _records_by_key(
    records: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str, str, int, str], list[Mapping[str, Any]]]:
    output: dict[
        tuple[str, str, str, int, str],
        list[Mapping[str, Any]],
    ] = {}
    for record in records:
        audit = record["audit"]
        key = (
            str(audit["dataset"]),
            str(audit["protocol_id"]),
            str(audit["expert_id"]),
            int(audit["expert_prediction_seed"]),
            str(audit["fold"]),
        )
        output.setdefault(key, []).append(record)
    return output


def build_completeness_matrix(
    records: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    by_key = _records_by_key(records)
    rows: list[dict[str, Any]] = []
    for dataset in REQUIRED_DATASETS:
        for protocol_id in REQUIRED_PROTOCOLS:
            for seed in REQUIRED_SEEDS:
                for expert_id in REQUIRED_EXPERTS:
                    key = (dataset, protocol_id, expert_id, seed, "fold0")
                    candidates = by_key.get(key, ())
                    unique_checksums = {
                        str(record["audit"]["original_checksum"])
                        for record in candidates
                    }
                    converted = [
                        record
                        for record in candidates
                        if record["conversion_status"] == "CONVERTED_V4"
                    ]
                    for role in REQUIRED_ROLES:
                        role_converted = [
                            record
                            for record in converted
                            if record["manifest"]["contract_role"] == role
                        ]
                        if len(role_converted) == 1:
                            status = "AVAILABLE_V4"
                            reasons = "available"
                            selected = role_converted[0]
                        elif len(role_converted) > 1:
                            status = "BLOCKED_AMBIGUOUS_CONVERTED_ARTIFACTS"
                            reasons = "AMBIGUOUS_CONVERTED_ARTIFACTS"
                            selected = None
                        elif not candidates:
                            status = "MISSING_ARTIFACT"
                            reasons = "MISSING_ARTIFACT"
                            selected = None
                        elif len(unique_checksums) > 1:
                            status = "BLOCKED_AMBIGUOUS_HISTORICAL_CANDIDATES"
                            reasons = "AMBIGUOUS_HISTORICAL_CANDIDATES"
                            selected = None
                        else:
                            status = "BLOCKED_METADATA_UNRESOLVED"
                            reasons = ";".join(
                                sorted(
                                    {
                                        str(reason)
                                        for record in candidates
                                        for reason in record[
                                            "availability_reason_codes"
                                        ]
                                    }
                                )
                            )
                            selected = None
                        rows.append(
                            {
                                "dataset": dataset,
                                "protocol_id": protocol_id,
                                "expert_prediction_seed": seed,
                                "fold": "fold0",
                                "expert_id": expert_id,
                                "contract_role": role,
                                "status": status,
                                "reason_codes": reasons,
                                "candidate_count": len(candidates),
                                "unique_candidate_checksums": len(
                                    unique_checksums
                                ),
                                "original_path": (
                                    selected["audit"]["original_path"]
                                    if selected is not None
                                    else ""
                                ),
                                "original_checksum": (
                                    selected["audit"]["original_checksum"]
                                    if selected is not None
                                    else ""
                                ),
                                "contract_coordinate_hash": (
                                    selected["manifest"][
                                        "contract_coordinate_hash"
                                    ]
                                    if selected is not None
                                    else ""
                                ),
                                "contract_id": (
                                    selected["manifest"]["contract_id"]
                                    if selected is not None
                                    else ""
                                ),
                            }
                        )
    return tuple(rows)


def build_blocked_leakage_report(
    matrix: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    reports = []
    for dataset in REQUIRED_DATASETS:
        for protocol_id in REQUIRED_PROTOCOLS:
            for seed in REQUIRED_SEEDS:
                cells = [
                    row
                    for row in matrix
                    if row["dataset"] == dataset
                    and row["protocol_id"] == protocol_id
                    and int(row["expert_prediction_seed"]) == seed
                ]
                available = all(
                    row["status"] == "AVAILABLE_V4" for row in cells
                )
                reports.append(
                    {
                        "dataset": dataset,
                        "target_protocol_id": protocol_id,
                        "expert_prediction_seed": seed,
                        "fold": "fold0",
                        "status": (
                            "READY_FOR_TYPED_LEAKAGE_AUDIT"
                            if available
                            else "NOT_RUN_BLOCKED_INCOMPLETE_OR_UNRESOLVED_MANIFESTS"
                        ),
                        "passed": False if not available else None,
                        "atomic_leakage_checked": False,
                        "reason_codes": sorted(
                            {
                                str(row["status"])
                                for row in cells
                                if row["status"] != "AVAILABLE_V4"
                            }
                        ),
                    }
                )
    return {
        "schema_version": "coregraph_manifest_leakage_audit_v4",
        "training_performed": False,
        "metric_computation_performed": False,
        "target_oracle_measurement_performed": False,
        "reports": reports,
        "summary": {
            "report_count": len(reports),
            "typed_audits_completed": sum(
                report["atomic_leakage_checked"] is True
                for report in reports
            ),
            "blocked_before_leakage_audit": sum(
                report["atomic_leakage_checked"] is False
                for report in reports
            ),
        },
    }


def validate_converted_bindings(
    artifacts: Sequence[Any],
    registry: Mapping[str, Any],
) -> list[dict[str, Any]]:
    return validate_protocol_bindings(artifacts, registry)


def status_counts(
    values: Iterable[Mapping[str, Any]],
    key: str,
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        status = str(value[key])
        counts[status] = counts.get(status, 0) + 1
    return dict(sorted(counts.items()))
