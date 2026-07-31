"""Canonical, precedence-aware recovery of historical prediction evidence."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from coregraph.contracts.serialization import stable_sha256

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
REQUIRED_DATASETS = ("elliptic", "dgraphfin")
REQUIRED_PROTOCOLS = (
    "strict_inductive",
    "isolated_inductive",
    "transductive_structure",
)
REQUIRED_EXPERTS = ("feature_mlp", "gcn", "graphsage")
REQUIRED_SEEDS = tuple(range(1, 11))
REQUIRED_FOLDS = ("fold0",)

RECOVERY_STATUSES = {
    "RECOVERED_CANONICAL",
    "RECOVERED_COMPATIBLE_ALIAS",
    "INDEX_REFERENCED_FILE_MISSING",
    "CHECKSUM_CONFLICT",
    "RESULT_SIDECAR_MISSING",
    "METADATA_UNRESOLVED",
    "ARTIFACT_GENUINELY_MISSING",
    "EXCLUDED_INTEGRITY",
}
EVIDENCE_PRECEDENCE = {
    "FINAL_EVIDENCE_LOCK": 1,
    "FINAL_MERGED_PREDICTION_INDEX": 2,
    "PER_LANE_MERGE_VALIDATION": 3,
    "PACKAGE_IMPORT_VALIDATION": 4,
    "RESULT_SIDECAR": 5,
    "RAW_NAVIGATION": 6,
}
_CANONICAL_NAME = re.compile(
    r"(?P<dataset>elliptic|dgraphfin)__"
    r"(?P<protocol>strict_inductive|inductive_isolated|isolated_inductive|"
    r"transductive|transductive_structure)__"
    r"(?P<model>mlp|feature_mlp|gcn|sage|graphsage)__"
    r"seed(?P<seed>[1-9]|10)\.csv$"
)


@dataclass(frozen=True)
class EvidenceSource:
    evidence_type: str
    precedence: int
    path: str
    sha256: str
    relevance: str


@dataclass(frozen=True)
class IndexedPredictionRecord:
    evidence_type: str
    evidence_path: str
    dataset: str
    protocol: str
    model: str
    seed: int
    prediction_path: str
    prediction_checksum: str = ""
    result_path: str = ""
    result_checksum: str = ""
    artifact_family: str = ""
    source_package: str = ""
    original_version_alias: str = ""
    import_validation: str = ""
    lock_membership: str = ""
    size_bytes: int | None = None

    @property
    def logical_key(self) -> tuple[str, str, str, int]:
        return (
            self.dataset,
            PROTOCOL_ALIASES.get(self.protocol, self.protocol),
            EXPERT_ALIASES.get(self.model, self.model),
            self.seed,
        )


@dataclass(frozen=True)
class IndexedResultRecord:
    evidence_type: str
    evidence_path: str
    artifact_family: str
    dataset: str
    protocol: str
    model: str
    seed: int
    result_path: str
    result_checksum: str
    prediction_reference: str
    source_package: str
    original_version_alias: str

    @property
    def logical_key(self) -> tuple[str, str, str, int]:
        return (
            self.dataset,
            PROTOCOL_ALIASES.get(self.protocol, self.protocol),
            EXPERT_ALIASES.get(self.model, self.model),
            self.seed,
        )


@dataclass(frozen=True)
class CanonicalRecoveryRecord:
    dataset: str
    task: str
    protocol_id: str
    source_protocol: str
    expert_id: str
    source_model: str
    expert_prediction_seed: int
    fold: str
    status: str
    reason_codes: tuple[str, ...]
    artifact_family: str
    canonical_inventory_path: str
    canonical_inventory_sha256: str
    indexed_prediction_path: str
    indexed_prediction_checksum: str
    indexed_size_bytes: int
    resolved_prediction_path: str
    resolved_prediction_checksum: str
    raw_navigation_candidates: tuple[str, ...]
    source_package: str
    source_archive_path: str
    source_archive_sha256: str
    source_archive_member: str
    archive_present: bool
    original_version_alias: str
    alias_lineage: tuple[str, ...]
    result_path: str
    result_checksum: str
    import_validation: str
    import_validation_path: str
    lock_membership: str
    config_payload: Mapping[str, Any]
    config_sha256: str
    config_provenance_type: str
    config_provenance_path: str
    code_provenance_type: str
    code_provenance_value: str
    code_provenance_path: str
    routing_cost_value: float | None
    routing_cost_unit: str
    routing_cost_provenance: str
    measured_compute_available: bool
    measured_compute_record: Mapping[str, Any]
    base_coordinate_id: str
    base_artifact_hash: str

    def to_csv_row(self) -> dict[str, Any]:
        payload = asdict(self)
        for field in (
            "reason_codes",
            "raw_navigation_candidates",
            "alias_lineage",
        ):
            payload[field] = ";".join(payload[field])
        for field in ("config_payload", "measured_compute_record"):
            payload[field] = json.dumps(
                payload[field], sort_keys=True, separators=(",", ":")
            )
        payload["archive_present"] = str(payload["archive_present"]).lower()
        payload["measured_compute_available"] = str(
            payload["measured_compute_available"]
        ).lower()
        return payload


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_payload(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _prediction_records_from_payload(
    payload: Any,
    *,
    path: Path,
    evidence_type: str,
) -> list[IndexedPredictionRecord]:
    """Extract index records without using the referenced filename as identity."""

    records: list[IndexedPredictionRecord] = []

    def visit(value: Any, context: Mapping[str, Any] | None = None) -> None:
        if isinstance(value, Mapping):
            inherited = dict(context or {})
            for key in (
                "artifact_family",
                "family",
                "lane_id",
                "run_id",
                "version",
            ):
                if value.get(key) is not None and value.get(key) != "":
                    inherited[key] = value[key]
            prediction_value = next(
                (
                    value[key]
                    for key in (
                        "prediction_path",
                        "logical_path",
                        "path",
                        "source_member",
                    )
                    if isinstance(value.get(key), str)
                    and str(value[key]).lower().endswith(".csv")
                ),
                "",
            )
            dataset = str(value.get("dataset", ""))
            protocol = str(value.get("protocol", value.get("protocol_id", "")))
            model = str(value.get("model", value.get("expert_id", "")))
            seed = value.get("seed", value.get("expert_prediction_seed"))
            if prediction_value and dataset and protocol and model and seed is not None:
                records.append(
                    IndexedPredictionRecord(
                        evidence_type=evidence_type,
                        evidence_path=str(path),
                        dataset=dataset,
                        protocol=protocol,
                        model=model,
                        seed=int(seed),
                        prediction_path=str(prediction_value),
                        prediction_checksum=str(
                            value.get(
                                "prediction_checksum",
                                value.get("sha256", value.get("checksum", "")),
                            )
                        ),
                        result_path=str(value.get("result_path", "")),
                        result_checksum=str(value.get("result_checksum", "")),
                        artifact_family=str(
                            value.get(
                                "artifact_family",
                                inherited.get("artifact_family", inherited.get("family", "")),
                            )
                        ),
                        source_package=str(
                            value.get(
                                "source_package",
                                value.get(
                                    "source_path",
                                    inherited.get("lane_id", inherited.get("run_id", "")),
                                ),
                            )
                        ),
                        original_version_alias=str(
                            value.get(
                                "original_version_alias",
                                inherited.get("version", ""),
                            )
                        ),
                        import_validation=str(value.get("import_validation", "")),
                        lock_membership=str(value.get("lock_membership", "")),
                        size_bytes=(
                            int(value["size_bytes"])
                            if value.get("size_bytes") is not None
                            else None
                        ),
                    )
                )
            for child in value.values():
                if isinstance(child, (Mapping, list)):
                    visit(child, inherited)
        elif isinstance(value, list):
            for child in value:
                visit(child, context)

    visit(payload)
    unique = {
        (
            record.logical_key,
            record.prediction_path,
            record.prediction_checksum,
            record.evidence_path,
        ): record
        for record in records
    }
    return [unique[key] for key in sorted(unique)]


def discover_prediction_index_records(
    roots: Sequence[str | Path],
) -> tuple[IndexedPredictionRecord, ...]:
    """Discover JSON/CSV prediction indexes before raw prediction filenames."""

    paths: set[Path] = set()
    for root_value in roots:
        root = Path(root_value).expanduser().resolve()
        if not root.exists():
            continue
        iterator = (root,) if root.is_file() else root.rglob("*")
        for path in iterator:
            if not path.is_file():
                continue
            upper = path.name.upper()
            if (
                "PREDICTION" in upper
                and ("INDEX" in upper or "MANIFEST" in upper)
                and path.suffix.lower() in {".json", ".csv"}
            ):
                paths.add(path.resolve())
    records: list[IndexedPredictionRecord] = []
    for path in sorted(paths):
        evidence_type = (
            "FINAL_MERGED_PREDICTION_INDEX"
            if "FULL10" in path.name.upper()
            else "PACKAGE_IMPORT_VALIDATION"
        )
        try:
            if path.suffix.lower() == ".json":
                records.extend(
                    _prediction_records_from_payload(
                        _json_payload(path),
                        path=path,
                        evidence_type=evidence_type,
                    )
                )
            else:
                with path.open(newline="", encoding="utf-8") as handle:
                    for row in csv.DictReader(handle):
                        records.extend(
                            _prediction_records_from_payload(
                                row,
                                path=path,
                                evidence_type=evidence_type,
                            )
                        )
        except (OSError, ValueError, json.JSONDecodeError):
            continue
    return tuple(
        sorted(
            records,
            key=lambda record: (
                EVIDENCE_PRECEDENCE.get(record.evidence_type, 99),
                record.logical_key,
                record.prediction_path,
                record.evidence_path,
            ),
        )
    )


def _result_records_from_payload(
    payload: Any,
    *,
    path: Path,
    evidence_type: str,
) -> list[IndexedResultRecord]:
    """Extract result identities from JSON, JSONL, or CSV record payloads."""

    records: list[IndexedResultRecord] = []

    def visit(value: Any, context: Mapping[str, Any] | None = None) -> None:
        if isinstance(value, Mapping):
            inherited = dict(context or {})
            for key in (
                "artifact_family",
                "family",
                "lane_id",
                "run_id",
                "version",
            ):
                if value.get(key) not in (None, ""):
                    inherited[key] = value[key]
            nested = value.get("payload")
            scientific = nested if isinstance(nested, Mapping) else value
            dataset = str(
                value.get("dataset", scientific.get("dataset", ""))
            )
            protocol = str(
                value.get(
                    "protocol",
                    value.get(
                        "protocol_id",
                        scientific.get(
                            "protocol",
                            scientific.get("protocol_id", ""),
                        ),
                    ),
                )
            )
            model = str(
                value.get(
                    "model",
                    value.get(
                        "expert_id",
                        scientific.get(
                            "model",
                            scientific.get("expert_id", ""),
                        ),
                    ),
                )
            )
            seed = value.get(
                "seed",
                value.get(
                    "expert_prediction_seed",
                    scientific.get(
                        "seed",
                        scientific.get("expert_prediction_seed"),
                    ),
                ),
            )
            result_reference = next(
                (
                    str(value[key])
                    for key in (
                        "result_path",
                        "logical_path",
                        "source_member",
                        "path",
                    )
                    if isinstance(value.get(key), str)
                    and str(value[key]).lower().endswith((".json", ".jsonl"))
                ),
                str(path),
            )
            prediction_reference = next(
                (
                    str(value[key])
                    for key in (
                        "prediction_path",
                        "prediction_reference",
                        "prediction_member",
                    )
                    if isinstance(value.get(key), str)
                ),
                "",
            )
            if dataset and protocol and model and seed is not None:
                records.append(
                    IndexedResultRecord(
                        evidence_type=evidence_type,
                        evidence_path=str(path),
                        artifact_family=str(
                            value.get(
                                "artifact_family",
                                inherited.get(
                                    "artifact_family",
                                    inherited.get("family", ""),
                                ),
                            )
                        ),
                        dataset=dataset,
                        protocol=protocol,
                        model=model,
                        seed=int(seed),
                        result_path=result_reference,
                        result_checksum=str(
                            value.get(
                                "result_checksum",
                                value.get("sha256", value.get("checksum", "")),
                            )
                        ),
                        prediction_reference=prediction_reference,
                        source_package=str(
                            value.get(
                                "source_package",
                                value.get(
                                    "source_path",
                                    inherited.get(
                                        "lane_id",
                                        inherited.get("run_id", ""),
                                    ),
                                ),
                            )
                        ),
                        original_version_alias=str(
                            value.get(
                                "original_version_alias",
                                inherited.get("version", ""),
                            )
                        ),
                    )
                )
            for child in value.values():
                if isinstance(child, (Mapping, list)):
                    visit(child, inherited)
        elif isinstance(value, list):
            for child in value:
                visit(child, context)

    visit(payload)
    unique = {
        (
            record.logical_key,
            record.result_path,
            record.evidence_path,
        ): record
        for record in records
    }
    return [unique[key] for key in sorted(unique)]


def discover_result_index_records(
    roots: Sequence[str | Path],
) -> tuple[IndexedResultRecord, ...]:
    """Discover result indexes and JSONL sidecars by structured record fields."""

    paths: set[Path] = set()
    for root_value in roots:
        root = Path(root_value).expanduser().resolve()
        if not root.exists():
            continue
        iterator = (root,) if root.is_file() else root.rglob("*")
        for path in iterator:
            if not path.is_file():
                continue
            upper = path.name.upper()
            if (
                (
                    ("RESULT" in upper and "INDEX" in upper)
                    or upper.startswith("RESULTS_FULL10")
                    or path.as_posix().endswith("results/runs_rb09v3/runs.csv")
                )
                and path.suffix.lower() in {".json", ".jsonl", ".csv"}
            ):
                paths.add(path.resolve())
    records: list[IndexedResultRecord] = []
    for path in sorted(paths):
        try:
            if path.suffix.lower() == ".json":
                payloads: Iterable[Any] = (_json_payload(path),)
            elif path.suffix.lower() == ".jsonl":
                payloads = (
                    json.loads(line)
                    for line in path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                )
            else:
                with path.open(newline="", encoding="utf-8") as handle:
                    payloads = tuple(csv.DictReader(handle))
            for payload in payloads:
                records.extend(
                    _result_records_from_payload(
                        payload,
                        path=path,
                        evidence_type="RESULT_SIDECAR",
                    )
                )
        except (OSError, ValueError, json.JSONDecodeError):
            continue
    return tuple(
        sorted(
            records,
            key=lambda record: (
                record.logical_key,
                record.result_path,
                record.evidence_path,
            ),
        )
    )


def discover_authoritative_sources(
    roots: Sequence[str | Path],
) -> tuple[EvidenceSource, ...]:
    """Inventory relevant locks, indexes, imports, sidecars, and alias records."""

    sources: dict[Path, tuple[str, str]] = {}
    for root_value in roots:
        root = Path(root_value).expanduser().resolve()
        if not root.exists():
            continue
        iterator = (root,) if root.is_file() else root.rglob("*")
        for path in iterator:
            if not path.is_file():
                continue
            upper = path.name.upper()
            evidence_type = ""
            relevance = ""
            if "EVIDENCE_LOCK" in upper and path.suffix.lower() == ".json":
                evidence_type = "FINAL_EVIDENCE_LOCK"
                relevance = "lock"
            elif "PREDICTION" in upper and "INDEX" in upper:
                evidence_type = "FINAL_MERGED_PREDICTION_INDEX"
                relevance = "prediction_index"
            elif "MERGE_VALIDATION" in upper:
                evidence_type = "PER_LANE_MERGE_VALIDATION"
                relevance = "merge_validation"
            elif (
                upper == "IMPORT_MANIFEST.JSON"
                or "IMPORT_VALIDATION" in upper
                or "SOURCE_TRACE" in upper
            ):
                evidence_type = "PACKAGE_IMPORT_VALIDATION"
                relevance = "import_or_source_trace"
            elif "VALIDATION" in upper and path.suffix.lower() == ".json":
                evidence_type = "PER_LANE_MERGE_VALIDATION"
                relevance = "validation_report"
            elif (
                upper.startswith("RESULTS_FULL10")
                or "RESULT_INDEX" in upper
                or path.as_posix().endswith("results/runs_rb09v3/runs.csv")
            ):
                evidence_type = "RESULT_SIDECAR"
                relevance = "result_sidecar"
            elif upper in {
                "ARTIFACT_FAMILY.JSON",
                "PREDICTIONS_MANIFEST.JSON",
                "KAGGLE_WORKSPACE_FILE_MANIFEST.JSON",
                "KAGGLE_DISCOVERY_REPORT.JSON",
                "KAGGLE_FILE_MIGRATION_REPORT.JSON",
            }:
                evidence_type = (
                    "FINAL_MERGED_PREDICTION_INDEX"
                    if upper == "PREDICTIONS_MANIFEST.JSON"
                    else "PACKAGE_IMPORT_VALIDATION"
                )
                relevance = "canonical_inventory_or_alias"
            if evidence_type:
                sources[path.resolve()] = (evidence_type, relevance)
    return tuple(
        EvidenceSource(
            evidence_type=evidence_type,
            precedence=EVIDENCE_PRECEDENCE[evidence_type],
            path=str(path),
            sha256=sha256_path(path),
            relevance=relevance,
        )
        for path, (evidence_type, relevance) in sorted(
            sources.items(), key=lambda item: str(item[0])
        )
    )


def _evidence_sources_of_type(
    roots: Sequence[str | Path],
    evidence_type: str,
) -> tuple[EvidenceSource, ...]:
    return tuple(
        source
        for source in discover_authoritative_sources(roots)
        if source.evidence_type == evidence_type
    )


def discover_evidence_locks(
    roots: Sequence[str | Path],
) -> tuple[EvidenceSource, ...]:
    return _evidence_sources_of_type(roots, "FINAL_EVIDENCE_LOCK")


def discover_package_import_manifests(
    roots: Sequence[str | Path],
) -> tuple[EvidenceSource, ...]:
    return _evidence_sources_of_type(roots, "PACKAGE_IMPORT_VALIDATION")


def discover_validation_reports(
    roots: Sequence[str | Path],
) -> tuple[EvidenceSource, ...]:
    return _evidence_sources_of_type(roots, "PER_LANE_MERGE_VALIDATION")


def discover_raw_filesystem_predictions(
    roots: Sequence[str | Path],
) -> Mapping[str, tuple[str, ...]]:
    """Return filename candidates for navigation only, never identity."""

    return {
        name: tuple(paths)
        for name, paths in sorted(
            _raw_candidates(
                tuple(Path(root).expanduser().resolve() for root in roots)
            ).items()
        )
    }


def _find_nested_rb15_manifests(payload: Any) -> list[Mapping[str, Any]]:
    matches: list[Mapping[str, Any]] = []
    if isinstance(payload, Mapping):
        if (
            payload.get("artifact_family") == "RB15_graphsafe_tta"
            and isinstance(payload.get("prediction_locations"), list)
            and isinstance(payload.get("source_files"), list)
        ):
            matches.append(payload)
        for value in payload.values():
            matches.extend(_find_nested_rb15_manifests(value))
    elif isinstance(payload, list):
        for value in payload:
            matches.extend(_find_nested_rb15_manifests(value))
    return matches


def _rb15_import_evidence(
    roots: Sequence[Path],
    canonical_members: Mapping[str, tuple[str, str, str, int]],
) -> tuple[
    dict[tuple[str, str, str, int], tuple[str, str, str]],
    dict[str, tuple[str, int]],
]:
    locations: dict[tuple[str, str, str, int], tuple[str, str, str]] = {}
    archives: dict[str, tuple[str, int]] = {}
    for root in roots:
        for path in root.rglob("import_manifest.json"):
            try:
                payload = _json_payload(path)
            except (OSError, json.JSONDecodeError):
                continue
            for manifest in _find_nested_rb15_manifests(payload):
                source_files = {
                    str(item.get("path", "")): (
                        str(item.get("sha256", "")),
                        int(item.get("size_bytes", 0)),
                    )
                    for item in manifest["source_files"]
                    if isinstance(item, Mapping)
                }
                archives.update(source_files)
                for value in manifest["prediction_locations"]:
                    archive_path, separator, member = str(value).partition("::")
                    key = canonical_members.get(Path(member).name)
                    if not separator or key is None:
                        continue
                    archive_sha = source_files.get(archive_path, ("", 0))[0]
                    locations.setdefault(
                        key,
                        (archive_path, member, archive_sha),
                    )
    return locations, archives


def _raw_candidates(roots: Sequence[Path]) -> dict[str, list[str]]:
    candidates: dict[str, list[str]] = {}
    for root in roots:
        for path in root.rglob("*.csv"):
            if _CANONICAL_NAME.fullmatch(path.name) is not None:
                candidates.setdefault(path.name, []).append(str(path.resolve()))
    return {key: sorted(set(value)) for key, value in candidates.items()}


def _resolve_archive_path(
    recorded_path: str,
    roots: Sequence[Path],
) -> tuple[Path | None, tuple[str, ...]]:
    candidates: list[Path] = []
    direct = Path(recorded_path)
    if direct.is_file():
        candidates.append(direct.resolve())
    for root in roots:
        for candidate in root.rglob(direct.name):
            if candidate.is_file():
                candidates.append(candidate.resolve())
    unique = tuple(sorted({str(path) for path in candidates}))
    return (Path(unique[0]) if unique else None, unique)


def _archive_member_checksum(archive: Path, member: str) -> str:
    digest = hashlib.sha256()
    with zipfile.ZipFile(archive) as handle:
        with handle.open(member) as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _result_records(path: Path) -> dict[tuple[str, str, str, int], Mapping[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {
        (
            row["dataset"],
            PROTOCOL_ALIASES[row["protocol"]],
            EXPERT_ALIASES[row["model"]],
            int(row["seed"]),
        ): row
        for row in rows
    }


def recover_rb09v3(
    *,
    historical_root: str | Path,
    search_roots: Sequence[str | Path],
) -> tuple[
    tuple[CanonicalRecoveryRecord, ...],
    tuple[EvidenceSource, ...],
    Mapping[str, Any],
]:
    """Reconcile the canonical RB09v3 inventory against current local files."""

    historical = Path(historical_root).expanduser().resolve()
    roots = tuple(Path(value).expanduser().resolve() for value in search_roots)
    inventory_path = historical / "results/runs_rb09v3/predictions_manifest.json"
    family_path = historical / "results/runs_rb09v3/ARTIFACT_FAMILY.json"
    results_path = historical / "results/runs_rb09v3/runs.csv"
    inventory = _json_payload(inventory_path)
    family = _json_payload(family_path)
    if (
        family.get("n_run_rows") != 180
        or family.get("n_prediction_files") != 180
        or inventory.get("n_prediction_files") != 180
        or len(inventory.get("files", ())) != 180
    ):
        raise ValueError("RB09v3 canonical inventory does not assert exactly 180 files")
    result_records = _result_records(results_path)
    if len(result_records) != 180:
        raise ValueError("RB09v3 result sidecar does not contain exactly 180 records")
    canonical_members = {
        Path(str(item["path"])).name: (
            str(item["dataset"]),
            PROTOCOL_ALIASES[str(item["protocol"])],
            EXPERT_ALIASES[str(item["model"])],
            int(item["seed"]),
        )
        for item in inventory["files"]
    }
    if len(canonical_members) != 180:
        raise ValueError("RB09v3 canonical inventory filenames are not unique")
    import_locations, _ = _rb15_import_evidence(roots, canonical_members)
    raw_by_name = _raw_candidates(roots)
    inventory_checksum = sha256_path(inventory_path)
    result_checksum = sha256_path(results_path)
    records: list[CanonicalRecoveryRecord] = []
    for item in inventory["files"]:
        source_protocol = str(item["protocol"])
        source_model = str(item["model"])
        key = (
            str(item["dataset"]),
            PROTOCOL_ALIASES[source_protocol],
            EXPERT_ALIASES[source_model],
            int(item["seed"]),
        )
        result = result_records.get(key)
        indexed_path = historical / str(item["path"])
        archive_path_value, archive_member, archive_sha = import_locations.get(
            key,
            ("", "", ""),
        )
        archive, archive_alias_candidates = _resolve_archive_path(
            archive_path_value,
            roots,
        ) if archive_path_value else (None, ())
        resolved_path = ""
        resolved_checksum = ""
        status = "INDEX_REFERENCED_FILE_MISSING"
        reasons: list[str] = []
        if indexed_path.is_file():
            actual_size = indexed_path.stat().st_size
            if actual_size != int(item["size_bytes"]):
                status = "CHECKSUM_CONFLICT"
                reasons.append("CANONICAL_PATH_SIZE_CONFLICT")
            else:
                status = "RECOVERED_CANONICAL"
                resolved_path = str(indexed_path.resolve())
                resolved_checksum = sha256_path(indexed_path)
        elif archive is not None:
            actual_archive_sha = sha256_path(archive)
            if archive_sha and actual_archive_sha != archive_sha:
                status = "CHECKSUM_CONFLICT"
                reasons.append("SOURCE_ARCHIVE_CHECKSUM_CONFLICT")
            else:
                try:
                    with zipfile.ZipFile(archive) as handle:
                        info = handle.getinfo(archive_member)
                    if info.file_size != int(item["size_bytes"]):
                        status = "CHECKSUM_CONFLICT"
                        reasons.append("ARCHIVE_MEMBER_SIZE_CONFLICT")
                    else:
                        status = "RECOVERED_CANONICAL"
                        resolved_path = f"{archive.resolve()}::{archive_member}"
                        resolved_checksum = _archive_member_checksum(
                            archive,
                            archive_member,
                        )
                except (KeyError, zipfile.BadZipFile):
                    status = "INDEX_REFERENCED_FILE_MISSING"
                    reasons.append("INDEXED_ARCHIVE_MEMBER_MISSING")
        else:
            reasons.extend(
                (
                    "CANONICAL_PREDICTION_PATH_MISSING",
                    "INDEXED_SOURCE_ARCHIVE_MISSING",
                )
            )
        raw_candidates = tuple(raw_by_name.get(Path(str(item["path"])).name, ()))
        if raw_candidates and status == "INDEX_REFERENCED_FILE_MISSING":
            reasons.append("UNVERIFIED_RAW_COORDINATE_CANDIDATES_NOT_ALIASED")
        if result is None:
            status = "RESULT_SIDECAR_MISSING"
            reasons.append("CANONICAL_RESULT_ROW_MISSING")
            result = {}
        config_payload: dict[str, Any] = {
            key_name: result.get(key_name, "")
            for key_name in (
                "command",
                "early_stopping_metric",
                "early_stopping_split",
                "scaler_mode",
                "graph_mode",
                "split_name",
            )
            if result.get(key_name, "") != ""
        }
        config_sha = stable_sha256(config_payload) if config_payload else ""
        runtime = result.get("runtime_seconds", "")
        measured_record: dict[str, Any] = {}
        if runtime != "":
            measured_record = {
                "runtime_seconds": float(runtime),
                "measurement_scope": "per_run_result_record",
                "device": (
                    "cuda"
                    if "--device cuda" in str(result.get("command", ""))
                    else "unresolved"
                ),
                "hardware": "unresolved_from_per_run_record",
                "peak_memory": None,
            }
        code_value = str(result.get("git_commit", ""))
        code_type = (
            "UNRESOLVED_LEGACY_CODE"
            if not re.fullmatch(r"[0-9a-f]{40}", code_value)
            else "GIT_COMMIT"
        )
        base_coordinate_id = stable_sha256(
            {
                "dataset": key[0],
                "task": "node_classification",
                "protocol_id": key[1],
                "expert_id": key[2],
                "expert_prediction_seed": key[3],
                "fold": "fold0",
            }
        )
        base_artifact_hash = (
            stable_sha256(
                {
                    "base_coordinate_id": base_coordinate_id,
                    "prediction_checksum": resolved_checksum,
                    "artifact_family": family["artifact_family"],
                }
            )
            if resolved_checksum
            else ""
        )
        records.append(
            CanonicalRecoveryRecord(
                dataset=key[0],
                task="node_classification",
                protocol_id=key[1],
                source_protocol=source_protocol,
                expert_id=key[2],
                source_model=source_model,
                expert_prediction_seed=key[3],
                fold="fold0",
                status=status,
                reason_codes=tuple(sorted(set(reasons))),
                artifact_family=str(family["artifact_family"]),
                canonical_inventory_path=str(inventory_path),
                canonical_inventory_sha256=inventory_checksum,
                indexed_prediction_path=str(item["path"]),
                indexed_prediction_checksum="",
                indexed_size_bytes=int(item["size_bytes"]),
                resolved_prediction_path=resolved_path,
                resolved_prediction_checksum=resolved_checksum,
                raw_navigation_candidates=raw_candidates,
                source_package=Path(archive_path_value).name if archive_path_value else "",
                source_archive_path=archive_path_value,
                source_archive_sha256=archive_sha,
                source_archive_member=archive_member,
                archive_present=archive is not None,
                original_version_alias="RB09v3",
                alias_lineage=archive_alias_candidates,
                result_path=f"{results_path}#{key[0]}:{source_protocol}:"
                f"{source_model}:seed{key[3]}",
                result_checksum=result_checksum,
                import_validation=(
                    "RB15_IMPORT_ERRORS_EMPTY_AND_MEMBER_REFERENCED"
                    if archive_path_value
                    else "NO_RB15_IMPORT_REFERENCE"
                ),
                import_validation_path=(
                    str(
                        historical
                        / "results/runs_rb15_graphsafe_tta/import_manifest.json"
                    )
                    if archive_path_value
                    else ""
                ),
                lock_membership=(
                    "RB09V3_ARTIFACT_FAMILY_180_AND_RB15_CONSUMED_MEMBER"
                    if archive_path_value
                    else "RB09V3_ARTIFACT_FAMILY_180"
                ),
                config_payload=config_payload,
                config_sha256=config_sha,
                config_provenance_type="CANONICAL_RESULT_COMMAND_AND_FIELDS",
                config_provenance_path=str(results_path),
                code_provenance_type=code_type,
                code_provenance_value=code_value or "UNRESOLVED",
                code_provenance_path=str(results_path),
                routing_cost_value=None,
                routing_cost_unit="",
                routing_cost_provenance="UNRESOLVED",
                measured_compute_available=bool(measured_record),
                measured_compute_record=measured_record,
                base_coordinate_id=base_coordinate_id,
                base_artifact_hash=base_artifact_hash,
            )
        )
    expected_keys = {
        (dataset, protocol, expert, seed)
        for dataset in REQUIRED_DATASETS
        for protocol in REQUIRED_PROTOCOLS
        for expert in REQUIRED_EXPERTS
        for seed in REQUIRED_SEEDS
    }
    actual_keys = {
        (
            record.dataset,
            record.protocol_id,
            record.expert_id,
            record.expert_prediction_seed,
        )
        for record in records
    }
    if len(records) != 180 or actual_keys != expected_keys:
        raise ValueError("RB09v3 recovered coordinate surface is not exactly 180")
    status_counts: dict[str, int] = {}
    for record in records:
        if record.status not in RECOVERY_STATUSES:
            raise ValueError(f"unrecognized recovery status {record.status}")
        status_counts[record.status] = status_counts.get(record.status, 0) + 1
    summary = {
        "canonical_inventory_count": len(records),
        "usable_artifact_count": sum(
            record.status
            in {"RECOVERED_CANONICAL", "RECOVERED_COMPATIBLE_ALIAS"}
            for record in records
        ),
        "missing_index_reference_count": sum(
            record.status == "INDEX_REFERENCED_FILE_MISSING"
            for record in records
        ),
        "metadata_blocker_count": sum(
            record.status
            in {"RESULT_SIDECAR_MISSING", "METADATA_UNRESOLVED"}
            for record in records
        ),
        "unresolved_code_provenance_count": sum(
            record.code_provenance_type == "UNRESOLVED_LEGACY_CODE"
            for record in records
        ),
        "unresolved_routing_cost_count": sum(
            record.routing_cost_value is None for record in records
        ),
        "unresolved_contract_binding_count": sum(
            not record.base_artifact_hash for record in records
        ),
        "true_missing_artifact_count": sum(
            record.status == "ARTIFACT_GENUINELY_MISSING"
            for record in records
        ),
        "status_counts": dict(sorted(status_counts.items())),
        "source_archive_count": len(
            {record.source_archive_path for record in records if record.source_archive_path}
        ),
        "source_archives_present": len(
            {
                record.source_archive_path
                for record in records
                if record.source_archive_path and record.archive_present
            }
        ),
    }
    return (
        tuple(
            sorted(
                records,
                key=lambda record: (
                    record.dataset,
                    record.protocol_id,
                    record.expert_id,
                    record.expert_prediction_seed,
                ),
            )
        ),
        discover_authoritative_sources(roots),
        summary,
    )


def base_completeness_matrix(
    records: Sequence[CanonicalRecoveryRecord],
) -> list[dict[str, Any]]:
    rows = []
    for record in records:
        if record.status in {"RECOVERED_CANONICAL", "RECOVERED_COMPATIBLE_ALIAS"}:
            status = (
                "COMPLETE"
                if record.routing_cost_value is not None
                else "BLOCKED_PROVENANCE"
            )
        elif record.status == "EXCLUDED_INTEGRITY":
            status = "EXCLUDED_INTEGRITY"
        else:
            status = "BLOCKED_BASE_ARTIFACT"
        rows.append(
            {
                "dataset": record.dataset,
                "task": record.task,
                "protocol_id": record.protocol_id,
                "expert_id": record.expert_id,
                "expert_prediction_seed": record.expert_prediction_seed,
                "fold": record.fold,
                "status": status,
                "recovery_status": record.status,
                "reason_codes": ";".join(record.reason_codes),
                "base_coordinate_id": record.base_coordinate_id,
                "base_artifact_hash": record.base_artifact_hash,
                "prediction_path": record.resolved_prediction_path,
                "prediction_checksum": record.resolved_prediction_checksum,
                "contract_coordinate_hash": "",
                "config_sha256": record.config_sha256,
                "code_provenance_type": record.code_provenance_type,
                "routing_cost_provenance": record.routing_cost_provenance,
            }
        )
    if len(rows) != 180:
        raise ValueError("base artifact completeness matrix must have 180 rows")
    return rows


def scenario_completeness_surfaces(
    base_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_key = {
        (
            str(row["dataset"]),
            str(row["protocol_id"]),
            str(row["expert_id"]),
            int(row["expert_prediction_seed"]),
            str(row["fold"]),
        ): row
        for row in base_rows
    }
    scenarios: list[dict[str, Any]] = []
    scenario_rows: list[dict[str, Any]] = []
    for dataset in REQUIRED_DATASETS:
        for target_protocol in REQUIRED_PROTOCOLS:
            source_protocols = tuple(
                protocol
                for protocol in REQUIRED_PROTOCOLS
                if protocol != target_protocol
            )
            for seed in REQUIRED_SEEDS:
                scenario_id = make_recovery_scenario_id(
                    dataset=dataset,
                    target_protocol_id=target_protocol,
                    seed=seed,
                    fold="fold0",
                )
                bindings: list[dict[str, Any]] = []
                for protocol in REQUIRED_PROTOCOLS:
                    role = "target" if protocol == target_protocol else "source"
                    for expert in REQUIRED_EXPERTS:
                        base = by_key[(dataset, protocol, expert, seed, "fold0")]
                        binding_status = (
                            "COMPLETE"
                            if base["status"] == "COMPLETE"
                            else "BLOCKED_BASE_ARTIFACT"
                        )
                        role_binding_id = stable_sha256(
                            {
                                "scenario_id": scenario_id,
                                "base_coordinate_id": base["base_coordinate_id"],
                                "base_artifact_hash": base["base_artifact_hash"],
                                "role": role,
                                "protocol_id": protocol,
                                "expert_id": expert,
                                "permitted_splits": (
                                    ["test"]
                                    if role == "target"
                                    else ["train", "validation"]
                                ),
                                "require_label_known": True,
                            }
                        )
                        bindings.append(
                            {
                                "scenario_id": scenario_id,
                                "role_binding_id": role_binding_id,
                                "base_coordinate_id": base["base_coordinate_id"],
                                "base_artifact_hash": base["base_artifact_hash"],
                                "base_protocol_id": protocol,
                                "bound_protocol_id": protocol,
                                "expert_id": expert,
                                "role": role,
                                "permitted_splits": (
                                    ["test"]
                                    if role == "target"
                                    else ["train", "validation"]
                                ),
                                "evaluation_split": (
                                    "test" if role == "target" else "validation"
                                ),
                                "require_label_known": True,
                                "status": binding_status,
                                "recovery_status": base["recovery_status"],
                            }
                        )
                statuses = {binding["status"] for binding in bindings}
                scenario_status = (
                    "BLOCKED_CONTRACT_BINDING"
                    if statuses == {"COMPLETE"}
                    else "BLOCKED_BASE_ARTIFACT"
                )
                contract_binding_status = (
                    "BLOCKED_UNRESOLVED_TARGET_OPERATIONAL_CONTRACT"
                    if statuses == {"COMPLETE"}
                    else "BLOCKED_UNRESOLVED_CANONICAL_BASE_ARTIFACT"
                )
                scenario_rows.append(
                    {
                        "scenario_id": scenario_id,
                        "dataset": dataset,
                        "target_protocol_id": target_protocol,
                        "source_protocol_ids": ";".join(source_protocols),
                        "expert_prediction_seed": seed,
                        "fold": "fold0",
                        "access_regime": "DG_NO_TARGET",
                        "binding_count": 9,
                        "source_binding_count": 6,
                        "target_binding_count": 3,
                        "status": scenario_status,
                        "contract_binding_status": contract_binding_status,
                        "reason_codes": (
                            ""
                            if scenario_status == "COMPLETE"
                            else (
                                "ONE_OR_MORE_BASE_ARTIFACTS_BLOCKED"
                                if scenario_status == "BLOCKED_BASE_ARTIFACT"
                                else "TARGET_OPERATIONAL_CONTRACT_UNRESOLVED"
                            )
                        ),
                        "leakage_status": (
                            "PASS_SCENARIO_STRUCTURE"
                            if target_protocol not in source_protocols
                            else "BLOCKED_LEAKAGE"
                        ),
                    }
                )
                scenarios.append(
                    {
                        "scenario_id": scenario_id,
                        "dataset": dataset,
                        "target_protocol_id": target_protocol,
                        "source_protocol_ids": list(source_protocols),
                        "expert_prediction_seed": seed,
                        "fold": "fold0",
                        "access_regime": "DG_NO_TARGET",
                        "target_operational_contract": None,
                        "target_contract_coordinate_hash": "",
                        "target_contract_id": "",
                        "contract_binding_status": contract_binding_status,
                        "no_target_labels_during_fitting": True,
                        "status": scenario_status,
                        "bindings": bindings,
                    }
                )
    total_bindings = sum(len(scenario["bindings"]) for scenario in scenarios)
    if len(scenario_rows) != 60 or len(scenarios) != 60 or total_bindings != 540:
        raise ValueError("scenario surfaces must contain exactly 60/540 records")
    return (
        scenario_rows,
        {
            "schema_version": "coregraph_scenario_binding_index_v5",
            "scenario_count": 60,
            "binding_count": 540,
            "source_binding_count": 360,
            "target_binding_count": 180,
            "scenarios": scenarios,
        },
    )


def make_recovery_scenario_id(
    *,
    dataset: str,
    target_protocol_id: str,
    seed: int,
    fold: str,
) -> str:
    return "scenario-" + stable_sha256(
        {
            "dataset": dataset,
            "target_protocol_id": target_protocol_id,
            "expert_prediction_seed": seed,
            "fold": fold,
            "access_regime": "DG_NO_TARGET",
        }
    )[:24]
