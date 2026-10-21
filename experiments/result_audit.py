"""
experiments.result_audit

Utilities for planning and validating long-running experiment artifacts.

This module is intentionally training-free. It inspects expected JSON result
files, validates their schema, and writes small manifest reports so a partial
sweep can be resumed or reviewed without launching any models.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence


DEFAULT_DATASETS = ("elliptic", "dgraphfin", "tfinance")
DEFAULT_MULTI_MODELS = (
    "graph_transformer",
    "gps",
    "pcgnn",
    "snapshot_tgn",
    "sage",
    "gcn",
)
DEFAULT_ABLATION_MODELS = ("sage", "graph_transformer", "gps", "pcgnn")
DEFAULT_MULTI_SEEDS = (42, 43, 44, 45, 46)
DEFAULT_ABLATION_SEEDS = (42, 43, 44)
SCALER_MODES = {"train_only", "full_population", "none"}


@dataclass(frozen=True)
class SweepSpec:
    name: str
    result_dir: str
    models: Sequence[str]
    seeds: Sequence[int]
    validator: Callable[[Mapping], List[str]]


@dataclass
class ArtifactAuditRow:
    sweep: str
    dataset: str
    model: str
    seed: int
    path: str
    status: str
    detail: str = ""
    size_bytes: int = 0
    sha256: str = ""


def _is_number(value) -> bool:
    if isinstance(value, bool):
        return False
    if not isinstance(value, (int, float)):
        return False
    return math.isfinite(float(value))


def _check_metric_dict(
    payload: Mapping,
    prefix: str,
    required: Sequence[str],
    problems: List[str],
) -> None:
    if not isinstance(payload, Mapping):
        problems.append(f"{prefix} must be an object")
        return
    for key in required:
        if key not in payload:
            problems.append(f"{prefix}.{key} missing")
            continue
        value = payload[key]
        if not _is_number(value):
            problems.append(f"{prefix}.{key} must be a finite number")
            continue
        if key in {"f1", "precision", "recall", "accuracy", "auc"}:
            if not 0.0 <= float(value) <= 1.0:
                problems.append(f"{prefix}.{key} outside [0, 1]")


def _check_identity(payload: Mapping, problems: List[str]) -> None:
    for key in ("dataset", "model", "seed"):
        if key not in payload:
            problems.append(f"{key} missing")
    if "seed" in payload and not isinstance(payload["seed"], int):
        problems.append("seed must be an integer")


def _check_schema_metadata(
    payload: Mapping,
    problems: List[str],
    expected_schema: str,
) -> None:
    if payload.get("result_schema_version") != expected_schema:
        problems.append(
            f"result_schema_version must be {expected_schema!r}"
        )
    if payload.get("scaler_mode") not in SCALER_MODES:
        problems.append("scaler_mode missing or invalid")
    if not isinstance(payload.get("hyperparams"), Mapping):
        problems.append("hyperparams must be an object")
    if not isinstance(payload.get("run_metadata"), Mapping):
        problems.append("run_metadata must be an object")


def validate_multi_result(payload: Mapping) -> List[str]:
    problems: List[str] = []
    _check_identity(payload, problems)
    _check_schema_metadata(payload, problems, expected_schema="multi_v2")
    _check_metric_dict(
        payload.get("transductive", {}),
        "transductive",
        ("f1", "precision", "recall", "auc"),
        problems,
    )
    _check_metric_dict(
        payload.get("inductive", {}),
        "inductive",
        ("f1", "precision", "recall", "auc"),
        problems,
    )
    tpc = payload.get("tpc_tta", {})
    if not isinstance(tpc, Mapping):
        problems.append("tpc_tta must be an object")
    else:
        for variant in ("raw", "temp", "tpc_tta"):
            _check_metric_dict(
                tpc.get(variant, {}),
                f"tpc_tta.{variant}",
                ("f1", "precision", "recall"),
                problems,
            )
    return problems


def validate_shuffle_result(payload: Mapping) -> List[str]:
    problems: List[str] = []
    _check_identity(payload, problems)
    _check_schema_metadata(payload, problems, expected_schema="shuffle_v2")
    for variant in ("real", "shuffled", "none"):
        _check_metric_dict(
            payload.get(variant, {}),
            variant,
            ("f1", "precision", "recall", "auc"),
            problems,
        )
    return problems


def validate_tpc_result(payload: Mapping) -> List[str]:
    problems: List[str] = []
    _check_identity(payload, problems)
    _check_schema_metadata(payload, problems, expected_schema="tpc_tta_v2")
    ablation = payload.get("ablation", {})
    if not isinstance(ablation, Mapping):
        problems.append("ablation must be an object")
        return problems
    for variant in ("raw", "temp", "prior", "thresh", "tpc_tta"):
        _check_metric_dict(
            ablation.get(variant, {}),
            f"ablation.{variant}",
            ("f1", "precision", "recall"),
            problems,
        )
    calibration = ablation.get("calibration", {})
    if not isinstance(calibration, Mapping):
        problems.append("ablation.calibration must be an object")
    else:
        for key in ("temperature", "threshold"):
            if key not in calibration:
                problems.append(f"ablation.calibration.{key} missing")
            elif not _is_number(calibration[key]):
                problems.append(f"ablation.calibration.{key} must be finite")
    return problems


SWEEP_SPECS: Dict[str, SweepSpec] = {
    "multi": SweepSpec(
        name="multi",
        result_dir="results/multi",
        models=DEFAULT_MULTI_MODELS,
        seeds=DEFAULT_MULTI_SEEDS,
        validator=validate_multi_result,
    ),
    "shuffle": SweepSpec(
        name="shuffle",
        result_dir="results/shuffle_multi",
        models=DEFAULT_ABLATION_MODELS,
        seeds=DEFAULT_ABLATION_SEEDS,
        validator=validate_shuffle_result,
    ),
    "tpc_tta": SweepSpec(
        name="tpc_tta",
        result_dir="results/tpc_tta",
        models=DEFAULT_ABLATION_MODELS,
        seeds=DEFAULT_ABLATION_SEEDS,
        validator=validate_tpc_result,
    ),
}


def result_filename(dataset: str, model: str, seed: int) -> str:
    return f"{dataset}__{model}__seed{seed}.json"


def write_json_atomic(path: str, payload) -> None:
    """Write JSON through a same-directory temp file, then atomically replace."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix=f".{os.path.basename(path)}.",
        suffix=".tmp",
        dir=os.path.dirname(path) or ".",
        text=True,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
            fh.write("\n")
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _read_json(path: str):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _fingerprint_file(path: str) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    return size, digest.hexdigest()


def audit_sweep(
    sweep: str,
    result_dir: Optional[str] = None,
    datasets: Optional[Iterable[str]] = None,
    models: Optional[Iterable[str]] = None,
    seeds: Optional[Iterable[int]] = None,
) -> List[ArtifactAuditRow]:
    if sweep not in SWEEP_SPECS:
        raise ValueError(f"Unknown sweep '{sweep}'. Available: {sorted(SWEEP_SPECS)}")
    spec = SWEEP_SPECS[sweep]
    datasets = tuple(datasets or DEFAULT_DATASETS)
    models = tuple(models or spec.models)
    seeds = tuple(seeds or spec.seeds)
    result_dir = result_dir or spec.result_dir

    rows: List[ArtifactAuditRow] = []
    for dataset in datasets:
        for model in models:
            for seed in seeds:
                path = os.path.join(result_dir, result_filename(dataset, model, seed))
                if not os.path.exists(path):
                    rows.append(ArtifactAuditRow(
                        sweep, dataset, model, int(seed), path, "MISSING",
                        "expected result file is absent",
                    ))
                    continue
                size_bytes, sha256 = _fingerprint_file(path)
                try:
                    payload = _read_json(path)
                except json.JSONDecodeError as exc:
                    rows.append(ArtifactAuditRow(
                        sweep, dataset, model, int(seed), path, "INVALID_JSON",
                        str(exc), size_bytes, sha256,
                    ))
                    continue
                if not isinstance(payload, Mapping):
                    rows.append(ArtifactAuditRow(
                        sweep, dataset, model, int(seed), path, "INVALID_SCHEMA",
                        "top-level JSON must be an object", size_bytes, sha256,
                    ))
                    continue
                if "error" in payload:
                    rows.append(ArtifactAuditRow(
                        sweep, dataset, model, int(seed), path, "ERROR_RESULT",
                        str(payload.get("error", ""))[:240], size_bytes, sha256,
                    ))
                    continue
                identity_mismatches = []
                if payload.get("dataset") != dataset:
                    identity_mismatches.append(
                        f"dataset={payload.get('dataset')!r}, expected {dataset!r}"
                    )
                if payload.get("model") != model:
                    identity_mismatches.append(
                        f"model={payload.get('model')!r}, expected {model!r}"
                    )
                if payload.get("seed") != int(seed):
                    identity_mismatches.append(
                        f"seed={payload.get('seed')!r}, expected {int(seed)!r}"
                    )
                if identity_mismatches:
                    rows.append(ArtifactAuditRow(
                        sweep, dataset, model, int(seed), path, "INVALID_SCHEMA",
                        "; ".join(identity_mismatches), size_bytes, sha256,
                    ))
                    continue
                problems = spec.validator(payload)
                if problems:
                    rows.append(ArtifactAuditRow(
                        sweep, dataset, model, int(seed), path, "INVALID_SCHEMA",
                        "; ".join(problems[:8]), size_bytes, sha256,
                    ))
                else:
                    rows.append(ArtifactAuditRow(
                        sweep, dataset, model, int(seed), path, "COMPLETE",
                        "schema ok", size_bytes, sha256,
                    ))
    return rows


def summarize_rows(rows: Sequence[ArtifactAuditRow]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for row in rows:
        counts[row.status] = counts.get(row.status, 0) + 1
    return counts


def _escape_md(text: str) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def _short_hash(value: str) -> str:
    return value[:12] if value else ""


def write_audit_reports(
    rows: Sequence[ArtifactAuditRow],
    csv_path: str,
    md_path: str,
    title: str,
) -> None:
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=(
                "sweep",
                "dataset",
                "model",
                "seed",
                "status",
                "path",
                "detail",
                "size_bytes",
                "sha256",
            ),
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "sweep": row.sweep,
                "dataset": row.dataset,
                "model": row.model,
                "seed": row.seed,
                "status": row.status,
                "path": row.path,
                "detail": row.detail,
                "size_bytes": row.size_bytes,
                "sha256": row.sha256,
            })

    counts = summarize_rows(rows)
    total = len(rows)
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(f"# {title}\n\n")
        fh.write(
            "This manifest inspects expected result JSON files only. It does "
            "not train models, run GNNs, or fabricate missing artifacts.\n\n"
        )
        fh.write(f"- Expected files: {total}\n")
        for status in sorted(counts):
            fh.write(f"- {status}: {counts[status]}\n")
        fh.write("\n")
        fh.write("| sweep | dataset | model | seed | status | bytes | sha256 | path | detail |\n")
        fh.write("| --- | --- | --- | ---: | --- | ---: | --- | --- | --- |\n")
        for row in rows:
            fh.write(
                f"| {_escape_md(row.sweep)} | {_escape_md(row.dataset)} | "
                f"{_escape_md(row.model)} | {row.seed} | "
                f"{_escape_md(row.status)} | {row.size_bytes} | "
                f"`{_short_hash(row.sha256)}` | `{_escape_md(row.path)}` | "
                f"{_escape_md(row.detail)} |\n"
            )
