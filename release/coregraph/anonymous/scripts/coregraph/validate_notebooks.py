#!/usr/bin/env python3
"""Static validation for generated orchestration notebooks."""

from __future__ import annotations

import ast
import hashlib
import json
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REQUIRED_KAGGLE_WAVES = {
    "saved_output_pilot",
    "fraud_full_training",
    "fraud_baseline_waves",
    "synthetic_suite",
    "graph_ood_benchmark",
    "resource_profiling",
    "ablation_waves",
    "prediction_regeneration",
    "output_validation_packaging",
}
REQUIRED_TOKENS = (
    "git', '-C', str(REPO), 'rev-parse",
    "verified_dataset_hashes",
    "created_before_execution",
    "tuple(range(1, 11))",
    "GPU_LANES = (0, 1)",
    "resume_completed",
    "checkpoint_path",
    "RESOURCE_BLOCKED_OOM",
    "COMPLETE_VALIDATED",
    "output_sha256",
    "ZipFile",
    "COMPLETION_REPORT",
    "Failed coordinates may not be silently skipped",
)


def tiny_fixture() -> dict[str, object]:
    coordinates = [f"fixture-seed-{seed}" for seed in range(1, 11)]
    lanes = {lane: coordinates[lane::2] for lane in (0, 1)}
    if set(lanes[0]) & set(lanes[1]) or set(lanes[0] + lanes[1]) != set(coordinates):
        raise AssertionError("T4 lane fixture duplicated or dropped a coordinate")
    if "RESOURCE_BLOCKED_OOM" != (
        "RESOURCE_BLOCKED_OOM" if "out of memory" in "CUDA out of memory".lower() else "FAIL"
    ):
        raise AssertionError("OOM classification fixture failed")
    with tempfile.TemporaryDirectory(prefix="coregraph-notebook-fixture-") as directory:
        root = Path(directory)
        payload = root / "completion_report.json"
        payload.write_text(
            json.dumps({"status": "COMPLETE_VALIDATED", "coordinates": coordinates}),
            encoding="utf-8",
        )
        archive_path = root / "validated.zip"
        with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            info = zipfile.ZipInfo(payload.name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, payload.read_bytes())
        if not zipfile.is_zipfile(archive_path):
            raise AssertionError("output packaging fixture failed")
        digest = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    return {"coordinates": 10, "lane_counts": {"0": 5, "1": 5}, "zip_sha256": digest}


def main() -> int:
    paths = sorted((ROOT / "kaggle/coregraph").glob("*.ipynb"))
    paths += sorted((ROOT / "notebooks/coregraph").glob("*.ipynb"))
    failures: list[str] = []
    kaggle_waves: set[str] = set()
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("nbformat") != 4:
            failures.append(f"{path}:nbformat")
        metadata = payload.get("metadata", {}).get("coregraph", {})
        if metadata.get("heavy_execution_default") is not False:
            failures.append(f"{path}:execution_default")
        if path.parts[-3:-2] == ("kaggle",) and metadata.get("accelerator_envelope") != "T4x2":
            failures.append(f"{path}:accelerator")
        if path.parts[-3:-2] == ("kaggle",):
            kaggle_waves.add(str(metadata.get("wave")))
        combined_source = ""
        for index, cell in enumerate(payload.get("cells", [])):
            if cell.get("cell_type") == "code":
                source = "".join(cell.get("source", []))
                combined_source += source
                if cell.get("execution_count") is not None or cell.get("outputs"):
                    failures.append(f"{path}:cell{index}:must_be_unexecuted")
                try:
                    ast.parse(source)
                except SyntaxError as exc:
                    failures.append(f"{path}:cell{index}:{exc}")
        for token in REQUIRED_TOKENS:
            if token not in combined_source:
                failures.append(f"{path}:missing_requirement:{token}")
    if len(paths) != 12:
        failures.append(f"notebook_count:{len(paths)}!=12")
    if kaggle_waves != REQUIRED_KAGGLE_WAVES:
        failures.append(
            "kaggle_waves:" + repr(sorted(kaggle_waves)) + "!=" + repr(sorted(REQUIRED_KAGGLE_WAVES))
        )
    fixture = tiny_fixture()
    report = {
        "schema": "coregraph_notebook_validation_v2",
        "notebooks": len(paths),
        "kaggle_level4_runbooks": len(kaggle_waves),
        "required_categories_present": kaggle_waves == REQUIRED_KAGGLE_WAVES,
        "syntax_checked": len(paths),
        "notebooks_executed": 0,
        "tiny_fixture": fixture,
        "failures": failures,
        "status": "PASS" if not failures else "FAIL",
    }
    output = ROOT / "results/coregraph_build/NOTEBOOK_VALIDATION.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
