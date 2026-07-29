#!/usr/bin/env python3
"""Static validation for generated orchestration notebooks."""

from __future__ import annotations

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    paths = sorted((ROOT / "kaggle/coregraph").glob("*.ipynb"))
    paths += sorted((ROOT / "notebooks/coregraph").glob("*.ipynb"))
    failures: list[str] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("nbformat") != 4:
            failures.append(f"{path}:nbformat")
        metadata = payload.get("metadata", {}).get("coregraph", {})
        if metadata.get("heavy_execution_default") is not False:
            failures.append(f"{path}:execution_default")
        if path.parts[-3:-2] == ("kaggle",) and metadata.get("accelerator_envelope") != "T4x2":
            failures.append(f"{path}:accelerator")
        for index, cell in enumerate(payload.get("cells", [])):
            if cell.get("cell_type") == "code":
                source = "".join(cell.get("source", []))
                try:
                    ast.parse(source)
                except SyntaxError as exc:
                    failures.append(f"{path}:cell{index}:{exc}")
    if len(paths) != 12:
        failures.append(f"notebook_count:{len(paths)}!=12")
    report = {
        "schema": "coregraph_notebook_validation_v1",
        "notebooks": len(paths),
        "failures": failures,
        "status": "PASS" if not failures else "FAIL",
    }
    output = ROOT / "results/coregraph_build/NOTEBOOK_VALIDATION.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
