#!/usr/bin/env python3
"""Hash the preregistered analysis plan and all run matrices."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    paths = [ROOT / "configs/coregraph/analysis_families.yaml"]
    paths.extend(sorted((ROOT / "configs/coregraph/run_matrices").glob("*.csv")))
    payload = {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
    }
    aggregate = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    report = {
        "schema": "coregraph_analysis_freeze_v1",
        "aggregate_sha256": aggregate,
        "files": payload,
        "note": "Regenerate only before inspecting final-run outcomes.",
    }
    output = ROOT / "results/coregraph_build/ANALYSIS_PLAN_FREEZE.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(aggregate)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
