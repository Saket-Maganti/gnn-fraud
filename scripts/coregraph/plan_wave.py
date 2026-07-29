#!/usr/bin/env python3
"""Create deterministic two-lane plans from a frozen CSV matrix."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", required=True)
    parser.add_argument("--include-blocked", action="store_true")
    args = parser.parse_args()
    path = ROOT / "configs/coregraph/run_matrices" / args.matrix
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    selected = [
        row for row in rows
        if args.include_blocked or row["execution_status"] == "PLANNED"
    ]
    output = {
        "schema": "coregraph_wave_plan_v1",
        "matrix": args.matrix,
        "matrix_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "heavy_execution_performed": False,
        "lanes": {
            "t4_lane_0": [row["run_key"] for row in selected[0::2]],
            "t4_lane_1": [row["run_key"] for row in selected[1::2]],
        },
        "excluded_rows": len(rows) - len(selected),
    }
    destination = ROOT / "results/coregraph_build" / f"WAVE_PLAN_{path.stem}.json"
    destination.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"planned": len(selected), "excluded": len(rows) - len(selected)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
