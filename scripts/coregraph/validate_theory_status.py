#!/usr/bin/env python3
"""Gate theorem wording on explicit proof status and numerical checks."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from coregraph.theory.numerical_checks import run_numerical_checks  # noqa: E402


def main() -> int:
    status_path = ROOT / "paper_iclr/theory/THEOREM_STATUS.yaml"
    payload = yaml.safe_load(status_path.read_text(encoding="utf-8"))
    numerical = run_numerical_checks()
    failures: list[str] = []
    for name, record in payload["results"].items():
        if record["status"] == "PROVED":
            for field in ("statement", "implementation", "numerical_check", "scope"):
                if not record.get(field):
                    failures.append(f"{name}:missing:{field}")
            statement = ROOT / "paper_iclr" / record["statement"]
            implementation = ROOT / record["implementation"]
            if not statement.is_file() or not implementation.is_file():
                failures.append(f"{name}:missing_source")
            check = numerical.get(record["numerical_check"], {})
            if check.get("pass") is not True:
                failures.append(f"{name}:numerical_check")
    report = {
        "schema": "coregraph_theory_gate_v1",
        "status": "PASS" if not failures else "FAIL",
        "proved_results": sum(
            record["status"] == "PROVED" for record in payload["results"].values()
        ),
        "failures": failures,
        "prohibited_promotions": payload["prohibited_promotions"],
    }
    output = ROOT / "results/coregraph_build/THEORY_STATUS_GATE.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
