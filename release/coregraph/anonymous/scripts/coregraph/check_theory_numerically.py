#!/usr/bin/env python3
"""Run the deterministic theory audit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from coregraph.theory.numerical_checks import run_numerical_checks


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="results/coregraph_build/THEORY_NUMERICAL_CHECKS.json",
    )
    args = parser.parse_args()
    report = run_numerical_checks()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
