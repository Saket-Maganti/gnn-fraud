"""Executable entry point for Level-4 finite theory checks."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from coregraph.theory_checks import run_level4_theory_checks  # noqa: E402


def main() -> int:
    checks = run_level4_theory_checks()
    print(json.dumps({"checks": checks, "all_pass": all(checks.values())}, indent=2, sort_keys=True))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
