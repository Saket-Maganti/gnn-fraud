"""
Run metadata helpers for heavy experiment outputs.

The functions here are deliberately lightweight and side-effect free. They add
provenance to future JSON artifacts without changing any training logic or
metric values.
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Optional


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _git(cmd: list[str]) -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["git", *cmd],
            cwd=REPO_ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except Exception:  # noqa: BLE001 - provenance should never break training
        return None
    return out.strip()


def git_metadata() -> Dict[str, Any]:
    status = _git(["status", "--short"])
    return {
        "commit": _git(["rev-parse", "HEAD"]),
        "commit_short": _git(["rev-parse", "--short", "HEAD"]),
        "branch": _git(["branch", "--show-current"]),
        "dirty": bool(status),
    }


def build_run_metadata(
    runner: str,
    config: Optional[Mapping[str, Any]] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    meta: Dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "runner": runner,
        "argv": sys.argv,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cwd": os.getcwd(),
        "git": git_metadata(),
    }
    if config is not None:
        meta["config"] = dict(config)
        try:
            from utils.reproducibility import config_hash

            meta["config_hash"] = config_hash(config)
        except Exception:  # noqa: BLE001 - provenance should never break a run
            pass
    try:
        from utils.reproducibility import determinism_report

        meta["determinism"] = determinism_report()
    except Exception:  # noqa: BLE001
        pass
    if extra:
        meta.update(dict(extra))
    return meta
