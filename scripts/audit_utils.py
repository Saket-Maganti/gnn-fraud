"""Lightweight utilities shared by dataset-free publication audits."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SECOND_DATASETS = ("dgraphfin", "tfinance")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: str | Path, text: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")


def second_dataset_results_exist(root: Path) -> bool:
    """Return true only for real imported second-dataset result artifacts."""

    runs_root = root / "results" / "runs"
    if not runs_root.is_dir():
        return False
    for dataset in SECOND_DATASETS:
        prediction_dir = runs_root / "predictions"
        if prediction_dir.is_dir() and any(prediction_dir.glob(f"{dataset}__*__seed*.csv")):
            return True
        if any(runs_root.glob(f"*/{dataset}__*__seed*.json")):
            return True
    return False
