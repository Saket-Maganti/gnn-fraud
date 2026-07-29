"""Small artifact-loading helpers for FraudShiftBench CLIs."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import pandas as pd


def parse_prediction_filename(path: Path) -> Optional[Tuple[str, str, str, int]]:
    parts = path.stem.split("__")
    if len(parts) != 4 or not parts[3].startswith("seed"):
        return None
    try:
        seed = int(parts[3][len("seed") :])
    except ValueError:
        return None
    return parts[0], parts[1], parts[2], seed


def load_csv_if_exists(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.is_file() else pd.DataFrame()
