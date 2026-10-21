"""Atomic aligned prediction export."""

from __future__ import annotations

import csv
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from coregraph.experiments.hashing import sha256_file


def export_predictions(
    path: str | Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    required_columns: Sequence[str],
) -> str:
    if not rows:
        raise ValueError("prediction export cannot be empty")
    missing = [
        (index, sorted(set(required_columns) - set(row)))
        for index, row in enumerate(rows)
        if set(required_columns) - set(row)
    ]
    if missing:
        raise ValueError(f"prediction rows missing required columns: {missing[:3]}")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fields = list(required_columns)
    extras = sorted(set().union(*(row.keys() for row in rows)) - set(fields))
    fields.extend(extras)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        dir=target.parent,
        prefix=f".{target.name}.",
        delete=False,
    ) as handle:
        temp_path = Path(handle.name)
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            extrasaction="raise",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp_path, target)
    return sha256_file(target)
