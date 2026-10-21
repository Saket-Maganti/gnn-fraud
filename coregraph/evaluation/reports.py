"""Machine-readable evaluation report assembly."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


def evaluation_report(
    *,
    run_id: str,
    contract_metrics: Mapping[str, Mapping[str, float]],
    statistical_rows: Sequence[Mapping[str, Any]],
    blocked_cells: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": "coregraph_evaluation_v2",
        "run_id": run_id,
        "contract_metrics": {
            key: dict(sorted(value.items())) for key, value in sorted(contract_metrics.items())
        },
        "statistical_rows": [dict(row) for row in statistical_rows],
        "blocked_cells": [
            {**dict(row), "predictive_ordering": "PROHIBITED"} for row in blocked_cells
        ],
    }
