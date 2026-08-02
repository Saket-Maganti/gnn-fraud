"""Selective metrics with declared zero-coverage behavior."""

from __future__ import annotations

import numpy as np


def selective_metrics(losses: np.ndarray, abstain: np.ndarray) -> dict[str, float | str]:
    values = np.asarray(losses, dtype=float).reshape(-1)
    decisions = np.asarray(abstain, dtype=bool).reshape(-1)
    if values.shape != decisions.shape or not len(values):
        raise ValueError("selective metrics require non-empty aligned arrays")
    accepted = ~decisions
    coverage = float(accepted.mean())
    if not accepted.any():
        return {
            "coverage": 0.0,
            "selective_risk": float("nan"),
            "selective_risk_status": "NOT_APPLICABLE_ZERO_COVERAGE",
            "abstention_rate": 1.0,
        }
    return {
        "coverage": coverage,
        "selective_risk": float(values[accepted].mean()),
        "selective_risk_status": "MEASURED",
        "abstention_rate": float(decisions.mean()),
    }
