"""Feasible-oracle monotonicity under resource masks."""

from __future__ import annotations

import numpy as np


def feasible_oracle(risks: np.ndarray, mask: np.ndarray) -> float:
    values = np.asarray(risks, dtype=float)
    feasible = np.asarray(mask, dtype=bool)
    if values.shape != feasible.shape or not feasible.any():
        raise ValueError("aligned risks and at least one feasible expert required")
    return float(values[feasible].min())


def resource_mask_monotonicity(
    risks: np.ndarray,
    broad_mask: np.ndarray,
    restricted_mask: np.ndarray,
) -> bool:
    broad = np.asarray(broad_mask, dtype=bool)
    restricted = np.asarray(restricted_mask, dtype=bool)
    if np.any(restricted & ~broad):
        raise ValueError("restricted mask must be a subset of broad mask")
    return feasible_oracle(risks, restricted) >= feasible_oracle(risks, broad)
