"""
data/feature_scaling.py

Temporal-safe feature scaling helpers.

``train_only`` fits ``StandardScaler`` on train-mask rows only, then transforms
all nodes. ``full_population`` preserves the legacy Elliptic loader behaviour
(fit on every row before split). ``none`` skips scaling.
"""

from __future__ import annotations

from typing import Literal, Optional, Tuple

import numpy as np
from sklearn.preprocessing import StandardScaler

ScalerMode = Literal["train_only", "full_population", "none"]
SCALER_MODES: Tuple[str, ...] = ("train_only", "full_population", "none")


def resolve_scaler_mode(
    scaler_mode: Optional[str] = None,
    *,
    normalize: Optional[bool] = None,
    default: str = "train_only",
) -> str:
    """Resolve the active scaler mode with backward-compatible ``normalize``."""
    if scaler_mode is not None:
        if scaler_mode not in SCALER_MODES:
            raise ValueError(
                f"Unknown scaler_mode '{scaler_mode}'. "
                f"Choose from: {', '.join(SCALER_MODES)}"
            )
        return scaler_mode
    if normalize is not None:
        return "full_population" if normalize else "none"
    if default not in SCALER_MODES:
        raise ValueError(f"Invalid default scaler_mode: {default}")
    return default


def scale_features(
    feat_mat: np.ndarray,
    fit_mask: Optional[np.ndarray],
    scaler_mode: str,
) -> Tuple[np.ndarray, str]:
    """Scale ``feat_mat`` and return ``(scaled_matrix, scaler_mode)``."""
    if scaler_mode not in SCALER_MODES:
        raise ValueError(
            f"Unknown scaler_mode '{scaler_mode}'. "
            f"Choose from: {', '.join(SCALER_MODES)}"
        )

    x = np.asarray(feat_mat, dtype=np.float32)
    if scaler_mode == "none":
        return x, "none"

    scaler = StandardScaler()
    if scaler_mode == "full_population":
        scaler.fit(x)
    else:
        if fit_mask is None:
            raise ValueError("train_only scaling requires fit_mask")
        fit_mask = np.asarray(fit_mask, dtype=bool)
        if fit_mask.shape[0] != x.shape[0]:
            raise ValueError("fit_mask length must match number of rows in feat_mat")
        if not fit_mask.any():
            raise ValueError("train_only scaling requires a non-empty fit_mask")
        scaler.fit(x[fit_mask])

    return scaler.transform(x).astype(np.float32), scaler_mode
