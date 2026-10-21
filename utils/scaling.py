"""
Leakage-safe feature scaling helpers.

``train_only`` fits ``StandardScaler`` on training rows only, then transforms
all rows. ``full_population`` fits on every row and is included only for
explicit legacy/reproduction use. ``none`` returns the original object.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from sklearn.preprocessing import StandardScaler

try:  # Torch support is optional for callers that already use tensors.
    import torch
except Exception:  # noqa: BLE001
    torch = None  # type: ignore[assignment]


SCALER_MODES = ("train_only", "full_population", "none")


def _is_torch_tensor(value: Any) -> bool:
    return torch is not None and torch.is_tensor(value)


def _as_numpy_mask(train_mask: Any, n_rows: int) -> np.ndarray:
    if train_mask is None:
        raise ValueError("train_only scaling requires train_mask")
    if _is_torch_tensor(train_mask):
        mask = train_mask.detach().cpu().numpy()
    else:
        mask = np.asarray(train_mask)
    mask = mask.astype(bool)
    if mask.ndim != 1:
        raise ValueError("train_mask must be one-dimensional")
    if mask.shape[0] != n_rows:
        raise ValueError("train_mask length must match number of feature rows")
    if not mask.any():
        raise ValueError("train_only scaling requires a non-empty train_mask")
    return mask


def _to_numpy(features: Any) -> np.ndarray:
    if _is_torch_tensor(features):
        return features.detach().cpu().numpy()
    return np.asarray(features)


def _restore_type(scaled: np.ndarray, original: Any) -> Any:
    if _is_torch_tensor(original):
        dtype = original.dtype if original.dtype.is_floating_point else torch.float32
        return torch.as_tensor(scaled, dtype=dtype, device=original.device)
    if isinstance(original, np.ndarray) and np.issubdtype(original.dtype, np.floating):
        return scaled.astype(original.dtype, copy=False)
    return scaled


def fit_transform_features_by_mode(
    features: Any,
    train_mask: Optional[Any] = None,
    scaler_mode: str = "train_only",
) -> Any:
    """Scale feature rows according to an explicit leakage policy.

    Parameters
    ----------
    features:
        A two-dimensional numpy array. Torch tensors are also accepted and are
        returned as tensors on the original device/dtype when possible.
    train_mask:
        Boolean mask identifying training rows. Required for ``train_only``.
    scaler_mode:
        One of ``train_only``, ``full_population``, or ``none``.
    """
    if scaler_mode not in SCALER_MODES:
        raise ValueError(
            f"Unknown scaler_mode '{scaler_mode}'. "
            f"Choose from: {', '.join(SCALER_MODES)}"
        )
    if scaler_mode == "none":
        return features

    x = _to_numpy(features)
    if x.ndim != 2:
        raise ValueError("features must be a two-dimensional array")

    scaler = StandardScaler()
    if scaler_mode == "train_only":
        mask = _as_numpy_mask(train_mask, x.shape[0])
        scaler.fit(x[mask])
    else:
        scaler.fit(x)

    scaled = scaler.transform(x)
    return _restore_type(scaled, features)
