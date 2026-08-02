"""Standard probabilistic calibration with explicit fit provenance."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss


def _prepare(
    labels: Sequence[int],
    probabilities: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    raw = np.asarray(labels).reshape(-1)
    p = np.asarray(probabilities, dtype=float).reshape(-1)
    if len(raw) != len(p):
        raise ValueError("labels and probabilities length mismatch")
    keep = raw != 0
    y = (raw[keep] == 1).astype(int)
    p = p[keep]
    if np.any((p < 0) | (p > 1)) or not np.isfinite(p).all():
        raise ValueError("probabilities must be finite in [0,1]")
    return y, p


@dataclass(frozen=True)
class LogisticCalibration:
    slope: float
    intercept: float
    n: int
    status: str


def logistic_calibration_slope_intercept(
    labels: Sequence[int],
    probabilities: Sequence[float],
) -> LogisticCalibration:
    y, p = _prepare(labels, probabilities)
    if len(y) < 3 or len(np.unique(y)) < 2:
        return LogisticCalibration(float("nan"), float("nan"), len(y), "DEGENERATE")
    eps = np.finfo(float).eps
    logit = np.log(np.clip(p, eps, 1 - eps) / np.clip(1 - p, eps, 1 - eps))
    if np.std(logit) < 1e-12:
        return LogisticCalibration(float("nan"), float("nan"), len(y), "DEGENERATE")
    model = LogisticRegression(
        penalty=None,
        solver="lbfgs",
        max_iter=10_000,
        fit_intercept=True,
    )
    model.fit(logit.reshape(-1, 1), y)
    return LogisticCalibration(
        slope=float(model.coef_[0, 0]),
        intercept=float(model.intercept_[0]),
        n=len(y),
        status="FIT",
    )


def brier_score(labels: Sequence[int], probabilities: Sequence[float]) -> float:
    y, p = _prepare(labels, probabilities)
    return float(brier_score_loss(y, p))


def expected_calibration_error(
    labels: Sequence[int],
    probabilities: Sequence[float],
    *,
    n_bins: int = 10,
    adaptive: bool = False,
) -> float:
    """Weighted absolute calibration gap using fixed-width or quantile bins."""

    if n_bins <= 0:
        raise ValueError("n_bins must be positive")
    y, p = _prepare(labels, probabilities)
    if len(y) == 0:
        return float("nan")
    if adaptive:
        edges = np.quantile(p, np.linspace(0, 1, n_bins + 1))
        edges[0], edges[-1] = 0.0, 1.0
        edges = np.maximum.accumulate(edges)
    else:
        edges = np.linspace(0, 1, n_bins + 1)
    total = 0.0
    for index in range(n_bins):
        keep = (p >= edges[index]) & (
            p <= edges[index + 1] if index == n_bins - 1 else p < edges[index + 1]
        )
        if keep.any():
            total += float(keep.mean()) * abs(float(p[keep].mean()) - float(y[keep].mean()))
    return float(total)


@dataclass(frozen=True)
class TemperatureScaler:
    temperature: float
    validation_nll: float

    def transform_logits(self, logits: np.ndarray) -> np.ndarray:
        return np.asarray(logits, dtype=float) / self.temperature


def fit_temperature(
    logits: np.ndarray,
    labels: Sequence[int],
) -> TemperatureScaler:
    values = np.asarray(logits, dtype=float)
    raw = np.asarray(labels).reshape(-1)
    keep = raw != 0
    y = (raw[keep] == 1).astype(int)
    z = values[keep]
    if z.ndim == 2:
        if z.shape[1] != 2:
            raise ValueError("temperature scaling supports binary two-logit arrays")
        margin = z[:, 1] - z[:, 0]
    else:
        margin = z.reshape(-1)

    def objective(log_temperature: float) -> float:
        temperature = float(np.exp(log_temperature))
        p = 1 / (1 + np.exp(-np.clip(margin / temperature, -50, 50)))
        return float(log_loss(y, p, labels=[0, 1]))

    result = minimize_scalar(objective, bounds=(-4, 4), method="bounded")
    temperature = float(np.exp(result.x))
    return TemperatureScaler(temperature, float(result.fun))


@dataclass
class IsotonicCalibrationAdapter:
    out_of_bounds: str = "clip"
    _model: IsotonicRegression | None = None

    def fit(
        self,
        labels: Sequence[int],
        probabilities: Sequence[float],
    ) -> "IsotonicCalibrationAdapter":
        y, p = _prepare(labels, probabilities)
        if len(np.unique(y)) < 2:
            raise ValueError("isotonic calibration requires both classes")
        self._model = IsotonicRegression(out_of_bounds=self.out_of_bounds)
        self._model.fit(p, y)
        return self

    def transform(self, probabilities: Sequence[float]) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("fit isotonic calibrator before transform")
        return np.asarray(self._model.predict(np.asarray(probabilities, dtype=float)))


def bootstrap_calibration_interval(
    labels: Sequence[int],
    probabilities: Sequence[float],
    *,
    metric: str = "brier",
    n_bootstrap: int = 1000,
    seed: int = 20260729,
) -> tuple[float, float]:
    y, p = _prepare(labels, probabilities)
    if len(y) < 2:
        return (float("nan"), float("nan"))
    metric_fn = brier_score_loss if metric == "brier" else None
    if metric_fn is None:
        raise ValueError("only the Brier interval is currently defensible")
    rng = np.random.default_rng(seed)
    values = np.empty(n_bootstrap)
    for index in range(n_bootstrap):
        sample = rng.integers(0, len(y), len(y))
        values[index] = metric_fn(y[sample], p[sample])
    low, high = np.quantile(values, [0.025, 0.975])
    return float(low), float(high)
