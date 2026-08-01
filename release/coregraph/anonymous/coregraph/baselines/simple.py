"""Strong simple routing baselines under a common feasible set."""

from __future__ import annotations

import numpy as np


def _validate(scores: np.ndarray, feasible: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(scores, dtype=float)
    mask = np.asarray(feasible, dtype=bool)
    if values.ndim != 2 or mask.shape not in {values.shape, (values.shape[1],)}:
        raise ValueError("scores must be [instances,experts] with aligned feasibility")
    if mask.ndim == 1:
        mask = np.broadcast_to(mask, values.shape)
    return values, mask


def uniform_average(scores: np.ndarray, feasible: np.ndarray) -> np.ndarray:
    values, mask = _validate(scores, feasible)
    count = mask.sum(axis=1)
    return np.divide((values * mask).sum(axis=1), count, out=np.full(len(values), np.nan), where=count > 0)


def validation_weighted_average(
    scores: np.ndarray,
    feasible: np.ndarray,
    source_validation_risk: np.ndarray,
) -> np.ndarray:
    values, mask = _validate(scores, feasible)
    risk = np.asarray(source_validation_risk, dtype=float)
    if risk.shape != (values.shape[1],) or np.any(risk < 0):
        raise ValueError("source validation risk must have one non-negative value per expert")
    quality = 1 / np.clip(risk, 1e-12, None)
    weights = mask * quality[None, :]
    weights = np.divide(weights, weights.sum(axis=1, keepdims=True), out=np.zeros_like(weights), where=weights.sum(axis=1, keepdims=True) > 0)
    output = (weights * values).sum(axis=1)
    output[~mask.any(axis=1)] = np.nan
    return output


def best_fixed_expert(
    scores: np.ndarray,
    feasible: np.ndarray,
    source_validation_risk: np.ndarray,
) -> np.ndarray:
    values, mask = _validate(scores, feasible)
    risk = np.asarray(source_validation_risk, dtype=float)
    if risk.shape != (values.shape[1],):
        raise ValueError("source validation risk must align with experts")
    ordered = np.argsort(risk, kind="stable")
    output = np.full(len(values), np.nan)
    for row in range(len(values)):
        candidates = [expert for expert in ordered if mask[row, expert]]
        if candidates:
            output[row] = values[row, candidates[0]]
    return output


def confidence_selection(scores: np.ndarray, feasible: np.ndarray) -> np.ndarray:
    values, mask = _validate(scores, feasible)
    confidence = np.abs(values - 0.5)
    chosen = np.where(mask, confidence, -np.inf).argmax(axis=1)
    output = values[np.arange(len(values)), chosen]
    output[~mask.any(axis=1)] = np.nan
    return output


def cheapest_feasible_expert(scores: np.ndarray, feasible: np.ndarray, costs: np.ndarray) -> np.ndarray:
    values, mask = _validate(scores, feasible)
    cost = np.asarray(costs, dtype=float)
    if cost.shape != (values.shape[1],) or np.any(cost < 0):
        raise ValueError("costs must be non-negative and align with experts")
    chosen = np.where(mask, cost[None, :], np.inf).argmin(axis=1)
    output = values[np.arange(len(values)), chosen]
    output[~mask.any(axis=1)] = np.nan
    return output


def random_feasible_expert(
    scores: np.ndarray,
    feasible: np.ndarray,
    *,
    seed: int,
) -> np.ndarray:
    values, mask = _validate(scores, feasible)
    rng = np.random.default_rng(seed)
    output = np.full(len(values), np.nan)
    for row in range(len(values)):
        candidates = np.flatnonzero(mask[row])
        if len(candidates):
            output[row] = values[row, int(rng.choice(candidates))]
    return output


def oracle_diagnostics(
    scores: np.ndarray,
    labels: np.ndarray,
    feasible: np.ndarray,
) -> dict[str, np.ndarray | int]:
    """Offline-only Brier oracles; never a deployable baseline."""

    values, mask = _validate(scores, feasible)
    targets = np.asarray(labels, dtype=float).reshape(-1)
    if len(targets) != len(values):
        raise ValueError("oracle labels must align with prediction rows")
    losses = (values - targets[:, None]) ** 2
    contract_feasible = mask.all(axis=0)
    if not contract_feasible.any():
        raise ValueError("contract oracle requires one expert feasible for every row")
    contract_risk = losses.mean(axis=0)
    contract_expert = int(np.where(contract_feasible, contract_risk, np.inf).argmin())
    instance = np.where(mask, losses, np.inf).argmin(axis=1)
    return {
        "contract_oracle_expert": contract_expert,
        "contract_oracle_scores": values[:, contract_expert],
        "instance_oracle_experts": instance,
        "instance_oracle_scores": values[np.arange(len(values)), instance],
    }


class SourceLogisticGate:
    """Small deterministic gate fitted only on source validation rows."""

    def __init__(self, learning_rate: float = 0.1, steps: int = 200) -> None:
        self.learning_rate = learning_rate
        self.steps = steps
        self.weights: np.ndarray | None = None

    def fit(self, features: np.ndarray, best_expert: np.ndarray) -> "SourceLogisticGate":
        x = np.asarray(features, dtype=float)
        y = np.asarray(best_expert, dtype=int)
        if x.ndim != 2 or y.shape != (len(x),) or np.any(y < 0):
            raise ValueError("source gate features and expert labels must align")
        classes = int(y.max()) + 1
        weights = np.zeros((x.shape[1] + 1, classes))
        augmented = np.column_stack((x, np.ones(len(x))))
        targets = np.eye(classes)[y]
        for _ in range(self.steps):
            logits = augmented @ weights
            logits -= logits.max(axis=1, keepdims=True)
            probability = np.exp(logits)
            probability /= probability.sum(axis=1, keepdims=True)
            weights -= self.learning_rate * augmented.T @ (probability - targets) / len(x)
        self.weights = weights
        return self

    def predict_weights(self, features: np.ndarray, feasible: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise RuntimeError("source logistic gate must be fitted first")
        x = np.asarray(features, dtype=float)
        mask = np.asarray(feasible, dtype=bool)
        if mask.ndim == 1:
            mask = np.broadcast_to(mask, (len(x), len(mask)))
        logits = np.column_stack((x, np.ones(len(x)))) @ self.weights
        logits = np.where(mask, logits, -np.inf)
        no_feasible = ~mask.any(axis=1)
        logits[no_feasible] = 0
        logits -= logits.max(axis=1, keepdims=True)
        weights = np.exp(logits) * mask
        return np.divide(weights, weights.sum(axis=1, keepdims=True), out=np.zeros_like(weights), where=weights.sum(axis=1, keepdims=True) > 0)
