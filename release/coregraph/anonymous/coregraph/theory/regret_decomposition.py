"""Bookkeeping identity for Level-4 contract regret components."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class RegretDecomposition:
    representation_error: float
    diagnostic_error: float
    routing_approximation_error: float
    resource_availability_penalty: float
    budget_penalty: float
    abstention_penalty: float
    finite_sample_error: float

    def __post_init__(self) -> None:
        if any(value < 0 for value in asdict(self).values()):
            raise ValueError("regret decomposition terms must be non-negative")

    @property
    def upper_bound(self) -> float:
        return float(sum(asdict(self).values()))

    def observable_status(self) -> dict[str, str]:
        return {
            "representation_error": "THEORETICAL_OR_SYNTHETIC_DIAGNOSTIC",
            "diagnostic_error": "SOURCE_VALIDATION_ESTIMABLE",
            "routing_approximation_error": "SOURCE_OBJECTIVE_ESTIMABLE",
            "resource_availability_penalty": "OFFLINE_COUNTERFACTUAL_ESTIMABLE",
            "budget_penalty": "OFFLINE_COUNTERFACTUAL_ESTIMABLE",
            "abstention_penalty": "OFFLINE_EVALUATION_ESTIMABLE",
            "finite_sample_error": "INFERENCE_BOUND_OR_BOOTSTRAP_ESTIMATE",
        }
