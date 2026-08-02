"""Partially observed deployment-contract schema for Level-4 routing."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping


CONTRACT_AXES = ("time", "visibility", "construction", "selection", "budget", "resource")


class ObservationState(str, Enum):
    OBSERVED = "observed"
    MISSING = "missing"
    UNCERTAIN = "uncertain"
    OUT_OF_RANGE = "out_of_range"
    UNSEEN = "unseen"


@dataclass(frozen=True)
class AxisObservation:
    categorical: str | None = None
    continuous: float | None = None
    state: ObservationState = ObservationState.OBSERVED
    confidence: float = 1.0

    def __post_init__(self) -> None:
        supplied = int(self.categorical is not None) + int(self.continuous is not None)
        if supplied > 1:
            raise ValueError("an axis observation cannot be categorical and continuous")
        if self.state is ObservationState.OBSERVED and supplied != 1:
            raise ValueError("an observed axis requires exactly one value")
        if self.state is ObservationState.MISSING and supplied:
            raise ValueError("a missing axis cannot carry a value")
        if not 0 <= self.confidence <= 1:
            raise ValueError("axis confidence must lie in [0,1]")
        if self.state is ObservationState.UNCERTAIN and self.confidence >= 1:
            raise ValueError("an uncertain axis must have confidence below one")
        if self.categorical is not None and not self.categorical.strip():
            raise ValueError("categorical axis values cannot be blank")


@dataclass(frozen=True)
class ContractObservation:
    axes: Mapping[str, AxisObservation]
    contract_id: str = "unknown"

    def __post_init__(self) -> None:
        if set(self.axes) != set(CONTRACT_AXES):
            missing = sorted(set(CONTRACT_AXES) - set(self.axes))
            extra = sorted(set(self.axes) - set(CONTRACT_AXES))
            raise ValueError(f"contract axes must be exact; missing={missing}; extra={extra}")

    @property
    def incomplete(self) -> bool:
        return any(axis.state is not ObservationState.OBSERVED for axis in self.axes.values())

    def manifest(self) -> dict[str, object]:
        return {
            "contract_id": self.contract_id,
            "axes": {
                name: {
                    "categorical": value.categorical,
                    "continuous": value.continuous,
                    "state": value.state.value,
                    "confidence": value.confidence,
                }
                for name, value in self.axes.items()
            },
        }
