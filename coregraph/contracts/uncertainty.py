"""Source-only preprocessing for uncertain and out-of-support axes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from coregraph.contracts.schema import CONTRACT_AXES, ContractObservation


@dataclass(frozen=True)
class SourceAxisStatistics:
    means: tuple[float, ...]
    scales: tuple[float, ...]
    minima: tuple[float, ...]
    maxima: tuple[float, ...]

    @classmethod
    def fit(cls, contracts: Sequence[ContractObservation]) -> "SourceAxisStatistics":
        if not contracts:
            raise ValueError("source statistics require at least one contract")
        columns: list[list[float]] = [[] for _ in CONTRACT_AXES]
        for contract in contracts:
            for index, name in enumerate(CONTRACT_AXES):
                value = contract.axes[name].continuous
                if value is not None:
                    columns[index].append(float(value))
        means, scales, minima, maxima = [], [], [], []
        for values in columns:
            array = np.asarray(values or [0.0], dtype=float)
            means.append(float(array.mean()))
            scales.append(float(array.std()) if float(array.std()) > 1e-12 else 1.0)
            minima.append(float(array.min()))
            maxima.append(float(array.max()))
        return cls(tuple(means), tuple(scales), tuple(minima), tuple(maxima))

    def normalize(self, axis_index: int, value: float) -> tuple[float, bool]:
        normalized = (float(value) - self.means[axis_index]) / self.scales[axis_index]
        outside = value < self.minima[axis_index] or value > self.maxima[axis_index]
        return normalized, outside
