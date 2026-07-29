"""Minimal model plug-in API for FraudShiftBench external models."""

from __future__ import annotations

from typing import Any, Protocol


class FraudShiftBenchModel(Protocol):
    def fit(self, train_data: Any, validation_data: Any | None = None) -> "FraudShiftBenchModel":
        ...

    def predict_proba(self, eval_data: Any) -> Any:
        ...

    def export_predictions(self, path: str, eval_data: Any) -> None:
        ...

    def metadata(self) -> dict[str, Any]:
        ...

    def supported_protocols(self) -> list[str]:
        ...

    def required_inputs(self) -> list[str]:
        ...
