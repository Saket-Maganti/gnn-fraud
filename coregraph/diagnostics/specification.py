"""Common declaration carried by every Level-4 diagnostic."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class MissingSemantics(str, Enum):
    ZERO_WITH_MASK = "ZERO_WITH_MASK"
    NAN_WITH_MASK = "NAN_WITH_MASK"
    NOT_APPLICABLE = "NOT_APPLICABLE"


@dataclass(frozen=True)
class DiagnosticDeclaration:
    name: str
    required_inputs: tuple[str, ...]
    target_labels_required: bool
    source_fitting: str
    missing_semantics: MissingSemantics
    computational_cost: str
    confidence_field: bool = True

    def assert_deployable(self) -> None:
        if self.target_labels_required:
            raise ValueError(f"diagnostic {self.name} is offline-only")
