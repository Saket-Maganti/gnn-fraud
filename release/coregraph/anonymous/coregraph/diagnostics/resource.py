"""Resource availability, latency, memory, and confidence diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

from coregraph.diagnostics.specification import DiagnosticDeclaration, MissingSemantics


DECLARATION = DiagnosticDeclaration(
    name="expert_resource",
    required_inputs=("availability", "latency", "memory", "measurement_status"),
    target_labels_required=False,
    source_fitting="NONE",
    missing_semantics=MissingSemantics.NAN_WITH_MASK,
    computational_cost="O(experts)",
)


@dataclass(frozen=True)
class ExpertResourceDiagnostic:
    expert_id: str
    available: bool
    latency_ms: float | None
    memory_gb: float | None
    invocation_cost: float | None
    diagnostic_confidence: float
    measurement_status: str

    def __post_init__(self) -> None:
        values = (self.latency_ms, self.memory_gb, self.invocation_cost)
        if any(value is not None and value < 0 for value in values):
            raise ValueError("resource quantities cannot be negative")
        if not 0 <= self.diagnostic_confidence <= 1:
            raise ValueError("diagnostic confidence must lie in [0,1]")
