"""CoReGraph: contract-aware graph expert routing.

This package is intentionally separate from the frozen FraudShiftBench
implementation. New scientific evidence must use these typed contracts, task
adapters, graph views, and hashed runners.
"""

from importlib import import_module
from typing import Any

__all__ = [
    "CoReGraph",
    "CoReRouter",
    "DeploymentContract",
    "EvidenceUnitV2",
    "SupportEngine",
    "SupportStatus",
    "TypedClaim",
]

__version__ = "0.1.0"

_LAZY_EXPORTS = {
    "CoReGraph": ("coregraph.method", "CoReGraph"),
    "CoReRouter": ("coregraph.routing.router", "CoReRouter"),
    "DeploymentContract": ("coregraph.contracts.contract", "DeploymentContract"),
    "EvidenceUnitV2": ("coregraph.evidence", "EvidenceUnitV2"),
    "SupportEngine": ("coregraph.evidence", "SupportEngine"),
    "SupportStatus": ("coregraph.evidence", "SupportStatus"),
    "TypedClaim": ("coregraph.evidence", "TypedClaim"),
}


def __getattr__(name: str) -> Any:
    """Load public scientific objects only when callers request them."""

    try:
        module_name, attribute = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
