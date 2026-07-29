"""CoReGraph: contract-aware graph expert routing.

This package is intentionally separate from the frozen FraudShiftBench
implementation. New scientific evidence must use these typed contracts, task
adapters, graph views, and hashed runners.
"""

from coregraph.contracts.contract import DeploymentContract
from coregraph.evidence import EvidenceUnitV2, SupportEngine, SupportStatus, TypedClaim
from coregraph.method import CoReGraph
from coregraph.routing.router import CoReRouter

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
