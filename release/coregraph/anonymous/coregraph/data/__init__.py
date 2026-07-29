"""Leakage-safe dataset and graph-view adapters.

Import concrete adapters from their modules so task adapters can type-reference
graph views without a package initialisation cycle.
"""

from coregraph.data.leakage import LeakageError, audit_temporal_experiment

__all__ = ["LeakageError", "audit_temporal_experiment"]
