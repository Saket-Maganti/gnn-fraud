"""Evidence badge definitions for FraudShiftBench."""

from __future__ import annotations

from typing import Dict

BADGES: Dict[str, Dict[str, str]] = {
    "SUPPORTED_10SEED": {
        "level": "supported",
        "label": "Supported 10-seed",
        "meaning": "Validated 10-seed result artifacts exist for the stated family.",
        "claim_scope": "Use only for the named dataset/protocol/model family.",
    },
    "SUPPORTED_DIAGNOSTIC": {
        "level": "diagnostic",
        "label": "Supported diagnostic",
        "meaning": "Diagnostic analysis is backed by existing artifacts.",
        "claim_scope": "Do not phrase as causal proof or deployment guarantee.",
    },
    "FDR_SENSITIVITY_ONLY": {
        "level": "sensitivity",
        "label": "FDR sensitivity only",
        "meaning": "Effect survives FDR/BH sensitivity but not Holm correction.",
        "claim_scope": "Exploratory or sensitivity language only.",
    },
    "SCAFFOLD_ONLY": {
        "level": "scaffold",
        "label": "Scaffold only",
        "meaning": "Code/design exists, but no empirical result is claimed.",
        "claim_scope": "Future-work or protocol-design language only.",
    },
    "PENDING_RB09": {
        "level": "blocked",
        "label": "Pending RB09",
        "meaning": "RB09/DGraphFin/T-Finance result import is not validated here.",
        "claim_scope": "No second-dataset result claim.",
    },
    "BLOCKED_NO_DATA": {
        "level": "blocked",
        "label": "Blocked no data",
        "meaning": "Required dataset artifact is absent or unvalidated.",
        "claim_scope": "No result claim.",
    },
    "RESOURCE_LIMITED": {
        "level": "diagnostic",
        "label": "Resource limited",
        "meaning": "Resource feasibility caveat exists for the model/dataset.",
        "claim_scope": "Feasibility evidence only, not failed science.",
    },
}


def badge_catalog() -> Dict[str, Dict[str, str]]:
    """Return a copy of the badge catalog."""

    return {key: value.copy() for key, value in BADGES.items()}


def badge_for_evidence(level: str) -> str:
    """Best default badge for an evidence level."""

    normalized = str(level).strip().lower()
    if normalized == "supported":
        return "SUPPORTED_10SEED"
    if normalized == "diagnostic":
        return "SUPPORTED_DIAGNOSTIC"
    if normalized == "sensitivity":
        return "FDR_SENSITIVITY_ONLY"
    if normalized == "scaffold":
        return "SCAFFOLD_ONLY"
    return "BLOCKED_NO_DATA"
