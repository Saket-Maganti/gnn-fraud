from __future__ import annotations
from dataclasses import asdict, dataclass

@dataclass(frozen=True)
class ModelContract:
    model_id: str
    model_family: str
    uses_graph: bool
    uses_edge_features: bool
    uses_node_features: bool
    uses_temporal_features: bool
    uses_future_information_allowed: bool
    requires_gpu: bool
    supports_prediction_export: bool
    supports_hidden_eval: bool
    supported_protocols: list[str]
    resource_profile: dict
    hyperparameter_defaults: dict
    known_failure_modes: list[str]
    claim_scope: str

    def to_dict(self) -> dict:
        return asdict(self)

def validate_model_contract(contract: ModelContract) -> list[str]:
    errors = []
    if contract.uses_future_information_allowed:
        errors.append("future_information_not_allowed")
    if not contract.supports_prediction_export:
        errors.append("prediction_export_required")
    if not contract.supported_protocols:
        errors.append("supported_protocols_required")
    if not contract.resource_profile:
        errors.append("resource_profile_required")
    if not contract.claim_scope:
        errors.append("claim_scope_required")
    return errors
