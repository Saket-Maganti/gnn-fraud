from __future__ import annotations
from fraudshiftbench.model_contracts import ModelContract, validate_model_contract

REGISTRY: dict[str, ModelContract] = {}

def register_model(contract: ModelContract) -> None:
    errors = validate_model_contract(contract)
    if errors:
        raise ValueError(errors)
    REGISTRY[contract.model_id] = contract

def get_model_contract(model_id: str) -> ModelContract:
    return REGISTRY[model_id]
