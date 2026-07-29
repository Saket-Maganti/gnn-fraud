"""CoReGraph method composition."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from coregraph.contracts.contract import DeploymentContract
from coregraph.objectives.scores import ScoreType
from coregraph.routing.contract_encoder import (
    AtomicContractEncoder,
    FactorisedContractEncoder,
    NoContractEncoder,
)
from coregraph.routing.router import CoReRouter, RouterOutput


class CoReGraph(nn.Module):
    """Factorised contract encoder plus resource-aware CoReRouter."""

    def __init__(
        self,
        *,
        num_experts: int,
        diagnostic_dim: int,
        per_expert_diagnostic_dim: int = 0,
        contract_embedding_dim: int = 8,
        contract_output_dim: int = 32,
        router_hidden_dim: int = 64,
        router_mode: str = "mlp",
        expert_identity_dim: int = 8,
        expert_family_dim: int = 0,
        num_expert_families: int = 0,
        pairwise_interactions: bool = True,
        axis_dropout: float = 0.1,
        contract_noise_std: float = 0.01,
        contract_encoder_kind: str = "factorised",
        seen_contract_ids: Sequence[str] = (),
    ):
        super().__init__()
        if contract_encoder_kind == "factorised":
            self.contract_encoder: nn.Module = FactorisedContractEncoder(
                embedding_dim=contract_embedding_dim,
                output_dim=contract_output_dim,
                pairwise_interactions=pairwise_interactions,
                axis_dropout=axis_dropout,
                contract_noise_std=contract_noise_std,
            )
        elif contract_encoder_kind == "atomic":
            if not seen_contract_ids:
                raise ValueError("atomic contract encoder requires source IDs")
            self.contract_encoder = AtomicContractEncoder(
                seen_contract_ids,
                output_dim=contract_output_dim,
            )
        elif contract_encoder_kind == "none":
            self.contract_encoder = NoContractEncoder(
                output_dim=contract_output_dim,
            )
        else:
            raise ValueError(
                "contract_encoder_kind must be factorised atomic or none"
            )
        self.router = CoReRouter(
            num_experts=num_experts,
            contract_dim=contract_output_dim,
            diagnostic_dim=diagnostic_dim,
            per_expert_diagnostic_dim=per_expert_diagnostic_dim,
            expert_identity_dim=expert_identity_dim,
            expert_family_dim=expert_family_dim,
            num_expert_families=num_expert_families,
            hidden_dim=router_hidden_dim,
            mode=router_mode,
        )

    def forward(
        self,
        *,
        contracts: Sequence[DeploymentContract],
        expert_scores: torch.Tensor,
        score_type: ScoreType,
        shared_diagnostics: torch.Tensor,
        per_expert_diagnostics: torch.Tensor,
        availability_mask: torch.Tensor,
        expert_costs: torch.Tensor,
        expert_identity_indices: torch.Tensor | None = None,
        expert_family_indices: torch.Tensor | None = None,
        expert_names: Sequence[str] | None = None,
    ) -> RouterOutput:
        if len(contracts) != expert_scores.shape[0]:
            raise ValueError("one deployment contract is required per example")
        embedding = self.contract_encoder(contracts)
        return self.router(
            expert_scores=expert_scores,
            score_type=score_type,
            contract_embedding=embedding,
            shared_diagnostics=shared_diagnostics,
            per_expert_diagnostics=per_expert_diagnostics,
            availability_mask=availability_mask,
            expert_costs=expert_costs,
            expert_identity_indices=expert_identity_indices,
            expert_family_indices=expert_family_indices,
            expert_names=expert_names,
        )
