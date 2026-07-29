"""CoReGraph method composition."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from coregraph.contracts.contract import DeploymentContract
from coregraph.routing.contract_encoder import FactorisedContractEncoder
from coregraph.routing.router import CoReRouter, RouterOutput


class CoReGraph(nn.Module):
    """Factorised contract encoder plus resource-aware CoReRouter."""

    def __init__(
        self,
        *,
        num_experts: int,
        diagnostic_dim: int,
        contract_embedding_dim: int = 8,
        contract_output_dim: int = 32,
        router_hidden_dim: int = 64,
        router_mode: str = "mlp",
        pairwise_interactions: bool = True,
        axis_dropout: float = 0.1,
        contract_noise_std: float = 0.01,
    ):
        super().__init__()
        self.contract_encoder = FactorisedContractEncoder(
            embedding_dim=contract_embedding_dim,
            output_dim=contract_output_dim,
            pairwise_interactions=pairwise_interactions,
            axis_dropout=axis_dropout,
            contract_noise_std=contract_noise_std,
        )
        self.router = CoReRouter(
            num_experts=num_experts,
            contract_dim=contract_output_dim,
            diagnostic_dim=diagnostic_dim,
            hidden_dim=router_hidden_dim,
            mode=router_mode,
        )

    def forward(
        self,
        *,
        contracts: Sequence[DeploymentContract],
        expert_scores: torch.Tensor,
        diagnostics: torch.Tensor,
        availability_mask: torch.Tensor,
        expert_costs: torch.Tensor | None = None,
        expert_names: Sequence[str] | None = None,
    ) -> RouterOutput:
        if len(contracts) != expert_scores.shape[0]:
            raise ValueError("one deployment contract is required per example")
        embedding = self.contract_encoder(contracts)
        return self.router(
            expert_scores=expert_scores,
            contract_embedding=embedding,
            diagnostics=diagnostics,
            availability_mask=availability_mask,
            expert_costs=expert_costs,
            expert_names=expert_names,
        )
