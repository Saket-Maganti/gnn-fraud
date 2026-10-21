"""Stable contract-level router: one feasible distribution per contract."""

from __future__ import annotations

import torch
import torch.nn as nn

from coregraph.routing.masks import MaskedRouting, apply_feasible_mask


class ContractRouter(nn.Module):
    def __init__(self, contract_dim: int, diagnostic_dim: int, num_experts: int, hidden_dim: int = 32):
        super().__init__()
        if min(contract_dim, diagnostic_dim, num_experts, hidden_dim) <= 0:
            raise ValueError("contract router dimensions must be positive")
        self.network = nn.Sequential(
            nn.Linear(contract_dim + diagnostic_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_experts),
        )

    def forward(
        self,
        contract_embedding: torch.Tensor,
        contract_diagnostics: torch.Tensor,
        feasible_mask: torch.Tensor,
    ) -> MaskedRouting:
        if contract_embedding.ndim != 2 or contract_diagnostics.ndim != 2:
            raise ValueError("contract router inputs must be matrices")
        if contract_embedding.shape[0] != contract_diagnostics.shape[0]:
            raise ValueError("contract router batches must align")
        logits = self.network(torch.cat([contract_embedding, contract_diagnostics], dim=-1))
        return apply_feasible_mask(logits, feasible_mask)
