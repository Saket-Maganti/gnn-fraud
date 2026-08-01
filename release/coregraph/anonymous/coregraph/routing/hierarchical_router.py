"""Contract prior plus a bounded, optionally sparse instance correction."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from coregraph.routing.masks import MaskedRouting, apply_feasible_mask


@dataclass(frozen=True)
class HierarchicalRoutingOutput:
    routing: MaskedRouting
    contract_logits: torch.Tensor
    instance_correction: torch.Tensor
    stability_penalty: torch.Tensor
    sparsity_penalty: torch.Tensor


class HierarchicalRouter(nn.Module):
    def __init__(
        self,
        contract_dim: int,
        instance_dim: int,
        diagnostic_dim: int,
        num_experts: int,
        *,
        hidden_dim: int = 32,
        max_correction: float = 1.0,
        correction_threshold: float = 0.0,
    ) -> None:
        super().__init__()
        if max_correction < 0 or correction_threshold < 0:
            raise ValueError("correction controls must be non-negative")
        self.max_correction = max_correction
        self.correction_threshold = correction_threshold
        self.contract_head = nn.Sequential(
            nn.Linear(contract_dim + diagnostic_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_experts),
        )
        self.instance_head = nn.Sequential(
            nn.Linear(instance_dim + diagnostic_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_experts),
        )

    def forward(
        self,
        *,
        contract_embedding: torch.Tensor,
        contract_diagnostics: torch.Tensor,
        instance_features: torch.Tensor,
        instance_diagnostics: torch.Tensor,
        feasible_mask: torch.Tensor,
        contract_group: torch.Tensor | None = None,
        contract_only: bool = False,
        instance_only: bool = False,
    ) -> HierarchicalRoutingOutput:
        if contract_only and instance_only:
            raise ValueError("contract-only and instance-only ablations are mutually exclusive")
        batch = instance_features.shape[0]
        if contract_embedding.shape[0] not in {1, batch}:
            raise ValueError("contract embeddings must be per-instance or one shared contract")
        if contract_embedding.shape[0] == 1:
            contract_embedding = contract_embedding.expand(batch, -1)
            contract_diagnostics = contract_diagnostics.expand(batch, -1)
        contract_logits = self.contract_head(
            torch.cat([contract_embedding, contract_diagnostics], dim=-1)
        )
        raw = self.instance_head(torch.cat([instance_features, instance_diagnostics], dim=-1))
        correction = torch.tanh(raw) * self.max_correction
        if self.correction_threshold:
            correction = torch.where(
                correction.abs() >= self.correction_threshold,
                correction,
                torch.zeros_like(correction),
            )
        if contract_only:
            correction = torch.zeros_like(correction)
        logits = correction if instance_only else contract_logits + correction
        routing = apply_feasible_mask(logits, feasible_mask)
        if contract_group is None:
            mean = routing.probabilities.mean(dim=0, keepdim=True)
            stability = (routing.probabilities - mean).pow(2).mean()
        else:
            if contract_group.shape != (batch,):
                raise ValueError("contract_group must align with instances")
            penalties = []
            for group in torch.unique(contract_group, sorted=True):
                group_weights = routing.probabilities[contract_group == group]
                penalties.append((group_weights - group_weights.mean(dim=0, keepdim=True)).pow(2).mean())
            stability = torch.stack(penalties).mean()
        return HierarchicalRoutingOutput(
            routing=routing,
            contract_logits=contract_logits,
            instance_correction=correction,
            stability_penalty=stability,
            sparsity_penalty=correction.abs().mean(),
        )
