"""CoReRouter implementation with masks, fallback, and abstention."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch
import torch.nn as nn

from coregraph.routing.fallback import FallbackStrategy, fallback_weights


@dataclass(frozen=True)
class RouterOutput:
    expert_weights: torch.Tensor
    selected_expert: torch.Tensor
    blended_score: torch.Tensor
    abstention_probability: torch.Tensor
    routing_entropy: torch.Tensor
    expected_compute: torch.Tensor
    explanation_records: tuple[dict[str, Any], ...]


class CoReRouter(nn.Module):
    def __init__(
        self,
        *,
        num_experts: int,
        contract_dim: int,
        diagnostic_dim: int,
        hidden_dim: int = 64,
        mode: str = "mlp",
        fallback: FallbackStrategy = FallbackStrategy.FEATURE_ONLY_SAFE,
        feature_expert_index: int = 0,
    ):
        super().__init__()
        if num_experts < 1:
            raise ValueError("router requires at least one expert")
        if mode not in {"linear", "mlp", "attention"}:
            raise ValueError("router mode must be linear mlp or attention")
        self.num_experts = num_experts
        self.contract_dim = contract_dim
        self.diagnostic_dim = diagnostic_dim
        self.mode = mode
        self.fallback = fallback
        self.feature_expert_index = feature_expert_index
        token_dim = 1 + contract_dim + diagnostic_dim
        if mode == "linear":
            self.scorer: nn.Module = nn.Linear(token_dim, 1)
        else:
            self.scorer = nn.Sequential(
                nn.Linear(token_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )
        self.attention: nn.MultiheadAttention | None
        if mode == "attention":
            self.attention = nn.MultiheadAttention(
                token_dim,
                num_heads=1,
                batch_first=True,
            )
        else:
            self.attention = None
        self.abstention_head = nn.Sequential(
            nn.Linear(contract_dim + diagnostic_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        *,
        expert_scores: torch.Tensor,
        contract_embedding: torch.Tensor,
        diagnostics: torch.Tensor,
        availability_mask: torch.Tensor,
        expert_costs: torch.Tensor | None = None,
        expert_names: Sequence[str] | None = None,
    ) -> RouterOutput:
        if expert_scores.ndim != 2 or expert_scores.shape[1] != self.num_experts:
            raise ValueError("expert_scores must have shape [batch,num_experts]")
        batch = expert_scores.shape[0]
        if contract_embedding.shape != (batch, self.contract_dim):
            raise ValueError("contract embedding shape mismatch")
        if diagnostics.shape != (batch, self.diagnostic_dim):
            raise ValueError("diagnostic shape mismatch")
        if availability_mask.shape != expert_scores.shape:
            raise ValueError("availability mask shape mismatch")

        context = torch.cat([contract_embedding, diagnostics], dim=-1)
        repeated = context[:, None, :].expand(-1, self.num_experts, -1)
        tokens = torch.cat([expert_scores[:, :, None], repeated], dim=-1)
        if self.attention is not None:
            tokens, _ = self.attention(
                tokens,
                tokens,
                tokens,
                key_padding_mask=~availability_mask.bool(),
                need_weights=False,
            )
        routing_logits = self.scorer(tokens).squeeze(-1)
        no_experts = ~availability_mask.bool().any(dim=-1)
        safe_mask = availability_mask.bool().clone()
        safe_mask[no_experts, self.feature_expert_index] = True
        routing_logits = routing_logits.masked_fill(~safe_mask, -torch.inf)
        weights = torch.softmax(routing_logits, dim=-1)
        fallback, fallback_abstain = fallback_weights(
            availability_mask,
            strategy=self.fallback,
            feature_expert_index=self.feature_expert_index,
        )
        if no_experts.any():
            weights = torch.where(no_experts[:, None], fallback, weights)
        # Mathematically unavailable experts always receive exactly zero.
        weights = weights * availability_mask.float()
        sums = weights.sum(dim=-1, keepdim=True)
        needs_fallback = sums.squeeze(-1) <= 0
        weights = torch.where(
            needs_fallback[:, None],
            fallback,
            weights / sums.clamp_min(1e-12),
        )
        abstention_probability = torch.sigmoid(self.abstention_head(context)).squeeze(-1)
        abstention_probability = torch.where(
            fallback_abstain,
            torch.ones_like(abstention_probability),
            abstention_probability,
        )
        blended = (weights * expert_scores).sum(dim=-1)
        entropy = -(weights * torch.log(weights.clamp_min(1e-12))).sum(dim=-1)
        if expert_costs is None:
            costs = torch.zeros(self.num_experts, device=expert_scores.device)
        else:
            if expert_costs.numel() != self.num_experts:
                raise ValueError("one cost is required per expert")
            costs = expert_costs.to(expert_scores.device)
        expected_compute = (weights * costs).sum(dim=-1)
        names = tuple(expert_names or [f"expert_{i}" for i in range(self.num_experts)])
        selected = weights.argmax(dim=-1)
        selected = torch.where(no_experts, torch.full_like(selected, -1), selected)
        records = tuple(
            {
                "selected_expert": (
                    None if bool(no_experts[row]) else names[int(selected[row].item())]
                ),
                "weights": {
                    names[index]: float(weights[row, index].detach().cpu())
                    for index in range(self.num_experts)
                },
                "unavailable": [
                    names[index]
                    for index in range(self.num_experts)
                    if not bool(availability_mask[row, index])
                ],
                "fallback_used": bool(needs_fallback[row] or no_experts[row]),
                "routing_entropy": float(entropy[row].detach().cpu()),
                "expected_compute": float(expected_compute[row].detach().cpu()),
            }
            for row in range(batch)
        )
        return RouterOutput(
            expert_weights=weights,
            selected_expert=selected,
            blended_score=blended,
            abstention_probability=abstention_probability,
            routing_entropy=entropy,
            expected_compute=expected_compute,
            explanation_records=records,
        )
