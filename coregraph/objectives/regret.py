"""Feasible-oracle contract regret."""

from __future__ import annotations

import torch


def feasible_oracle_risk(
    expert_risks: torch.Tensor,
    availability_mask: torch.Tensor,
) -> torch.Tensor:
    if expert_risks.shape != availability_mask.shape:
        raise ValueError("risk and availability tensors must align")
    if not availability_mask.bool().any(dim=-1).all():
        raise ValueError("every contract requires at least one feasible expert")
    inf = torch.full_like(expert_risks, torch.inf)
    return torch.where(availability_mask.bool(), expert_risks, inf).min(dim=-1).values


def contract_regret(
    router_risk: torch.Tensor,
    expert_risks: torch.Tensor,
    availability_mask: torch.Tensor,
) -> torch.Tensor:
    oracle = feasible_oracle_risk(expert_risks, availability_mask)
    if router_risk.shape != oracle.shape:
        raise ValueError("router risk must have one value per contract")
    return router_risk - oracle


def aggregate_regret(regrets: torch.Tensor) -> dict[str, torch.Tensor]:
    if regrets.numel() == 0:
        raise ValueError("regrets cannot be empty")
    return {
        "mean": regrets.mean(),
        "maximum": regrets.max(),
        "median": regrets.median(),
        "q90": torch.quantile(regrets, 0.9),
    }
