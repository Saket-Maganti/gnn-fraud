"""Compositional held-out contract split families."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Sequence

from coregraph.contracts.axes import (
    AccessRegime,
    BudgetSpec,
    ConstructionSpec,
    ContractRole,
    ResourceSpec,
    SelectionAxis,
    TimeSpec,
    VisibilityAxis,
)
from coregraph.contracts.contract import DeploymentContract


class SplitFamily(str, Enum):
    LEAVE_ONE_CONTRACT_OUT = "leave_one_contract_out"
    LEAVE_ONE_COMBINATION_OUT = "leave_one_combination_out"
    LEAVE_ONE_AXIS_VALUE_OUT = "leave_one_axis_value_out"
    RESOURCE_HOLDOUT = "resource_holdout"
    BUDGET_HOLDOUT = "budget_holdout"
    CONSTRUCTION_HOLDOUT = "construction_holdout"
    VISIBILITY_HOLDOUT = "visibility_holdout"
    TEMPORAL_HORIZON_HOLDOUT = "temporal_horizon_holdout"
    MIXED_COMPOSITIONAL_HOLDOUT = "mixed_compositional_holdout"
    RANDOM_ENVIRONMENT_WEAK_BASELINE = "random_environment_weak_baseline"


@dataclass(frozen=True)
class ContractSplit:
    split_id: str
    family: SplitFamily
    source: tuple[DeploymentContract, ...]
    target: tuple[DeploymentContract, ...]
    access_regime: AccessRegime
    atomic_target_id_seen: bool = False

    def __post_init__(self) -> None:
        source_ids = {contract.contract_id for contract in self.source}
        target_ids = {contract.contract_id for contract in self.target}
        if not self.source or not self.target:
            raise ValueError("contract split requires non-empty source and target sets")
        if source_ids & target_ids:
            raise ValueError("target contract leaked into source contract training")
        if self.atomic_target_id_seen:
            raise ValueError("atomic target contract IDs cannot be seen in held-out evaluation")

    def manifest(self) -> dict[str, object]:
        return {
            "split_id": self.split_id,
            "family": self.family.value,
            "access_regime": self.access_regime.value,
            "source_contract_ids": [contract.contract_id for contract in self.source],
            "target_contract_ids": [contract.contract_id for contract in self.target],
            "target_labels_for_selection": "FORBIDDEN"
            if self.access_regime is not AccessRegime.FEW_LABEL_TARGET
            else "ACCOUNTED_FEW_LABEL_BUDGET_ONLY",
            "atomic_target_id_seen": False,
        }


def compose_contracts(
    *,
    dataset_id: str,
    task_id: str,
    time_values: Sequence[TimeSpec],
    visibility_values: Sequence[VisibilityAxis],
    construction_values: Sequence[ConstructionSpec],
    selection_values: Sequence[SelectionAxis],
    budget_values: Sequence[BudgetSpec],
    resource_values: Sequence[ResourceSpec],
    exclusion_rules: Sequence[Callable[[DeploymentContract], bool]] = (),
) -> tuple[DeploymentContract, ...]:
    contracts: list[DeploymentContract] = []
    for index, coordinates in enumerate(
        itertools.product(
            time_values,
            visibility_values,
            construction_values,
            selection_values,
            budget_values,
            resource_values,
        )
    ):
        time, visibility, construction, selection, budget, resource = coordinates
        access = (
            AccessRegime.TTA_UNLABELLED_TARGET
            if selection is SelectionAxis.UNLABELLED_TARGET_ADAPTATION
            else AccessRegime.FEW_LABEL_TARGET
            if selection is SelectionAxis.FEW_LABEL_ADAPTATION
            else AccessRegime.DG_NO_TARGET
        )
        try:
            contract = DeploymentContract(
                environment_id=f"env_{index:05d}",
                role=ContractRole.SOURCE,
                time=time,
                visibility=visibility,
                construction=construction,
                selection=selection,
                budget=budget,
                resource=resource,
                access_regime=access,
                dataset_id=dataset_id,
                task_id=task_id,
            )
        except ValueError:
            continue
        if not any(rule(contract) for rule in exclusion_rules):
            contracts.append(contract)
    return tuple(contracts)


def leave_one_contract_out(
    contracts: Sequence[DeploymentContract],
    target_index: int,
    *,
    access_regime: AccessRegime = AccessRegime.DG_NO_TARGET,
) -> ContractSplit:
    if not 0 <= target_index < len(contracts):
        raise IndexError("target contract index out of range")
    target_base = contracts[target_index]
    target = target_base.as_role(
        ContractRole.TARGET,
        environment_id=f"{target_base.environment_id}_target",
    )
    source = tuple(
        contract.as_role(ContractRole.SOURCE)
        for index, contract in enumerate(contracts)
        if index != target_index
    )
    return ContractSplit(
        split_id=f"loco_{target_index:05d}",
        family=SplitFamily.LEAVE_ONE_CONTRACT_OUT,
        source=source,
        target=(target,),
        access_regime=access_regime,
    )


def observed_axes_unseen_combination_split(
    contracts: Sequence[DeploymentContract],
    target_index: int,
) -> ContractSplit:
    split = leave_one_contract_out(contracts, target_index)
    target = split.target[0]
    for axis in ("time", "visibility", "construction", "selection", "budget", "resource"):
        if not any(getattr(source, axis) == getattr(target, axis) for source in split.source):
            raise ValueError(f"target contains unseen axis value on {axis}")
    return ContractSplit(
        split_id=f"unseen_combination_{target_index:05d}",
        family=SplitFamily.LEAVE_ONE_COMBINATION_OUT,
        source=split.source,
        target=split.target,
        access_regime=split.access_regime,
    )


def leave_one_axis_value_out(
    contracts: Sequence[DeploymentContract],
    *,
    axis: str,
    value: object,
    access_regime: AccessRegime = AccessRegime.DG_NO_TARGET,
) -> ContractSplit:
    allowed = {"time", "visibility", "construction", "selection", "budget", "resource"}
    if axis not in allowed:
        raise ValueError(f"unknown contract axis {axis}")
    target_base = [contract for contract in contracts if getattr(contract, axis) == value]
    source_base = [contract for contract in contracts if getattr(contract, axis) != value]
    if not target_base or not source_base:
        raise ValueError("axis holdout must produce non-empty source and target")
    family = {
        "resource": SplitFamily.RESOURCE_HOLDOUT,
        "budget": SplitFamily.BUDGET_HOLDOUT,
        "construction": SplitFamily.CONSTRUCTION_HOLDOUT,
        "visibility": SplitFamily.VISIBILITY_HOLDOUT,
        "time": SplitFamily.TEMPORAL_HORIZON_HOLDOUT,
    }.get(axis, SplitFamily.LEAVE_ONE_AXIS_VALUE_OUT)
    return ContractSplit(
        split_id=f"holdout_{axis}_{str(value).replace(' ', '_')}",
        family=family,
        source=tuple(contract.as_role(ContractRole.SOURCE) for contract in source_base),
        target=tuple(
            contract.as_role(
                ContractRole.TARGET,
                environment_id=f"{contract.environment_id}_target",
            )
            for contract in target_base
        ),
        access_regime=access_regime,
    )
