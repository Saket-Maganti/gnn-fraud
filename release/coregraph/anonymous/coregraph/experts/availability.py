"""Resource and task availability masks."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from coregraph.contracts.contract import DeploymentContract
from coregraph.experts.base import Availability, Expert
from coregraph.tasks.base import TaskBatch


def availability_mask(
    experts: Sequence[Expert],
    batch: TaskBatch,
    contract: DeploymentContract,
) -> tuple[np.ndarray, tuple[tuple[str, ...], ...]]:
    states = [expert.availability(batch, contract) for expert in experts]
    return (
        np.asarray([state.available for state in states], dtype=bool),
        tuple(
            tuple(reason.value for reason in state.reason_codes)
            for state in states
        ),
    )


def availability_states(
    experts: Sequence[Expert],
    batch: TaskBatch,
    contract: DeploymentContract,
) -> tuple[Availability, ...]:
    """Return complete structured availability records for audit surfaces."""

    return tuple(expert.availability(batch, contract) for expert in experts)
