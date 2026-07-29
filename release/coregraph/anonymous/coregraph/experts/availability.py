"""Resource and task availability masks."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from coregraph.contracts.contract import DeploymentContract
from coregraph.experts.base import Expert
from coregraph.tasks.base import TaskBatch


def availability_mask(
    experts: Sequence[Expert],
    batch: TaskBatch,
    contract: DeploymentContract,
) -> tuple[np.ndarray, tuple[str, ...]]:
    states = [expert.availability(batch, contract) for expert in experts]
    return (
        np.asarray([state.available for state in states], dtype=bool),
        tuple(state.reason for state in states),
    )
