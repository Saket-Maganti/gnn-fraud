"""Task-adapter registry."""

from coregraph.tasks.base import TaskAdapter, TaskType
from coregraph.tasks.edge_task import EdgeTaskAdapter
from coregraph.tasks.node_task import NodeTaskAdapter
from coregraph.tasks.transaction_task import TransactionTaskAdapter

_TASKS: dict[TaskType, type[TaskAdapter]] = {
    TaskType.NODE_CLASSIFICATION: NodeTaskAdapter,
    TaskType.EDGE_CLASSIFICATION: EdgeTaskAdapter,
    TaskType.TRANSACTION_CLASSIFICATION: TransactionTaskAdapter,
}


def build_task_adapter(task_type: TaskType | str) -> TaskAdapter:
    task = TaskType(task_type)
    return _TASKS[task]()
