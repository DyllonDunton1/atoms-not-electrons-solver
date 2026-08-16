"""Task abstractions used by the robot allocator and solver."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto


class TaskStatus(Enum):
    QUEUED = auto()
    ACTIVE = auto()
    COMPLETE = auto()


@dataclass
class Task:
    """Base unit of work assigned to a robot."""

    task_id: int
    status: TaskStatus = TaskStatus.QUEUED


@dataclass
class FulfillOrderTask(Task):
    order_id: int = -1


@dataclass
class ReplenishPalletTask(Task):
    pallet_id: int = -1
    return_to_original_position: bool = True
