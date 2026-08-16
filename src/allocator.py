"""Task queue and robot allocation logic."""

from __future__ import annotations

from collections import deque

from .models import Robot
from .tasks import Task


class TaskQueue:
    """FIFO queue used by the baseline solver."""

    def __init__(self) -> None:
        self._queue: deque[Task] = deque()

    def push(self, task: Task) -> None:
        self._queue.append(task)

    def pop(self) -> Task | None:
        return self._queue.popleft() if self._queue else None

    def __len__(self) -> int:
        return len(self._queue)


class TaskAllocator:
    """Assign queued tasks to available robots.

    The baseline strategy is intentionally simple: available robots take the
    next task from the FIFO queue. More advanced allocation policies can later
    replace this class without changing the rest of the engine.
    """

    def assign_next(self, robot: Robot, queue: TaskQueue) -> Task | None:
        return queue.pop()
