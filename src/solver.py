"""Top-level solver coordination loop."""

from __future__ import annotations

from pathlib import Path

from .allocator import TaskAllocator, TaskQueue
from .models import Action
from .parser import parse_problem
from .world import WorldState
from .writer import write_submission


class Solver:
    """Coordinate parsing, task allocation, planning, and action generation."""

    def __init__(self, world: WorldState) -> None:
        self.world = world
        self.queue = TaskQueue()
        self.allocator = TaskAllocator()
        self.actions: list[Action] = []

    def solve(self) -> list[Action]:
        """Generate a complete action schedule for the current problem."""
        raise NotImplementedError


def solve_file(input_path: str | Path, output_path: str | Path) -> None:
    """Parse a challenge file, solve it, and write a submission file."""
    problem = parse_problem(input_path)
    world = WorldState(problem)
    actions = Solver(world).solve()
    write_submission(actions, output_path)
