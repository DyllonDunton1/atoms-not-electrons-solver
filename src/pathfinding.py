"""Shortest-path planning for robots and docked pallet footprints."""

from __future__ import annotations

from collections.abc import Iterable
from typing import FrozenSet

from .models import Position
from .world import WorldState


Footprint = FrozenSet[Position]
SINGLE_ROBOT_FOOTPRINT: Footprint = frozenset({(0, 0)})


class PathPlanner:
    """Plan shortest collision-free paths through the static warehouse map.

    Time-based robot reservations are handled by ``scheduler.py``. This module
    focuses on spatial path planning and supports larger footprints when a
    robot is moving one or more docked pallets.
    """

    def __init__(self, world: WorldState) -> None:
        self.world = world

    def find_path(
        self,
        start: Position,
        goal: Position,
        *,
        footprint: Footprint = SINGLE_ROBOT_FOOTPRINT,
        blocked: Iterable[Position] = (),
    ) -> list[Position]:
        """Return a shortest path from ``start`` to ``goal``.

        The first implementation will use A* with Manhattan distance. The
        returned path should include the start and goal positions.
        """
        raise NotImplementedError
