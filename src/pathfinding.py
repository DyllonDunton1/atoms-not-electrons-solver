"""Shortest-path planning for robots and docked pallet footprints."""

from __future__ import annotations

from heapq import heappop, heappush
from itertools import count
from typing import Dict, FrozenSet, Iterable, List, Set, Tuple

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

    @staticmethod
    def _manhattan(position: Position, goal: Position) -> int:
        """Return Manhattan distance between two grid positions."""
        return abs(position[0] - goal[0]) + abs(position[1] - goal[1])

    def _footprint_is_clear(
        self,
        center: Position,
        footprint: Footprint,
        blocked: Set[Position],
    ) -> bool:
        """Return whether every cell in a footprint is in bounds and unblocked."""
        center_x, center_y = center
        for offset_x, offset_y in footprint:
            occupied = (center_x + offset_x, center_y + offset_y)
            if not self.world.in_bounds(occupied) or occupied in blocked:
                return False
        return True

    def find_path(
        self,
        start: Position,
        goal: Position,
        *,
        footprint: Footprint = SINGLE_ROBOT_FOOTPRINT,
        blocked: Iterable[Position] = (),
    ) -> List[Position]:
        """Return a shortest path from ``start`` to ``goal`` using A*.

        The returned path contains both the start and goal robot-center
        positions. Pallet cells and explicitly blocked cells are treated as
        static obstacles. Other robots are intentionally not static obstacles;
        time-based robot conflicts belong to the scheduler/reservation layer.
        An empty list means no valid path exists.
        """
        if not footprint or (0, 0) not in footprint:
            raise ValueError("Footprint must include the robot center at (0, 0)")

        blocked_positions = {
            pallet.position for pallet in self.world.pallets.values()
        }
        blocked_positions.update(blocked)

        if not self._footprint_is_clear(start, footprint, blocked_positions):
            return []
        if not self._footprint_is_clear(goal, footprint, blocked_positions):
            return []
        if start == goal:
            return [start]

        # Heap entries are (f_score, g_score, tie_breaker, position).
        # The monotonically increasing tie breaker keeps heap ordering stable
        # without affecting shortest-path correctness.
        frontier: List[Tuple[int, int, int, Position]] = []
        tie_breaker = count()
        start_h = self._manhattan(start, goal)
        heappush(frontier, (start_h, 0, next(tie_breaker), start))

        came_from: Dict[Position, Position] = {}
        g_score: Dict[Position, int] = {start: 0}

        while frontier:
            _, current_g, _, current = heappop(frontier)

            # Ignore stale heap entries that were superseded by a shorter route.
            if current_g != g_score.get(current):
                continue

            if current == goal:
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path

            for neighbor in self.world.adjacent_positions(current):
                if not self._footprint_is_clear(
                    neighbor,
                    footprint,
                    blocked_positions,
                ):
                    continue

                tentative_g = current_g + 1
                if tentative_g >= g_score.get(neighbor, float("inf")):
                    continue

                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                heuristic = self._manhattan(neighbor, goal)
                heappush(
                    frontier,
                    (
                        tentative_g + heuristic,
                        tentative_g,
                        next(tie_breaker),
                        neighbor,
                    ),
                )

        return []
