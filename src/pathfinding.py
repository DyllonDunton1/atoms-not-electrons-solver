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

    def footprint_for_robot(self, robot_id: int) -> Footprint:
        """Return the robot-center-relative footprint for a docked robot."""
        robot = self.world.robots.get(robot_id)
        if robot is None:
            raise ValueError(f"Unknown robot id {robot_id}")

        offsets: Set[Position] = {(0, 0)}
        for pallet_id in robot.docked_pallets:
            pallet = self.world.pallets.get(pallet_id)
            if (
                pallet is None
                or pallet.docked_to != robot_id
                or pallet.docked_offset is None
            ):
                raise ValueError(
                    f"Robot {robot_id} has inconsistent docked pallet {pallet_id}"
                )
            offsets.add(pallet.docked_offset)

        return frozenset(offsets)

    def _footprint_is_clear(
        self,
        center: Position,
        footprint: Footprint,
        blocked: Set[Position],
        center_blocked: Set[Position],
    ) -> bool:
        """Return whether every cell in a footprint is in bounds and unblocked."""
        if center in center_blocked:
            return False

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
        ignored_pallet_ids: Iterable[int] = (),
    ) -> List[Position]:
        """Return a shortest path from ``start`` to ``goal`` using A*.

        The returned path contains both the start and goal robot-center
        positions. Every pallet home cell remains reserved even while that
        pallet is being carried elsewhere, and a moved pallet's current cell is
        blocked as well. ``ignored_pallet_ids`` is for pallets that are part of
        the moving footprint itself, such as pallets currently docked to the
        robot. Their own home/current cells may be occupied by that moving
        footprint, but robot centers still may not enter any pallet home cell.
        Other robots are intentionally not static obstacles; time-based robot
        conflicts belong to the scheduler/reservation layer. An empty list
        means no valid path exists.
        """
        if not footprint or (0, 0) not in footprint:
            raise ValueError("Footprint must include the robot center at (0, 0)")

        ignored_ids = set(ignored_pallet_ids)
        unknown_ignored_ids = ignored_ids - set(self.world.pallets)
        if unknown_ignored_ids:
            raise ValueError(
                f"Unknown ignored pallet ids: {sorted(unknown_ignored_ids)}"
            )

        pallet_home_positions = {
            pallet.original_position
            for pallet in self.world.pallets.values()
        }
        blocked_positions = {
            pallet.original_position
            for pallet_id, pallet in self.world.pallets.items()
            if pallet_id not in ignored_ids
        }
        blocked_positions.update(
            pallet.position
            for pallet_id, pallet in self.world.pallets.items()
            if pallet_id not in ignored_ids
        )
        blocked_positions.update(blocked)

        if not self._footprint_is_clear(
            start,
            footprint,
            blocked_positions,
            pallet_home_positions,
        ):
            return []
        if not self._footprint_is_clear(
            goal,
            footprint,
            blocked_positions,
            pallet_home_positions,
        ):
            return []
        if start == goal:
            return [start]

        # Heap entries are (f_score, g_score, tie_breaker, position).
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
                    pallet_home_positions,
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