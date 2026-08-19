"""Space-time A* used by the scheduled solver.

The planner searches states ``(x, y, t)`` and includes WAIT as a first-class
successor.  Static pallet homes remain blocked permanently.  A carried pallet
may overlap only its own original home through a per-footprint-offset exemption.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Dict, FrozenSet, Iterable, List, Mapping, Optional, Tuple

from .geometry import WarehouseGeometry
from .models import Offset, Position
from .reservations import Edge, ReservationTable


@dataclass
class AStarCounters:
    calls: int = 0
    expansions: int = 0


class SpaceTimeAStar:
    def __init__(
        self,
        geometry: WarehouseGeometry,
        reservations: ReservationTable,
        *,
        path_horizon: int = 512,
        max_expansions: int = 250_000,
        counters: Optional[AStarCounters] = None,
    ) -> None:
        self.geometry = geometry
        self.reservations = reservations
        self.path_horizon = path_horizon
        self.max_expansions = max_expansions
        self.counters = counters or AStarCounters()

    @staticmethod
    def _heuristic(position: Position, goal: Position) -> int:
        return abs(position[0] - goal[0]) + abs(position[1] - goal[1])

    def _cells(
        self,
        center: Position,
        footprint_offsets: FrozenSet[Offset],
    ) -> Tuple[Position, ...]:
        return tuple((center[0] + dx, center[1] + dy) for dx, dy in footprint_offsets)

    def _edges(
        self,
        source: Position,
        target: Position,
        footprint_offsets: FrozenSet[Offset],
    ) -> Tuple[Edge, ...]:
        return tuple(
            ((source[0] + dx, source[1] + dy), (target[0] + dx, target[1] + dy))
            for dx, dy in footprint_offsets
        )

    def _pose_valid(
        self,
        center: Position,
        timestep: int,
        footprint_offsets: FrozenSet[Offset],
        exemptions: Mapping[Offset, Position],
        owner: int,
    ) -> bool:
        if not self.geometry.pose_is_statically_valid(center, footprint_offsets, exemptions):
            return False
        return self.reservations.vertex_reservation_is_free(
            self._cells(center, footprint_offsets), timestep, owner
        )

    def _goal_hold_valid(
        self,
        center: Position,
        timestep: int,
        hold_steps: int,
        footprint_offsets: FrozenSet[Offset],
        exemptions: Mapping[Offset, Position],
        owner: int,
    ) -> bool:
        for future in range(timestep, timestep + hold_steps + 1):
            if not self._pose_valid(
                center, future, footprint_offsets, exemptions, owner
            ):
                return False
        return True

    def find_path(
        self,
        start: Position,
        start_time: int,
        goal: Position,
        *,
        owner: int,
        footprint_offsets: FrozenSet[Offset] = frozenset({(0, 0)}),
        static_exemptions: Mapping[Offset, Position] = {},
        min_goal_time: Optional[int] = None,
        goal_hold_steps: int = 0,
    ) -> Optional[List[Tuple[Position, int]]]:
        self.counters.calls += 1
        max_time = start_time + self.path_horizon
        minimum = start_time if min_goal_time is None else max(start_time, min_goal_time)
        start_state = (start, start_time)
        heap: List[Tuple[int, int, int, Position, int]] = []
        serial = 0
        heapq.heappush(heap, (self._heuristic(start, goal), 0, serial, start, start_time))
        parent: Dict[Tuple[Position, int], Optional[Tuple[Position, int]]] = {start_state: None}
        best_seen = {start_state}
        expansions = 0

        while heap:
            _, g, _, position, timestep = heapq.heappop(heap)
            state = (position, timestep)
            expansions += 1
            self.counters.expansions += 1
            if expansions > self.max_expansions:
                return None
            if (
                position == goal
                and timestep >= minimum
                and self._goal_hold_valid(
                    position,
                    timestep,
                    goal_hold_steps,
                    footprint_offsets,
                    static_exemptions,
                    owner,
                )
            ):
                return self._reconstruct(parent, state)
            if timestep >= max_time:
                continue
            x, y = position
            neighbors = (position, (x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1))
            next_time = timestep + 1
            for target in neighbors:
                if not self._pose_valid(
                    target, next_time, footprint_offsets, static_exemptions, owner
                ):
                    continue
                edges = self._edges(position, target, footprint_offsets)
                if not self.reservations.edge_reservation_is_free(edges, timestep, owner):
                    continue
                next_state = (target, next_time)
                if next_state in best_seen:
                    continue
                best_seen.add(next_state)
                parent[next_state] = state
                new_g = g + 1
                h = self._heuristic(target, goal)
                if next_time < minimum:
                    h = max(h, minimum - next_time)
                serial += 1
                heapq.heappush(heap, (new_g + h, new_g, serial, target, next_time))
        return None

    def find_path_to_row(
        self,
        start: Position,
        start_time: int,
        row: int,
        *,
        owner: int,
        footprint_offsets: FrozenSet[Offset],
        static_exemptions: Mapping[Offset, Position] = {},
        goal_hold_steps: int = 0,
        goal_hold_until: Optional[int] = None,
    ) -> Optional[List[Tuple[Position, int]]]:
        """Return the earliest reservation-valid arrival anywhere on ``row``.

        Because every row cell shares the same admissible vertical-distance
        heuristic, the first accepted row state is the minimum-time reachable
        row position.  ``goal_hold_until`` is used for fulfillment: the chosen
        endpoint must remain safe through every already-committed future
        schedule, after which a terminal hold protects it from later planners.
        """
        self.counters.calls += 1
        max_time = start_time + self.path_horizon
        start_state = (start, start_time)
        heap: List[Tuple[int, int, int, Position, int]] = []
        serial = 0
        heapq.heappush(heap, (abs(start[1] - row), 0, serial, start, start_time))
        parent: Dict[Tuple[Position, int], Optional[Tuple[Position, int]]] = {start_state: None}
        seen = {start_state}
        expansions = 0

        while heap:
            _, g, _, position, timestep = heapq.heappop(heap)
            state = (position, timestep)
            expansions += 1
            self.counters.expansions += 1
            if expansions > self.max_expansions:
                return None
            if position[1] == row:
                hold_steps = goal_hold_steps
                if goal_hold_until is not None and goal_hold_until > timestep:
                    hold_steps = max(hold_steps, goal_hold_until - timestep)
                if self._goal_hold_valid(
                    position,
                    timestep,
                    hold_steps,
                    footprint_offsets,
                    static_exemptions,
                    owner,
                ):
                    return self._reconstruct(parent, state)
            if timestep >= max_time:
                continue
            x, y = position
            next_time = timestep + 1
            for target in (position, (x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                if not self._pose_valid(
                    target, next_time, footprint_offsets, static_exemptions, owner
                ):
                    continue
                if not self.reservations.edge_reservation_is_free(
                    self._edges(position, target, footprint_offsets), timestep, owner
                ):
                    continue
                next_state = (target, next_time)
                if next_state in seen:
                    continue
                seen.add(next_state)
                parent[next_state] = state
                new_g = g + 1
                serial += 1
                heapq.heappush(
                    heap,
                    (new_g + abs(target[1] - row), new_g, serial, target, next_time),
                )
        return None

    @staticmethod
    def _reconstruct(
        parent: Mapping[Tuple[Position, int], Optional[Tuple[Position, int]]],
        state: Tuple[Position, int],
    ) -> List[Tuple[Position, int]]:
        result = []
        current: Optional[Tuple[Position, int]] = state
        while current is not None:
            result.append(current)
            current = parent[current]
        result.reverse()
        return result
