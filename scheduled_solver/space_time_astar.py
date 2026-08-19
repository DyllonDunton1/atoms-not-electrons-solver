"""Space-time A* used by the scheduled solver.

The planner searches states ``(x, y, t)`` and includes WAIT as a first-class
successor. Static pallet homes remain blocked permanently. A carried pallet may
overlap only its own original home through a per-footprint-offset exemption.
"""

from __future__ import annotations

import heapq
import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Mapping, Optional, Tuple

from .geometry import WarehouseGeometry
from .models import Offset, Position
from .reservations import Edge, ReservationTable


@dataclass
class AStarCounters:
    calls: int = 0
    expansions: int = 0
    capped_calls: int = 0
    max_call_expansions: int = 0
    max_call_seconds: float = 0.0
    worst_context: str = ""
    fast_row_hits: int = 0
    fast_point_hits: int = 0


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
        self._static_path_cache: Dict[
            Tuple[
                Position,
                Position,
                FrozenSet[Offset],
                Tuple[Tuple[Offset, Position], ...],
            ],
            Optional[Tuple[Position, ...]],
        ] = {}

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

    def _record_call(
        self,
        started: float,
        expansions: int,
        *,
        capped: bool,
        context: str,
    ) -> None:
        elapsed = time.perf_counter() - started
        if capped:
            self.counters.capped_calls += 1
        if expansions > self.counters.max_call_expansions:
            self.counters.max_call_expansions = expansions
        if elapsed > self.counters.max_call_seconds:
            self.counters.max_call_seconds = elapsed
            self.counters.worst_context = context

    @staticmethod
    def _cache_exemptions(
        exemptions: Mapping[Offset, Position],
    ) -> Tuple[Tuple[Offset, Position], ...]:
        return tuple(sorted(exemptions.items()))

    def _static_shortest_path(
        self,
        start: Position,
        goal: Position,
        footprint_offsets: FrozenSet[Offset],
        exemptions: Mapping[Offset, Position],
    ) -> Optional[Tuple[Position, ...]]:
        """Return one cached shortest path using only permanent geometry."""
        key = (
            start,
            goal,
            footprint_offsets,
            self._cache_exemptions(exemptions),
        )
        if key in self._static_path_cache:
            return self._static_path_cache[key]

        if not self.geometry.pose_is_statically_valid(
            start, footprint_offsets, exemptions
        ) or not self.geometry.pose_is_statically_valid(
            goal, footprint_offsets, exemptions
        ):
            self._static_path_cache[key] = None
            return None

        queue = deque([start])
        parent: Dict[Position, Optional[Position]] = {start: None}
        while queue:
            position = queue.popleft()
            if position == goal:
                break
            x, y = position
            for target in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                if target in parent:
                    continue
                if not self.geometry.pose_is_statically_valid(
                    target, footprint_offsets, exemptions
                ):
                    continue
                parent[target] = position
                queue.append(target)

        if goal not in parent:
            self._static_path_cache[key] = None
            return None

        reversed_path = []
        current: Optional[Position] = goal
        while current is not None:
            reversed_path.append(current)
            current = parent[current]
        path = tuple(reversed(reversed_path))
        self._static_path_cache[key] = path
        return path

    def _time_validate_static_path(
        self,
        static_path: Tuple[Position, ...],
        start_time: int,
        *,
        owner: int,
        footprint_offsets: FrozenSet[Offset],
        exemptions: Mapping[Offset, Position],
        min_goal_time: Optional[int],
        goal_hold_steps: int,
    ) -> Optional[List[Tuple[Position, int]]]:
        """Validate a cached geometric shortest path against time reservations."""
        if not static_path:
            return None

        travel_steps = len(static_path) - 1
        minimum = start_time if min_goal_time is None else max(start_time, min_goal_time)
        wait_steps = max(0, minimum - (start_time + travel_steps))
        arrival_time = start_time + wait_steps + travel_steps
        if arrival_time > start_time + self.path_horizon:
            return None

        position = static_path[0]
        timestep = start_time
        result: List[Tuple[Position, int]] = [(position, timestep)]

        for _ in range(wait_steps):
            next_time = timestep + 1
            if not self._pose_valid(
                position,
                next_time,
                footprint_offsets,
                exemptions,
                owner,
            ):
                return None
            if not self.reservations.edge_reservation_is_free(
                self._edges(position, position, footprint_offsets),
                timestep,
                owner,
            ):
                return None
            timestep = next_time
            result.append((position, timestep))

        for target in static_path[1:]:
            next_time = timestep + 1
            if not self._pose_valid(
                target,
                next_time,
                footprint_offsets,
                exemptions,
                owner,
            ):
                return None
            if not self.reservations.edge_reservation_is_free(
                self._edges(position, target, footprint_offsets),
                timestep,
                owner,
            ):
                return None
            position = target
            timestep = next_time
            result.append((position, timestep))

        if not self._goal_hold_valid(
            position,
            timestep,
            goal_hold_steps,
            footprint_offsets,
            exemptions,
            owner,
        ):
            return None
        return result

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
        max_expansions: Optional[int] = None,
        context: str = "point",
    ) -> Optional[List[Tuple[Position, int]]]:
        self.counters.calls += 1
        started = time.perf_counter()
        expansion_limit = self.max_expansions if max_expansions is None else max_expansions
        if expansion_limit <= 0:
            raise ValueError("max_expansions must be positive")

        static_path = self._static_shortest_path(
            start,
            goal,
            footprint_offsets,
            static_exemptions,
        )
        if static_path is not None:
            fast = self._time_validate_static_path(
                static_path,
                start_time,
                owner=owner,
                footprint_offsets=footprint_offsets,
                exemptions=static_exemptions,
                min_goal_time=min_goal_time,
                goal_hold_steps=goal_hold_steps,
            )
            if fast is not None:
                self.counters.fast_point_hits += 1
                self._record_call(started, 0, capped=False, context=context)
                return fast

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
            if expansions > expansion_limit:
                self._record_call(started, expansions, capped=True, context=context)
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
                result = self._reconstruct(parent, state)
                self._record_call(started, expansions, capped=False, context=context)
                return result
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

        self._record_call(started, expansions, capped=False, context=context)
        return None

    def _direct_row_path(
        self,
        start: Position,
        start_time: int,
        row: int,
        *,
        owner: int,
        footprint_offsets: FrozenSet[Offset],
        static_exemptions: Mapping[Offset, Position],
        goal_hold_steps: int,
        goal_hold_until: Optional[int],
    ) -> Optional[List[Tuple[Position, int]]]:
        """Return the vertical shortest path when it is already time-valid."""
        x, y = start
        step = 0 if y == row else (1 if row > y else -1)
        position = start
        timestep = start_time
        path: List[Tuple[Position, int]] = [(position, timestep)]

        while position[1] != row:
            target = (x, position[1] + step)
            next_time = timestep + 1
            if not self._pose_valid(
                target,
                next_time,
                footprint_offsets,
                static_exemptions,
                owner,
            ):
                return None
            if not self.reservations.edge_reservation_is_free(
                self._edges(position, target, footprint_offsets),
                timestep,
                owner,
            ):
                return None
            position = target
            timestep = next_time
            path.append((position, timestep))

        hold_steps = goal_hold_steps
        if goal_hold_until is not None and goal_hold_until > timestep:
            hold_steps = max(hold_steps, goal_hold_until - timestep)
        if not self._goal_hold_valid(
            position,
            timestep,
            hold_steps,
            footprint_offsets,
            static_exemptions,
            owner,
        ):
            return None
        return path

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
        max_expansions: Optional[int] = None,
        context: str = "row",
    ) -> Optional[List[Tuple[Position, int]]]:
        """Return the earliest reservation-valid arrival anywhere on ``row``.

        A clear vertical path is already a globally shortest route to the row,
        so validate and return it before constructing a space-time search. If
        anything blocks that route or its requested terminal hold, fall back to
        the complete A* search.
        """
        self.counters.calls += 1
        started = time.perf_counter()
        expansion_limit = self.max_expansions if max_expansions is None else max_expansions
        if expansion_limit <= 0:
            raise ValueError("max_expansions must be positive")

        direct = self._direct_row_path(
            start,
            start_time,
            row,
            owner=owner,
            footprint_offsets=footprint_offsets,
            static_exemptions=static_exemptions,
            goal_hold_steps=goal_hold_steps,
            goal_hold_until=goal_hold_until,
        )
        if direct is not None:
            self.counters.fast_row_hits += 1
            self._record_call(started, 0, capped=False, context=context)
            return direct

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
            if expansions > expansion_limit:
                self._record_call(started, expansions, capped=True, context=context)
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
                    result = self._reconstruct(parent, state)
                    self._record_call(started, expansions, capped=False, context=context)
                    return result
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

        self._record_call(started, expansions, capped=False, context=context)
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
