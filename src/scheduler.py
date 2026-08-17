"""Multi-robot scheduling and time-based collision reservations."""

from __future__ import annotations

from collections import defaultdict
from heapq import heappop, heappush
from itertools import count
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .models import Position
from .pathfinding import Footprint, SINGLE_ROBOT_FOOTPRINT
from .world import WorldState


TimedPosition = Tuple[int, Position]
TimedState = Tuple[Position, int]


class ReservationTable:
    """Track cells and directed edges reserved by robots over time."""

    def __init__(self) -> None:
        self.cells: Dict[int, Set[Position]] = defaultdict(set)
        self.edges: Dict[int, Set[Tuple[Position, Position]]] = defaultdict(set)

    @staticmethod
    def footprint_cells(center: Position, footprint: Footprint) -> Set[Position]:
        """Return the absolute cells occupied by a robot-relative footprint."""
        center_x, center_y = center
        return {
            (center_x + offset_x, center_y + offset_y)
            for offset_x, offset_y in footprint
        }

    def reserve_cell(self, timestep: int, position: Position) -> None:
        self.cells[timestep].add(position)

    def reserve_edge(
        self,
        timestep: int,
        start: Position,
        end: Position,
    ) -> None:
        self.edges[timestep].add((start, end))

    def reserve_footprint(
        self,
        timestep: int,
        center: Position,
        footprint: Footprint,
    ) -> None:
        """Reserve every cell occupied by a footprint at one state timestep."""
        for position in self.footprint_cells(center, footprint):
            self.reserve_cell(timestep, position)

    def reserve_transition(
        self,
        timestep: int,
        start: Position,
        end: Position,
        footprint: Footprint,
    ) -> None:
        """Reserve one wait/move transition for an entire rigid footprint.

        The destination footprint is reserved at both the action-start state and
        the resulting state. This mirrors the simulator's conservative movement
        rule: another robot may not occupy a cell at the start of a timestep if
        this transition intends to enter that cell during the timestep.
        """
        distance = abs(end[0] - start[0]) + abs(end[1] - start[1])
        if distance not in (0, 1):
            raise ValueError(
                f"Transition from {start} to {end} is not a wait or one-cell move"
            )

        self.reserve_footprint(timestep, start, footprint)
        self.reserve_footprint(timestep, end, footprint)
        self.reserve_footprint(timestep + 1, end, footprint)

        delta = (end[0] - start[0], end[1] - start[1])
        for offset_x, offset_y in footprint:
            entity_start = (start[0] + offset_x, start[1] + offset_y)
            entity_end = (
                entity_start[0] + delta[0],
                entity_start[1] + delta[1],
            )
            self.reserve_edge(timestep, entity_start, entity_end)

    def reserve_trajectory(
        self,
        trajectory: Sequence[TimedPosition],
        footprint: Footprint = SINGLE_ROBOT_FOOTPRINT,
    ) -> None:
        """Reserve every state and transition in a timed trajectory."""
        if not trajectory:
            return

        first_timestep, first_position = trajectory[0]
        self.reserve_footprint(first_timestep, first_position, footprint)

        for previous, current in zip(trajectory, trajectory[1:]):
            previous_timestep, previous_position = previous
            current_timestep, current_position = current

            if current_timestep != previous_timestep + 1:
                raise ValueError(
                    "Timed trajectory timesteps must increase by exactly one"
                )

            self.reserve_transition(
                previous_timestep,
                previous_position,
                current_position,
                footprint,
            )

    def cell_is_free(self, timestep: int, position: Position) -> bool:
        return position not in self.cells.get(timestep, set())

    def footprint_is_free(
        self,
        timestep: int,
        center: Position,
        footprint: Footprint,
    ) -> bool:
        """Return whether every footprint cell is unreserved at one timestep."""
        return all(
            self.cell_is_free(timestep, position)
            for position in self.footprint_cells(center, footprint)
        )

    def edge_is_free(
        self,
        timestep: int,
        start: Position,
        end: Position,
    ) -> bool:
        """Reject both duplicate and head-on edge conflicts."""
        reservations = self.edges.get(timestep, set())
        return (start, end) not in reservations and (end, start) not in reservations

    def transition_is_free(
        self,
        timestep: int,
        start: Position,
        end: Position,
        footprint: Footprint,
    ) -> bool:
        """Return whether a whole-footprint wait/move is reservation-safe.

        The destination footprint must be free both at the start of the action
        timestep and in the resulting state. Checking the start timestep keeps
        reservation planning consistent with the simulator's conservative rule:
        a robot may not enter a cell another robot is vacating that same step.
        """
        distance = abs(end[0] - start[0]) + abs(end[1] - start[1])
        if distance not in (0, 1):
            return False

        if not self.footprint_is_free(timestep, end, footprint):
            return False
        if not self.footprint_is_free(timestep + 1, end, footprint):
            return False

        delta = (end[0] - start[0], end[1] - start[1])
        for offset_x, offset_y in footprint:
            entity_start = (start[0] + offset_x, start[1] + offset_y)
            entity_end = (
                entity_start[0] + delta[0],
                entity_start[1] + delta[1],
            )
            if not self.edge_is_free(timestep, entity_start, entity_end):
                return False

        return True

    def latest_timestep(self) -> int:
        """Return the latest state timestep touched by any reservation."""
        latest_cell = max(self.cells.keys(), default=0)
        latest_edge = max(self.edges.keys(), default=-1) + 1
        return max(latest_cell, latest_edge)


class Scheduler:
    """Plan deterministic reservation-aware trajectories for multiple robots."""

    def __init__(self, world: WorldState) -> None:
        self.world = world
        self.reservations = ReservationTable()

    @staticmethod
    def _manhattan(position: Position, goal: Position) -> int:
        return abs(position[0] - goal[0]) + abs(position[1] - goal[1])

    def _static_blocked_positions(
        self,
        blocked: Iterable[Position],
        ignored_pallet_ids: Iterable[int],
    ) -> Tuple[Set[Position], Set[Position]]:
        ignored_ids = set(ignored_pallet_ids)
        unknown_ids = ignored_ids - set(self.world.pallets)
        if unknown_ids:
            raise ValueError(
                f"Unknown ignored pallet ids: {sorted(unknown_ids)}"
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
        return blocked_positions, pallet_home_positions

    def _footprint_is_clear(
        self,
        center: Position,
        footprint: Footprint,
        blocked: Set[Position],
        center_blocked: Set[Position],
    ) -> bool:
        if center in center_blocked:
            return False
        for position in self.reservations.footprint_cells(center, footprint):
            if not self.world.in_bounds(position) or position in blocked:
                return False
        return True

    def plan_timed_path(
        self,
        start: Position,
        goal: Position,
        *,
        start_timestep: int = 0,
        footprint: Footprint = SINGLE_ROBOT_FOOTPRINT,
        blocked: Iterable[Position] = (),
        ignored_pallet_ids: Iterable[int] = (),
        max_timestep: Optional[int] = None,
    ) -> List[TimedPosition]:
        """Return a shortest reservation-aware trajectory through space-time.

        States are ``(timestep, robot_center)`` pairs. Successors are the four
        orthogonal moves plus waiting in place. Pallet home cells stay reserved
        even while their pallets are being carried elsewhere; moved pallet
        cells, the full moving footprint, and time reservations are checked as
        well. An empty list means no path was found within the search horizon.
        """
        if start_timestep < 0:
            raise ValueError("start_timestep must be nonnegative")
        if not footprint or (0, 0) not in footprint:
            raise ValueError("Footprint must include the robot center at (0, 0)")

        blocked_positions, pallet_home_positions = self._static_blocked_positions(
            blocked,
            ignored_pallet_ids,
        )

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
        if not self.reservations.footprint_is_free(
            start_timestep,
            start,
            footprint,
        ):
            return []

        if start == goal:
            return [(start_timestep, start)]

        if max_timestep is None:
            map_area = self.world.width * self.world.height
            max_timestep = max(
                start_timestep + 2 * map_area,
                self.reservations.latest_timestep() + map_area,
            )
        if max_timestep < start_timestep:
            raise ValueError("max_timestep cannot be before start_timestep")

        # Heap entries are (f_score, g_score, tie_breaker, position, timestep).
        frontier: List[Tuple[int, int, int, Position, int]] = []
        tie_breaker = count()
        start_state: TimedState = (start, start_timestep)
        heappush(
            frontier,
            (
                self._manhattan(start, goal),
                0,
                next(tie_breaker),
                start,
                start_timestep,
            ),
        )

        came_from: Dict[TimedState, TimedState] = {}
        g_score: Dict[TimedState, int] = {start_state: 0}

        while frontier:
            _, current_g, _, current_position, current_timestep = heappop(frontier)
            current_state = (current_position, current_timestep)

            if current_g != g_score.get(current_state):
                continue

            if current_position == goal:
                states = [current_state]
                while current_state in came_from:
                    current_state = came_from[current_state]
                    states.append(current_state)
                states.reverse()
                return [
                    (timestep, position)
                    for position, timestep in states
                ]

            if current_timestep >= max_timestep:
                continue

            next_timestep = current_timestep + 1
            candidates = self.world.adjacent_positions(current_position)
            candidates.append(current_position)  # Waiting is a legal transition.

            for next_position in candidates:
                if not self._footprint_is_clear(
                    next_position,
                    footprint,
                    blocked_positions,
                    pallet_home_positions,
                ):
                    continue

                if not self.reservations.transition_is_free(
                    current_timestep,
                    current_position,
                    next_position,
                    footprint,
                ):
                    continue

                next_state: TimedState = (next_position, next_timestep)
                tentative_g = current_g + 1
                if tentative_g >= g_score.get(next_state, float("inf")):
                    continue

                came_from[next_state] = current_state
                g_score[next_state] = tentative_g
                heuristic = self._manhattan(next_position, goal)
                heappush(
                    frontier,
                    (
                        tentative_g + heuristic,
                        tentative_g,
                        next(tie_breaker),
                        next_position,
                        next_timestep,
                    ),
                )

        return []

    def reserve_timed_path(
        self,
        trajectory: Sequence[TimedPosition],
        footprint: Footprint = SINGLE_ROBOT_FOOTPRINT,
    ) -> None:
        """Reserve a previously planned trajectory for later-priority robots."""
        self.reservations.reserve_trajectory(trajectory, footprint)

    def plan_and_reserve(
        self,
        start: Position,
        goal: Position,
        *,
        start_timestep: int = 0,
        footprint: Footprint = SINGLE_ROBOT_FOOTPRINT,
        blocked: Iterable[Position] = (),
        ignored_pallet_ids: Iterable[int] = (),
        max_timestep: Optional[int] = None,
    ) -> List[TimedPosition]:
        """Plan one prioritized trajectory and reserve it if successful."""
        trajectory = self.plan_timed_path(
            start,
            goal,
            start_timestep=start_timestep,
            footprint=footprint,
            blocked=blocked,
            ignored_pallet_ids=ignored_pallet_ids,
            max_timestep=max_timestep,
        )
        if trajectory:
            self.reserve_timed_path(trajectory, footprint)
        return trajectory
