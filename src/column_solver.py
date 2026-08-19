"""Directed 24-column collection strategy built on the aisle-aware solver.

The existing 12-island aisle strategy and all spatial/fleet pathfinding remain
unchanged.  This module only changes how collection work is grouped and ordered:
each exposed pallet column is treated as its own service unit, both monotonic
travel directions are evaluated, and the best of the resulting 48 candidates is
chosen directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from .aisle_solver import AisleAwareSolver
from .aisles import (
    CONGESTION_DISTANCE_PENALTY,
    AisleStop,
    build_aisle_layout,
)
from .models import Pallet, Position
from .pathfinding import PathPlanner
from .world import WorldState


UP = "up"
DOWN = "down"
DIRECTIONS = (UP, DOWN)


@dataclass(frozen=True)
class ServiceColumn:
    """One 10-pallet column and its dedicated exposed-side service lane."""

    column_id: int
    pallet_ids: Tuple[int, ...]
    home_positions: Tuple[Position, ...]
    pallet_x: int
    service_x: int

    @property
    def aisle_id(self) -> int:
        """Compatibility alias used by the shared aisle-aware state machine."""
        return self.column_id

    @property
    def service_cells(self) -> Tuple[Position, ...]:
        return tuple((self.service_x, y) for _, y in self.home_positions)


@dataclass(frozen=True)
class ServiceColumnLayout:
    """Deterministic pallet-to-column mappings for the 24-column strategy."""

    columns: Tuple[ServiceColumn, ...]
    pallet_to_column: Dict[int, int]
    home_to_column: Dict[Position, int]

    @property
    def aisles(self) -> Tuple[ServiceColumn, ...]:
        """Compatibility alias for AisleAwareSolver helper methods."""
        return self.columns

    @property
    def pallet_to_aisle(self) -> Dict[int, int]:
        return self.pallet_to_column


@dataclass(frozen=True)
class DirectedColumnPlan:
    """One monotonic pass through a single service column."""

    aisle_id: int
    direction: str
    stops: Tuple[AisleStop, ...]
    useful_quantity: int
    planned_distance: int
    congestion: int
    score: float

    @property
    def column_id(self) -> int:
        return self.aisle_id

    @property
    def useful_sku_count(self) -> int:
        return self.useful_quantity


@dataclass(frozen=True)
class _ColumnOption:
    sku: int
    quantity: int
    pallet_id: int
    pickup: Position
    stock_sufficient: bool


def build_service_column_layout(pallets: Iterable[Pallet]) -> ServiceColumnLayout:
    """Split each physical 2x10 island into two exposed service columns."""
    pallet_list = list(pallets)
    physical_layout = build_aisle_layout(pallet_list)
    pallet_by_id = {pallet.pallet_id: pallet for pallet in pallet_list}

    raw_columns: List[Tuple[int, int, Tuple[int, ...], Tuple[Position, ...]]] = []

    for island in physical_layout.aisles:
        homes_by_x: Dict[int, List[Position]] = {}
        for home in island.home_positions:
            homes_by_x.setdefault(home[0], []).append(home)

        x_values = sorted(homes_by_x)
        if len(x_values) != 2:
            raise ValueError(
                f"Physical aisle {island.aisle_id} must contain exactly two pallet columns"
            )

        left_x, right_x = x_values
        for pallet_x in x_values:
            homes = tuple(sorted(homes_by_x[pallet_x], key=lambda p: p[1]))
            if len(homes) != 10:
                raise ValueError(
                    f"Pallet column x={pallet_x} must contain exactly 10 pallet homes"
                )

            service_x = pallet_x - 1 if pallet_x == left_x else pallet_x + 1
            home_set = set(homes)
            pallet_ids = tuple(
                sorted(
                    (
                        pallet.pallet_id
                        for pallet in pallet_list
                        if pallet.original_position in home_set
                    ),
                    key=lambda pallet_id: (
                        pallet_by_id[pallet_id].original_position[1],
                        pallet_id,
                    ),
                )
            )
            raw_columns.append((homes[0][1], pallet_x, pallet_ids, homes))

    raw_columns.sort(key=lambda item: (item[0], item[1]))

    columns: List[ServiceColumn] = []
    pallet_to_column: Dict[int, int] = {}
    home_to_column: Dict[Position, int] = {}

    for column_id, (_, pallet_x, pallet_ids, homes) in enumerate(raw_columns):
        # The service lane is the exposed side of the two-column physical island.
        # Recover which side is exposed by checking the paired x coordinate.
        home_y = homes[0][1]
        same_band_x = sorted(
            {
                other_x
                for other_y, other_x, _, _ in raw_columns
                if other_y == home_y
                and abs(other_x - pallet_x) == 1
            }
        )
        if len(same_band_x) != 1:
            raise ValueError(f"Could not identify paired pallet column for x={pallet_x}")
        paired_x = same_band_x[0]
        service_x = pallet_x - 1 if pallet_x < paired_x else pallet_x + 1

        column = ServiceColumn(
            column_id=column_id,
            pallet_ids=pallet_ids,
            home_positions=homes,
            pallet_x=pallet_x,
            service_x=service_x,
        )
        columns.append(column)
        for pallet_id in pallet_ids:
            pallet_to_column[pallet_id] = column_id
        for home in homes:
            home_to_column[home] = column_id

    if len(columns) != 24:
        raise ValueError(f"Expected 24 service columns, found {len(columns)}")

    return ServiceColumnLayout(
        columns=tuple(columns),
        pallet_to_column=pallet_to_column,
        home_to_column=home_to_column,
    )


class DirectedColumnPlanner:
    """Evaluate all useful directed service-column routes exactly enough to rank them."""

    def __init__(
        self,
        world: WorldState,
        *,
        congestion_distance_penalty: int = CONGESTION_DISTANCE_PENALTY,
    ) -> None:
        if congestion_distance_penalty < 0:
            raise ValueError("congestion_distance_penalty must be nonnegative")
        self.world = world
        self.spatial = PathPlanner(world)
        self.layout = build_service_column_layout(world.pallets.values())
        self.congestion_distance_penalty = congestion_distance_penalty

    def aisle_for_pallet(self, pallet_id: int) -> int:
        try:
            return self.layout.pallet_to_column[pallet_id]
        except KeyError as exception:
            raise ValueError(f"Unknown pallet id {pallet_id}") from exception

    def choose_plan(
        self,
        start: Position,
        remaining_by_sku: Mapping[int, int],
        *,
        congestion_by_aisle: Mapping[int, int],
        unavailable_pallet_ids: Iterable[int] = (),
        blocked: Iterable[Position] = (),
        excluded_aisle_ids: Iterable[int] = (),
    ) -> Optional[DirectedColumnPlan]:
        """Evaluate every useful column in both directions and return the best plan."""
        unavailable = set(unavailable_pallet_ids)
        blocked_set = set(blocked)
        excluded = set(excluded_aisle_ids)
        candidates: List[DirectedColumnPlan] = []

        for column in self.layout.columns:
            if column.column_id in excluded:
                continue
            congestion = congestion_by_aisle.get(column.column_id, 0)
            for direction in DIRECTIONS:
                plan = self.plan_column_direction(
                    column.column_id,
                    direction,
                    start,
                    remaining_by_sku,
                    congestion=congestion,
                    unavailable_pallet_ids=unavailable,
                    blocked=blocked_set,
                )
                if plan is not None:
                    candidates.append(plan)

        if not candidates:
            return None

        direction_rank = {UP: 0, DOWN: 1}
        candidates.sort(
            key=lambda plan: (
                -plan.score,
                plan.planned_distance,
                -plan.useful_sku_count,
                plan.column_id,
                direction_rank[plan.direction],
            )
        )
        return candidates[0]

    def plan_aisle(
        self,
        aisle_id: int,
        start: Position,
        remaining_by_sku: Mapping[int, int],
        *,
        congestion: int = 0,
        unavailable_pallet_ids: Iterable[int] = (),
        blocked: Iterable[Position] = (),
    ) -> Optional[DirectedColumnPlan]:
        """Compatibility entry point: choose the better direction for one column."""
        plans = [
            self.plan_column_direction(
                aisle_id,
                direction,
                start,
                remaining_by_sku,
                congestion=congestion,
                unavailable_pallet_ids=unavailable_pallet_ids,
                blocked=blocked,
            )
            for direction in DIRECTIONS
        ]
        valid = [plan for plan in plans if plan is not None]
        if not valid:
            return None
        direction_rank = {UP: 0, DOWN: 1}
        valid.sort(
            key=lambda plan: (
                -plan.score,
                plan.planned_distance,
                -plan.useful_sku_count,
                direction_rank[plan.direction],
            )
        )
        return valid[0]

    def plan_column_direction(
        self,
        column_id: int,
        direction: str,
        start: Position,
        remaining_by_sku: Mapping[int, int],
        *,
        congestion: int = 0,
        unavailable_pallet_ids: Iterable[int] = (),
        blocked: Iterable[Position] = (),
    ) -> Optional[DirectedColumnPlan]:
        """Plan one monotonic pass through one column in one direction."""
        if direction not in DIRECTIONS:
            raise ValueError(f"Unknown direction {direction!r}")
        if column_id < 0 or column_id >= len(self.layout.columns):
            raise ValueError(f"Unknown column id {column_id}")

        column = self.layout.columns[column_id]
        unavailable = set(unavailable_pallet_ids)
        blocked_set = set(blocked)
        options_by_sku = self._options_by_sku(
            column,
            remaining_by_sku,
            unavailable,
        )
        if not options_by_sku:
            return None

        selected: List[_ColumnOption] = []
        reverse = direction == UP
        for sku in sorted(options_by_sku):
            preferred = self._preferred_options(options_by_sku[sku])
            option = sorted(
                preferred,
                key=lambda item: (
                    -item.pickup[1] if reverse else item.pickup[1],
                    item.pallet_id,
                ),
            )[0]
            selected.append(option)

        selected.sort(
            key=lambda option: (
                -option.pickup[1] if reverse else option.pickup[1],
                option.pallet_id,
            )
        )
        first = selected[0]
        last = selected[-1]

        # Keep the approach from entering the future service segment from the
        # wrong end. Once the first stop is reached, the pass itself is monotonic.
        first_y = first.pickup[1]
        last_y = last.pickup[1]
        interior_lane_cells = {
            (column.service_x, y)
            for y in range(min(first_y, last_y) + 1, max(first_y, last_y))
        }
        approach_blocked = blocked_set | interior_lane_cells
        approach_path = self.spatial.find_path(
            start,
            first.pickup,
            blocked=approach_blocked,
        )
        if not approach_path:
            return None
        approach_distance = len(approach_path) - 1

        traversal_distance = abs(last_y - first_y)
        if traversal_distance:
            traversal_path = self.spatial.find_path(
                first.pickup,
                last.pickup,
                blocked=blocked_set,
            )
            if not traversal_path or len(traversal_path) - 1 != traversal_distance:
                return None
            if any(position[0] != column.service_x for position in traversal_path):
                return None

        stops = tuple(
            AisleStop(
                sku=option.sku,
                quantity=option.quantity,
                pallet_id=option.pallet_id,
                pickup=option.pickup,
            )
            for option in selected
        )
        useful_sku_count = len(stops)
        planned_distance = approach_distance + traversal_distance
        return DirectedColumnPlan(
            aisle_id=column_id,
            direction=direction,
            stops=stops,
            useful_quantity=useful_sku_count,
            planned_distance=planned_distance,
            congestion=congestion,
            score=self._score(useful_sku_count, planned_distance, congestion),
        )

    def _options_by_sku(
        self,
        column: ServiceColumn,
        remaining_by_sku: Mapping[int, int],
        unavailable_pallet_ids: Set[int],
    ) -> Dict[int, List[_ColumnOption]]:
        result: Dict[int, List[_ColumnOption]] = {}

        for pallet_id in column.pallet_ids:
            pallet = self.world.pallets[pallet_id]
            quantity = remaining_by_sku.get(pallet.sku, 0)
            if quantity <= 0:
                continue
            if pallet_id in unavailable_pallet_ids:
                continue
            if pallet.docked_to is not None:
                continue
            if pallet.position != pallet.original_position:
                continue

            pickup = (column.service_x, pallet.position[1])
            if (
                abs(pickup[0] - pallet.position[0])
                + abs(pickup[1] - pallet.position[1])
                != 1
            ):
                raise RuntimeError("Dedicated service pickup is not adjacent to pallet")

            result.setdefault(pallet.sku, []).append(
                _ColumnOption(
                    sku=pallet.sku,
                    quantity=quantity,
                    pallet_id=pallet_id,
                    pickup=pickup,
                    stock_sufficient=pallet.count >= quantity,
                )
            )

        return result

    @staticmethod
    def _preferred_options(options: Sequence[_ColumnOption]) -> Sequence[_ColumnOption]:
        stocked = [option for option in options if option.stock_sufficient]
        return stocked if stocked else options

    def _score(self, useful_sku_count: int, distance: int, congestion: int) -> float:
        denominator = distance + 1 + congestion * self.congestion_distance_penalty
        return float(useful_sku_count) / float(denominator)


class ColumnAwareSolver(AisleAwareSolver):
    """Aisle-aware state machine with directed 24-column collection planning."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.aisle_planner = DirectedColumnPlanner(self.world)

    def _select_new_aisle(self, robot_id: int) -> bool:
        """Choose a column, applying adjacency only to previous-column fallback.

        Normal 48-route selection intentionally ignores transient persistent
        adjacency because a robot may be many timesteps away from the chosen
        column. Hard unavailability (claims, docking, moved pallets) still
        applies everywhere. If no non-previous column is useful, the fallback
        may reconsider the previous column, but pallets there that are still
        persistently adjacent to a higher-priority robot stay unavailable for
        this timestep. That turns immediate same-column reselection into a wait
        instead of an internal state-machine loop.
        """
        state = self.states[robot_id]
        robot = self.world.robots[robot_id]
        congestion = self._aisle_congestion(robot_id)
        unavailable = self._base_unavailable_pallet_ids(robot_id)
        blocked = self._permanent_robot_cells(robot_id)
        previous_column_id = state.previous_aisle_id
        excluded = [] if previous_column_id is None else [previous_column_id]

        plan = self.aisle_planner.choose_plan(
            robot.position,
            state.remaining_by_sku,
            congestion_by_aisle=congestion,
            unavailable_pallet_ids=unavailable,
            blocked=blocked,
            excluded_aisle_ids=excluded,
        )

        if plan is None and previous_column_id is not None:
            previous_pallet_ids = set(
                self.aisle_planner.layout.columns[previous_column_id].pallet_ids
            )
            previous_adjacency_blockers = (
                self._persistent_priority_blocked_pallet_ids(robot_id)
                & previous_pallet_ids
            )
            fallback_unavailable = unavailable | previous_adjacency_blockers
            plan = self.aisle_planner.choose_plan(
                robot.position,
                state.remaining_by_sku,
                congestion_by_aisle=congestion,
                unavailable_pallet_ids=fallback_unavailable,
                blocked=blocked,
            )
        if plan is None:
            return False

        state.active_aisle_id = plan.aisle_id
        state.aisle_plan = plan
        state.aisle_stop_index = 0
        state.pallet_id = None
        state.pickup = None
        state.remaining = 0
        state.deferred_pallet_ids.clear()
        state.greedy_plan_timestep = None
        return True

    def _extend_active_aisle_if_useful(self, robot_id: int) -> bool:
        """Never rescan/backtrack inside a completed column pass.

        Any still-required SKU is left to the next global 48-route decision.
        The previous-unit exclusion prevents immediate re-entry; the fallback
        may choose the same column only when no other useful column exists and
        its useful pallets are no longer persistently adjacency-blocked.
        """
        return False

    def _advance_stop(self, robot_id: int) -> None:
        """Advance monotonically and leave the column immediately after its last stop."""
        state = self.states[robot_id]
        self._release_pallet(robot_id)
        state.pallet_id = None
        state.pickup = None
        state.remaining = 0
        state.row_goal = None
        state.aisle_stop_index += 1

        if (
            state.aisle_plan is None
            or state.aisle_stop_index >= len(state.aisle_plan.stops)
        ):
            self._finish_active_aisle(robot_id)

    def _stop_is_locally_adjacency_blocked(self, robot_id: int, pallet_id: int, pickup: Position) -> bool:
        """Return whether a nearby stop is persistently blocked by higher priority.

        Adjacency is intentionally ignored while a robot is still far from a
        chosen column. Once it is at most one move from the pickup cell, the
        conflict is local enough to matter and the stop should be deferred.
        """
        robot = self.world.robots[robot_id]
        distance_to_pickup = abs(robot.position[0] - pickup[0]) + abs(
            robot.position[1] - pickup[1]
        )
        if distance_to_pickup > 1:
            return False
        return pallet_id in self._persistent_priority_blocked_pallet_ids(robot_id)

    def _replan_active_aisle(self, robot_id: int) -> bool:
        """Preserve direction, but defer a locally blocked stop before collision."""
        state = self.states[robot_id]
        if state.active_aisle_id is None or state.aisle_plan is None:
            return self._select_new_aisle(robot_id)

        state.greedy_plan_timestep = self.world.timestep
        if (
            state.pallet_id is not None
            and state.pickup is not None
            and self._stop_is_locally_adjacency_blocked(
                robot_id,
                state.pallet_id,
                state.pickup,
            )
        ):
            state.deferred_pallet_ids.add(state.pallet_id)
            self._advance_stop(robot_id)
            return state.aisle_plan is not None

        if state.pallet_id is not None and not self._current_stop_is_still_valid(robot_id):
            state.deferred_pallet_ids.add(state.pallet_id)
            self._advance_stop(robot_id)
        return state.aisle_plan is not None

    def _activate_current_stop(self, robot_id: int) -> bool:
        """Claim the current stop unless hard or locally adjacency-blocked.

        Persistent adjacency does not influence the global 48-route choice and
        does not discard a far-away route. It is enforced only once the robot is
        within one move of the pickup, which preserves long-range planning while
        preventing the local goal-swap deadlock seen when two robots occupy the
        same service lane.
        """
        state = self.states[robot_id]
        stop = self._current_stop(robot_id)
        if stop is None:
            self._finish_active_aisle(robot_id)
            return False

        quantity = state.remaining_by_sku.get(stop.sku, 0)
        if quantity <= 0:
            self._advance_stop(robot_id)
            return False

        unavailable = self._stop_is_locally_adjacency_blocked(
            robot_id,
            stop.pallet_id,
            stop.pickup,
        )
        pallet = self.world.pallets[stop.pallet_id]
        claim = self.pallet_claims.get(stop.pallet_id)
        if claim is not None and claim != robot_id:
            unavailable = True
        if pallet.docked_to is not None and pallet.docked_to != robot_id:
            unavailable = True
        if pallet.docked_to is None and pallet.position != pallet.original_position:
            unavailable = True
        if self.aisle_planner.aisle_for_pallet(stop.pallet_id) != state.active_aisle_id:
            unavailable = True

        if unavailable:
            state.deferred_pallet_ids.add(stop.pallet_id)
            self._advance_stop(robot_id)
            return False

        self.pallet_claims[stop.pallet_id] = robot_id
        state.pallet_id = stop.pallet_id
        state.pickup = stop.pickup
        state.remaining = quantity
        return True
