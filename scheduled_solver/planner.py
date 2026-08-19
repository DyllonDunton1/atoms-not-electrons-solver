"""Full-horizon beam planning over directed service columns."""

from __future__ import annotations

import time
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from typing import Dict, FrozenSet, List, Mapping, Optional, Sequence, Tuple

from .config import SchedulerConfig
from .geometry import DIRECTIONS, DOWN, UP, ServiceColumn, WarehouseGeometry
from .inventory import InventoryTimeline
from .models import (
    Action,
    ActionType,
    ColumnVisit,
    CommittedOrderSchedule,
    InventoryEvent,
    Offset,
    OrderSpec,
    PalletReservation,
    PalletSpec,
    PlannerStats,
    Position,
    TimedPose,
)
from .reservations import ReservationTable
from .space_time_astar import AStarCounters, SpaceTimeAStar


SINGLE: FrozenSet[Offset] = frozenset({(0, 0)})


@dataclass(frozen=True)
class _BeamState:
    position: Position
    timestep: int
    remaining: Tuple[Tuple[int, int], ...]
    actions: Tuple[Action, ...]
    poses: Tuple[TimedPose, ...]
    inventory_events: Tuple[InventoryEvent, ...]
    pallet_reservations: Tuple[PalletReservation, ...]
    column_visits: Tuple[ColumnVisit, ...]
    refill_trips: int = 0

    @property
    def remaining_map(self) -> Dict[int, int]:
        return dict(self.remaining)


class FullHorizonBeamPlanner:
    """Plan one complete order against already committed global reservations."""

    def __init__(
        self,
        geometry: WarehouseGeometry,
        pallets: Sequence[PalletSpec],
        reservations: ReservationTable,
        inventory: InventoryTimeline,
        config: SchedulerConfig,
        stats: Optional[PlannerStats] = None,
    ) -> None:
        self.geometry = geometry
        self.pallets = {pallet.pallet_id: pallet for pallet in pallets}
        self.reservations = reservations
        self.inventory = inventory
        self.config = config
        self.stats = stats or PlannerStats()
        self._astar_counters = AStarCounters()
        self.astar = SpaceTimeAStar(
            geometry,
            reservations,
            path_horizon=config.path_horizon,
            max_expansions=config.max_path_expansions,
            counters=self._astar_counters,
        )

        sku_pickups = defaultdict(list)
        for column in self.geometry.columns:
            for pallet_id in column.pallet_ids:
                pallet = self.pallets[pallet_id]
                sku_pickups[pallet.sku].append((column.service_x, pallet.home[1]))
        self._sku_pickups = {
            sku: tuple(sorted(set(pickups))) for sku, pickups in sku_pickups.items()
        }

    @staticmethod
    def _remaining_tuple(remaining: Mapping[int, int]) -> Tuple[Tuple[int, int], ...]:
        return tuple(
            sorted((sku, quantity) for sku, quantity in remaining.items() if quantity > 0)
        )

    def _sync_astar_stats(self) -> None:
        counters = self._astar_counters
        self.stats.astar_calls = counters.calls
        self.stats.astar_expansions = counters.expansions
        self.stats.astar_capped_calls = counters.capped_calls
        self.stats.astar_max_call_expansions = counters.max_call_expansions
        self.stats.astar_max_call_seconds = counters.max_call_seconds
        self.stats.astar_worst_context = counters.worst_context
        self.stats.row_fast_path_hits = counters.fast_row_hits
        self.stats.point_fast_path_hits = counters.fast_point_hits

    def _append_path(
        self,
        state: _BeamState,
        path: List[Tuple[Position, int]],
        robot_id: int,
        footprint_offsets: FrozenSet[Offset],
        exemptions: Mapping[Offset, Position],
    ) -> _BeamState:
        if not path or path[0] != (state.position, state.timestep):
            raise ValueError("Path does not begin at beam state")
        actions = list(state.actions)
        poses = list(state.poses)
        position = state.position
        timestep = state.timestep
        for next_position, next_time in path[1:]:
            if next_time != timestep + 1:
                raise ValueError("Space-time path must advance one timestep at a time")
            if next_position != position:
                actions.append(Action(timestep, robot_id, ActionType.MOVE, next_position))
            position = next_position
            timestep = next_time
            poses.append(
                TimedPose(
                    timestep,
                    position,
                    footprint_offsets,
                    tuple(sorted(exemptions.items())),
                )
            )
        return replace(
            state,
            position=position,
            timestep=timestep,
            actions=tuple(actions),
            poses=tuple(poses),
        )

    def _append_fixed(
        self,
        state: _BeamState,
        robot_id: int,
        action_type: ActionType,
        target: Position,
        *,
        next_offsets: FrozenSet[Offset],
        next_exemptions: Mapping[Offset, Position] = {},
    ) -> Optional[_BeamState]:
        next_time = state.timestep + 1
        if not self.geometry.pose_is_statically_valid(
            state.position,
            next_offsets,
            next_exemptions,
        ):
            return None
        next_cells = self.geometry.footprint_cells(state.position, next_offsets)
        if not self.reservations.vertex_reservation_is_free(
            next_cells, next_time, robot_id
        ):
            return None
        action = Action(state.timestep, robot_id, action_type, target)
        pose = TimedPose(
            next_time,
            state.position,
            next_offsets,
            tuple(sorted(next_exemptions.items())),
        )
        return replace(
            state,
            timestep=next_time,
            actions=state.actions + (action,),
            poses=state.poses + (pose,),
        )

    def _plan_to(
        self,
        state: _BeamState,
        goal: Position,
        robot_id: int,
        *,
        footprint_offsets: FrozenSet[Offset] = SINGLE,
        exemptions: Mapping[Offset, Position] = {},
        min_goal_time: Optional[int] = None,
        goal_hold_steps: int = 0,
        context: str,
        candidate_budget: bool = True,
    ) -> Optional[_BeamState]:
        started = time.perf_counter()
        try:
            path = self.astar.find_path(
                state.position,
                state.timestep,
                goal,
                owner=robot_id,
                footprint_offsets=footprint_offsets,
                static_exemptions=exemptions,
                min_goal_time=min_goal_time,
                goal_hold_steps=goal_hold_steps,
                max_expansions=(
                    self.config.candidate_max_path_expansions
                    if candidate_budget
                    else self.config.max_path_expansions
                ),
                context=context,
            )
        finally:
            self.stats.astar_seconds += time.perf_counter() - started
        if path is None:
            return None
        return self._append_path(
            state, path, robot_id, footprint_offsets, exemptions
        )

    def _plan_to_row(
        self,
        state: _BeamState,
        row: int,
        robot_id: int,
        footprint_offsets: FrozenSet[Offset],
        exemptions: Mapping[Offset, Position],
        *,
        context: str,
    ) -> Optional[_BeamState]:
        started = time.perf_counter()
        try:
            path = self.astar.find_path_to_row(
                state.position,
                state.timestep,
                row,
                owner=robot_id,
                footprint_offsets=footprint_offsets,
                static_exemptions=exemptions,
                max_expansions=self.config.candidate_max_path_expansions,
                context=context,
            )
        finally:
            self.stats.astar_seconds += time.perf_counter() - started
        if path is None:
            return None
        return self._append_path(
            state, path, robot_id, footprint_offsets, exemptions
        )

    def _future_committed_pallet_intervals(
        self,
        pallet_id: int,
        timestep: int,
        robot_id: int,
    ) -> Tuple[Tuple[int, int, int, int], ...]:
        """Return earlier-planned service intervals not finished by ``timestep``."""
        return tuple(
            interval
            for interval in self.reservations.pallet_intervals(pallet_id)
            if interval[2] != robot_id and interval[1] >= timestep
        )

    def _service_once(
        self,
        base: _BeamState,
        pallet: PalletSpec,
        pickup: Position,
        quantity: int,
        robot_id: int,
        order_id: int,
        min_goal_time: Optional[int],
        path_context: str,
    ) -> Optional[_BeamState]:
        state = self._plan_to(
            base,
            pickup,
            robot_id,
            min_goal_time=min_goal_time,
            goal_hold_steps=1,
            context=f"{path_context} pallet={pallet.pallet_id} approach",
        )
        if state is None:
            return None
        service_start = state.timestep
        remaining = quantity
        local_events = list(state.inventory_events)
        refill_trips = state.refill_trips
        dock_offset = (pallet.home[0] - pickup[0], pallet.home[1] - pickup[1])
        if abs(dock_offset[0]) + abs(dock_offset[1]) != 1:
            return None

        while remaining > 0:
            inventory_started = time.perf_counter()
            try:
                stock = self.inventory.stock_at(
                    pallet.pallet_id, state.timestep, local_events
                )
            finally:
                self.stats.inventory_seconds += time.perf_counter() - inventory_started
            inventory_blocked = False
            while stock > 0 and remaining > 0:
                action_time = state.timestep
                pick_event = InventoryEvent(
                    action_time,
                    pallet.pallet_id,
                    "pick",
                    1,
                    robot_id,
                )

                inventory_started = time.perf_counter()
                try:
                    pick_feasible = self.inventory.pick_is_feasible(
                        pallet.pallet_id,
                        action_time,
                        1,
                        robot_id,
                        local_events,
                    )
                finally:
                    self.stats.inventory_seconds += (
                        time.perf_counter() - inventory_started
                    )
                if not pick_feasible:
                    inventory_blocked = True
                    break

                advanced = self._append_fixed(
                    state,
                    robot_id,
                    ActionType.PICK,
                    pallet.home,
                    next_offsets=SINGLE,
                )
                if advanced is None:
                    return None
                state = advanced
                local_events.append(pick_event)
                remaining -= 1
                stock -= 1

            if remaining == 0:
                break

            future_intervals = self._future_committed_pallet_intervals(
                pallet.pallet_id,
                state.timestep,
                robot_id,
            )
            if future_intervals:
                return None
            if inventory_blocked:
                return None

            carried_offsets = frozenset({(0, 0), dock_offset})
            exemptions = {dock_offset: pallet.home}
            advanced = self._append_fixed(
                state,
                robot_id,
                ActionType.DOCK,
                pallet.home,
                next_offsets=carried_offsets,
                next_exemptions=exemptions,
            )
            if advanced is None:
                return None
            state = advanced

            to_refill = self._plan_to_row(
                state,
                self.geometry.replenishment_y,
                robot_id,
                carried_offsets,
                exemptions,
                context=f"{path_context} pallet={pallet.pallet_id} refill-out",
            )
            if to_refill is None:
                return None
            state = to_refill
            refill_event_time = state.timestep - 1
            local_events.append(
                InventoryEvent(
                    refill_event_time,
                    pallet.pallet_id,
                    "refill",
                    0,
                    robot_id,
                )
            )

            returned = self._plan_to(
                state,
                pickup,
                robot_id,
                footprint_offsets=carried_offsets,
                exemptions=exemptions,
                goal_hold_steps=1,
                context=f"{path_context} pallet={pallet.pallet_id} refill-return",
            )
            if returned is None:
                return None
            state = returned
            advanced = self._append_fixed(
                state,
                robot_id,
                ActionType.UNDOCK,
                pallet.home,
                next_offsets=SINGLE,
            )
            if advanced is None:
                return None
            state = advanced
            refill_trips += 1

        service_end = state.timestep - 1
        inventory_started = time.perf_counter()
        try:
            events_feasible = self.inventory.events_are_feasible(
                event
                for event in local_events
                if event.pallet_id == pallet.pallet_id
            )
        finally:
            self.stats.inventory_seconds += time.perf_counter() - inventory_started
        if not events_feasible:
            return None

        conflict = self.reservations.first_pallet_conflict(
            pallet.pallet_id,
            service_start,
            service_end,
            robot_id,
        )
        if conflict is not None:
            return None

        reservation = PalletReservation(
            pallet.pallet_id,
            service_start,
            service_end,
            robot_id,
            order_id,
        )
        return replace(
            state,
            inventory_events=tuple(local_events),
            pallet_reservations=state.pallet_reservations + (reservation,),
            refill_trips=refill_trips,
        )

    def _service_pallet(
        self,
        state: _BeamState,
        pallet: PalletSpec,
        pickup: Position,
        quantity: int,
        robot_id: int,
        order_id: int,
        path_context: str,
    ) -> Optional[_BeamState]:
        min_goal_time: Optional[int] = None
        tried_goal_times = set()

        while True:
            candidate = self._service_once(
                state,
                pallet,
                pickup,
                quantity,
                robot_id,
                order_id,
                min_goal_time,
                path_context,
            )
            if candidate is not None:
                return candidate

            earliest = state.timestep if min_goal_time is None else min_goal_time
            later = self._future_committed_pallet_intervals(
                pallet.pallet_id,
                earliest,
                robot_id,
            )
            if not later:
                return None

            _, end, _, _ = min(later, key=lambda item: (item[1], item[0]))
            new_min = end + self.reservations.padding + 1
            if new_min <= earliest or new_min in tried_goal_times:
                return None
            tried_goal_times.add(new_min)
            min_goal_time = new_min

    def _expand_column(
        self,
        state: _BeamState,
        column: ServiceColumn,
        direction: str,
        robot_id: int,
        order_id: int,
    ) -> Optional[_BeamState]:
        remaining = state.remaining_map
        if not remaining:
            return None
        ordered_ids = list(column.pallet_ids)
        if direction == UP:
            ordered_ids.reverse()
        elif direction != DOWN:
            raise ValueError(direction)

        visit_start = state.timestep
        current = state
        serviced_skus = set()
        used_pallets: List[int] = []
        path_context = (
            f"order={order_id} robot={robot_id} column={column.column_id} dir={direction}"
        )

        for pallet_id in ordered_ids:
            pallet = self.pallets[pallet_id]
            quantity = remaining.get(pallet.sku, 0)
            if quantity <= 0 or pallet.sku in serviced_skus:
                continue
            pickup = (column.service_x, pallet.home[1])
            candidate = self._service_pallet(
                current,
                pallet,
                pickup,
                quantity,
                robot_id,
                order_id,
                path_context,
            )
            if candidate is None:
                continue
            current = candidate
            serviced_skus.add(pallet.sku)
            used_pallets.append(pallet_id)
            del remaining[pallet.sku]

        if not used_pallets:
            return None

        visit = ColumnVisit(
            column.column_id,
            direction,
            visit_start,
            current.timestep,
            tuple(used_pallets),
        )
        return replace(
            current,
            remaining=self._remaining_tuple(remaining),
            column_visits=current.column_visits + (visit,),
        )

    def _lower_bound(self, state: _BeamState) -> int:
        remaining = state.remaining_map
        if not remaining:
            return abs(state.position[1] - self.geometry.fulfillment_y) + 1
        useful_pickups = []
        for sku in remaining:
            useful_pickups.extend(self._sku_pickups.get(sku, ()))
        if not useful_pickups:
            return 10**9
        nearest = min(
            abs(state.position[0] - pickup[0]) + abs(state.position[1] - pickup[1])
            for pickup in useful_pickups
        )
        remaining_picks = sum(remaining.values())
        to_fulfillment = min(
            abs(pickup[1] - self.geometry.fulfillment_y)
            for pickup in useful_pickups
        )
        return nearest + remaining_picks + to_fulfillment + 1

    def _cheap_candidate_key(
        self,
        state: _BeamState,
        column: ServiceColumn,
        direction: str,
    ) -> Optional[Tuple[int, int, int, int, int]]:
        """Cheap static estimate used before expensive reservation-aware expansion."""
        remaining = state.remaining_map
        ordered_ids = list(column.pallet_ids)
        if direction == UP:
            ordered_ids.reverse()
            direction_rank = 0
        elif direction == DOWN:
            direction_rank = 1
        else:
            raise ValueError(direction)

        useful = []
        seen_skus = set()
        for pallet_id in ordered_ids:
            pallet = self.pallets[pallet_id]
            quantity = remaining.get(pallet.sku, 0)
            if quantity <= 0 or pallet.sku in seen_skus:
                continue
            seen_skus.add(pallet.sku)
            useful.append((pallet, quantity))

        if not useful:
            return None

        first_y = useful[0][0].home[1]
        last_y = useful[-1][0].home[1]
        approach = abs(state.position[0] - column.service_x) + abs(
            state.position[1] - first_y
        )
        traversal = abs(last_y - first_y)
        picks = sum(quantity for _, quantity in useful)

        mandatory_refill_cost = 0
        for pallet, quantity in useful:
            capacity = pallet.max_count
            if capacity > 0 and quantity > capacity:
                trips = (quantity - 1) // capacity
                pickup_y = pallet.home[1]
                mandatory_refill_cost += trips * (
                    2 * abs(self.geometry.replenishment_y - pickup_y) + 2
                )

        remaining_after = {
            sku: quantity
            for sku, quantity in remaining.items()
            if sku not in seen_skus
        }
        end_position = (column.service_x, last_y)
        if not remaining_after:
            tail = abs(last_y - self.geometry.fulfillment_y) + 1
        else:
            next_pickups = [
                pickup
                for sku in remaining_after
                for pickup in self._sku_pickups.get(sku, ())
            ]
            if not next_pickups:
                tail = 10**9
            else:
                nearest_next = min(
                    abs(end_position[0] - pickup[0])
                    + abs(end_position[1] - pickup[1])
                    for pickup in next_pickups
                )
                remaining_picks = sum(remaining_after.values())
                min_to_fulfillment = min(
                    abs(pickup[1] - self.geometry.fulfillment_y)
                    for pickup in next_pickups
                )
                tail = nearest_next + remaining_picks + min_to_fulfillment + 1

        estimated = approach + traversal + picks + mandatory_refill_cost + tail
        return (
            estimated,
            -len(seen_skus),
            approach,
            column.column_id,
            direction_rank,
        )

    def _ranked_candidate_specs(
        self,
        state: _BeamState,
    ) -> List[Tuple[ServiceColumn, str]]:
        ranked = []
        for column in self.geometry.columns:
            for direction in DIRECTIONS:
                key = self._cheap_candidate_key(state, column, direction)
                if key is not None:
                    ranked.append((key, column, direction))
        ranked.sort(key=lambda item: item[0])
        return [(column, direction) for _, column, direction in ranked]

    def _expand_preselected_candidates(
        self,
        state: _BeamState,
        robot_id: int,
        order_id: int,
    ) -> List[_BeamState]:
        specs = self._ranked_candidate_specs(state)
        if not specs:
            return []

        batch_width = self.config.candidate_width
        target_feasible = max(1, min(self.config.beam_width, batch_width))
        generated: List[_BeamState] = []
        examined = 0

        while examined < len(specs):
            batch = specs[examined : examined + batch_width]
            for column, direction in batch:
                candidate_started = time.perf_counter()
                try:
                    candidate = self._expand_column(
                        state,
                        column,
                        direction,
                        robot_id,
                        order_id,
                    )
                finally:
                    self.stats.candidate_seconds += (
                        time.perf_counter() - candidate_started
                    )
                if candidate is None:
                    self.stats.failed_expansions += 1
                    continue
                self.stats.beam_generated += 1
                generated.append(candidate)

            examined += len(batch)
            if len(generated) >= target_feasible:
                break

        self.stats.candidate_expansions_skipped += len(specs) - examined
        return generated

    def _finish_order(
        self,
        state: _BeamState,
        robot_id: int,
        order_id: int,
        start_time: int,
        start_position: Position,
    ) -> Optional[CommittedOrderSchedule]:
        if state.remaining:
            return None

        started = time.perf_counter()
        try:
            path = self.astar.find_path_to_row(
                state.position,
                state.timestep,
                self.geometry.fulfillment_y,
                owner=robot_id,
                footprint_offsets=SINGLE,
                goal_hold_steps=1,
                goal_hold_until=self.reservations.reservation_horizon(),
                max_expansions=self.config.max_path_expansions,
                context=f"order={order_id} robot={robot_id} fulfill",
            )
        finally:
            self.stats.astar_seconds += time.perf_counter() - started
        if path is None:
            return None
        finished = self._append_path(state, path, robot_id, SINGLE, {})
        final_action = self._append_fixed(
            finished,
            robot_id,
            ActionType.FULFILL,
            (0, 0),
            next_offsets=SINGLE,
        )
        if final_action is None:
            return None
        return CommittedOrderSchedule(
            robot_id=robot_id,
            order_id=order_id,
            start_timestep=start_time,
            finish_timestep=final_action.timestep,
            start_position=start_position,
            end_position=final_action.position,
            actions=final_action.actions,
            poses=final_action.poses,
            inventory_events=final_action.inventory_events,
            pallet_reservations=final_action.pallet_reservations,
            column_visits=final_action.column_visits,
        )

    def plan_order(
        self,
        robot_id: int,
        order: OrderSpec,
        start_position: Position,
        start_timestep: int,
    ) -> CommittedOrderSchedule:
        remaining = Counter(order.skus)
        start_pose = TimedPose(start_timestep, start_position, SINGLE)
        initial = _BeamState(
            position=start_position,
            timestep=start_timestep,
            remaining=self._remaining_tuple(remaining),
            actions=(),
            poses=(start_pose,),
            inventory_events=(),
            pallet_reservations=(),
            column_visits=(),
        )
        beam = [initial]
        best_complete: Optional[CommittedOrderSchedule] = None

        for _depth in range(self.config.max_beam_depth + 1):
            next_states: List[_BeamState] = []
            for state in beam:
                if not state.remaining:
                    complete = self._finish_order(
                        state,
                        robot_id,
                        order.order_id,
                        start_timestep,
                        start_position,
                    )
                    if complete is not None and (
                        best_complete is None
                        or complete.finish_timestep < best_complete.finish_timestep
                    ):
                        best_complete = complete
                    continue

                self.stats.beam_expansions += 1
                next_states.extend(
                    self._expand_preselected_candidates(
                        state,
                        robot_id,
                        order.order_id,
                    )
                )

            if not next_states:
                break

            dominant: Dict[
                Tuple[Position, Tuple[Tuple[int, int], ...]], _BeamState
            ] = {}
            for candidate_state in next_states:
                key = (candidate_state.position, candidate_state.remaining)
                previous = dominant.get(key)
                if previous is None or candidate_state.timestep < previous.timestep:
                    dominant[key] = candidate_state
            ranked = sorted(
                dominant.values(),
                key=lambda candidate_state: (
                    candidate_state.timestep + self._lower_bound(candidate_state),
                    candidate_state.timestep,
                    len(candidate_state.remaining),
                    candidate_state.position[1],
                    candidate_state.position[0],
                ),
            )
            if best_complete is not None:
                ranked = [
                    candidate_state
                    for candidate_state in ranked
                    if candidate_state.timestep + self._lower_bound(candidate_state)
                    < best_complete.finish_timestep
                ]
            kept = ranked[: self.config.beam_width]
            self.stats.beam_pruned += max(0, len(next_states) - len(kept))
            beam = kept
            if not beam:
                break

        self._sync_astar_stats()
        if best_complete is None:
            raise RuntimeError(
                f"Full-horizon beam search could not schedule order {order.order_id} "
                f"for robot {robot_id} from t={start_timestep}"
            )
        return best_complete
