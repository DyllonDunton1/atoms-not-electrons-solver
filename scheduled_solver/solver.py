"""Top-level asynchronous FIFO scheduler for the independent architecture."""

from __future__ import annotations

import heapq
import time
from collections import deque
from typing import Callable, Dict, Iterable, List, Optional

from .config import SchedulerConfig
from .geometry import WarehouseGeometry, build_geometry
from .inventory import InventoryTimeline
from .models import (
    Action,
    ActionType,
    CommittedOrderSchedule,
    PlannerStats,
    ProblemInstance,
    RobotScheduleState,
)
from .planner import FullHorizonBeamPlanner
from .reservations import ReservationConflict, ReservationTable
from .validation import validate_action_uniqueness, validate_schedule_structure


class ScheduledSolver:
    """Plan complete orders in FIFO assignment order using prioritized reservations."""

    def __init__(
        self,
        problem: ProblemInstance,
        *,
        robot_ids: Optional[Iterable[int]] = None,
        order_ids: Optional[Iterable[int]] = None,
        config: SchedulerConfig = SchedulerConfig(),
    ) -> None:
        self.problem = problem
        self.config = config
        all_robot_ids = {robot.robot_id for robot in problem.robots}
        self.active_robot_ids = sorted(all_robot_ids if robot_ids is None else set(robot_ids))
        if not self.active_robot_ids:
            raise ValueError("At least one active robot is required")
        unknown = set(self.active_robot_ids) - all_robot_ids
        if unknown:
            raise ValueError(f"Unknown robot ids: {sorted(unknown)}")

        all_order_ids = {order.order_id for order in problem.orders}
        self.order_ids = sorted(all_order_ids if order_ids is None else set(order_ids))
        unknown_orders = set(self.order_ids) - all_order_ids
        if unknown_orders:
            raise ValueError(f"Unknown order ids: {sorted(unknown_orders)}")

        base_geometry = build_geometry(
            problem.pallets,
            require_24_columns=config.require_24_columns,
        )
        inactive_positions = {
            robot.start
            for robot in problem.robots
            if robot.robot_id not in self.active_robot_ids
        }
        self.geometry = WarehouseGeometry(
            width=base_geometry.width,
            height=base_geometry.height,
            fulfillment_y=base_geometry.fulfillment_y,
            replenishment_y=base_geometry.replenishment_y,
            static_blocked=base_geometry.static_blocked | frozenset(inactive_positions),
            columns=base_geometry.columns,
            pallet_to_column=base_geometry.pallet_to_column,
        )
        self.reservations = ReservationTable(config.reservation_padding)
        self.inventory = InventoryTimeline(problem.pallets)
        self.stats = PlannerStats()
        self.planner = FullHorizonBeamPlanner(
            self.geometry,
            problem.pallets,
            self.reservations,
            self.inventory,
            config,
            self.stats,
        )
        robot_by_id = {robot.robot_id: robot for robot in problem.robots}
        self.robot_states: Dict[int, RobotScheduleState] = {
            robot_id: RobotScheduleState(robot_id, 0, robot_by_id[robot_id].start)
            for robot_id in self.active_robot_ids
        }
        self.schedules: List[CommittedOrderSchedule] = []
        self.assignment: Dict[int, int] = {}
        self.actions: List[Action] = []

        for robot_id in self.active_robot_ids:
            self.reservations.reserve_pose(
                [self.robot_states[robot_id].position],
                0,
                robot_id,
            )

    def _preflight_commit(self, schedule: CommittedOrderSchedule) -> None:
        validate_schedule_structure(schedule, self.geometry)
        pose_by_time = {pose.timestep: pose for pose in schedule.poses}
        for pose in schedule.poses:
            cells = self.geometry.footprint_cells(pose.center, pose.footprint_offsets)
            if not self.reservations.vertex_reservation_is_free(cells, pose.timestep, schedule.robot_id):
                raise ReservationConflict(
                    f"Schedule pose conflicts before commit at t={pose.timestep}"
                )
        for action in schedule.actions:
            if action.action != ActionType.MOVE:
                continue
            before = pose_by_time[action.timestep]
            after = pose_by_time[action.timestep + 1]
            edges = tuple(
                (
                    (before.center[0] + dx, before.center[1] + dy),
                    (after.center[0] + dx, after.center[1] + dy),
                )
                for dx, dy in before.footprint_offsets
            )
            if not self.reservations.edge_reservation_is_free(edges, action.timestep, schedule.robot_id):
                raise ReservationConflict(
                    f"Schedule edge conflicts before commit at t={action.timestep}"
                )
        for pallet in schedule.pallet_reservations:
            if not self.reservations.pallet_is_free(
                pallet.pallet_id,
                pallet.start_timestep,
                pallet.end_timestep,
                pallet.robot_id,
            ):
                raise ReservationConflict(
                    f"Pallet {pallet.pallet_id} conflicts before schedule commit"
                )

        if not self.reservations.terminal_hold_is_free(
            schedule.end_position,
            schedule.finish_timestep,
            schedule.robot_id,
        ):
            raise ReservationConflict(
                f"Terminal position {schedule.end_position} from "
                f"t={schedule.finish_timestep} conflicts before commit"
            )

    def _commit(self, schedule: CommittedOrderSchedule) -> None:
        self._preflight_commit(schedule)
        pose_by_time = {pose.timestep: pose for pose in schedule.poses}
        for pose in schedule.poses:
            self.reservations.reserve_pose(
                self.geometry.footprint_cells(pose.center, pose.footprint_offsets),
                pose.timestep,
                schedule.robot_id,
            )
        for action in schedule.actions:
            if action.action != ActionType.MOVE:
                continue
            before = pose_by_time[action.timestep]
            after = pose_by_time[action.timestep + 1]
            edges = tuple(
                (
                    (before.center[0] + dx, before.center[1] + dy),
                    (after.center[0] + dx, after.center[1] + dy),
                )
                for dx, dy in before.footprint_offsets
            )
            self.reservations.reserve_edges(edges, action.timestep, schedule.robot_id)
        for pallet in schedule.pallet_reservations:
            self.reservations.reserve_pallet(pallet)
        self.inventory.commit(schedule.inventory_events)
        self.reservations.set_terminal_hold(
            schedule.end_position,
            schedule.finish_timestep,
            schedule.robot_id,
        )

        actions_by_time = {action.timestep for action in schedule.actions}
        for before, after in zip(schedule.poses, schedule.poses[1:]):
            if before.center == after.center and before.timestep not in actions_by_time:
                self.stats.wait_steps += 1
        self.stats.refill_trips += sum(
            1 for action in schedule.actions if action.action == ActionType.DOCK
        )

    def solve(
        self,
        progress_callback: Optional[Callable[[int, int, CommittedOrderSchedule], None]] = None,
    ) -> List[Action]:
        if not self.order_ids:
            return []
        orders = {order.order_id: order for order in self.problem.orders}
        queue = deque(self.order_ids)
        available = [
            (state.available_timestep, robot_id)
            for robot_id, state in self.robot_states.items()
        ]
        heapq.heapify(available)
        started = time.perf_counter()

        while queue:
            start_timestep, robot_id = heapq.heappop(available)
            order_id = queue.popleft()
            state = self.robot_states[robot_id]
            if state.available_timestep != start_timestep:
                raise RuntimeError("Robot availability heap became stale")

            # start_timestep is the global minimum robot-availability time, so
            # no future order can begin before it.  Fold/discard dead history
            # while retaining the one-padding-width boundary future checks need.
            compact_started = time.perf_counter()
            self.reservations.compact_before(start_timestep)
            self.inventory.compact_before(start_timestep)
            self.stats.compaction_seconds += time.perf_counter() - compact_started

            schedule = self.planner.plan_order(
                robot_id,
                orders[order_id],
                state.position,
                start_timestep,
            )
            self._commit(schedule)
            self.schedules.append(schedule)
            self.actions.extend(schedule.actions)
            self.assignment[order_id] = robot_id
            self.stats.orders_planned += 1
            if progress_callback is not None:
                progress_callback(self.stats.orders_planned, len(self.order_ids), schedule)

            state.position = schedule.end_position
            state.available_timestep = schedule.finish_timestep
            state.assigned_orders.append(order_id)
            heapq.heappush(available, (state.available_timestep, robot_id))

        self.stats.planning_seconds = time.perf_counter() - started
        self.actions.sort(key=lambda action: (action.timestep, action.robot_id))
        validate_action_uniqueness(self.actions)
        return list(self.actions)
