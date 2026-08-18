"""Aisle-aware five-robot solver built on the FIFO fleet baseline."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional, Set

from .aisles import AislePlan, AislePlanner, AisleStop
from .models import ActionType, Order
from .multi_robot_solver import (
    COLLECT,
    FULFILL,
    RETURN_REFILL,
    TO_REFILL,
    Intent,
    MultiRobotSolver,
    RobotState,
)
from .pathfinding import SINGLE_ROBOT_FOOTPRINT
from .tasks import FulfillOrderTask, TaskStatus
from .world import WorldState


@dataclass
class AisleRobotState(RobotState):
    """Per-robot collection state for aisle-batched order execution."""

    remaining_by_sku: Dict[int, int] = field(default_factory=dict)
    active_aisle_id: Optional[int] = None
    aisle_plan: Optional[AislePlan] = None
    aisle_stop_index: int = 0


class AisleAwareSolver(MultiRobotSolver):
    """Five-robot solver that batches each order into aisle service plans."""

    def __init__(
        self,
        world: WorldState,
        *,
        robot_ids: Optional[Iterable[int]] = None,
        order_ids: Optional[Iterable[int]] = None,
        max_timesteps: int = 5_000_000,
    ) -> None:
        super().__init__(
            world,
            robot_ids=robot_ids,
            order_ids=order_ids,
            max_timesteps=max_timesteps,
        )
        self.aisle_planner = AislePlanner(world)
        self.states = {
            robot_id: AisleRobotState()
            for robot_id in world.robots
        }

    @staticmethod
    def _remaining_requirements(order: Order) -> Dict[int, int]:
        return dict(Counter(order.skus))

    def _assign_free_robots(self) -> None:
        """Assign FIFO work exactly as the baseline, but initialize aisle state."""
        for robot_id in self.active_robot_ids:
            state = self.states[robot_id]
            if state.task is not None or not len(self.queue):
                continue

            robot = self.world.robots[robot_id]
            task = self.allocator.assign_next(robot, self.queue)
            if not isinstance(task, FulfillOrderTask):
                raise RuntimeError("FIFO queue returned an unsupported task")
            if task.task_id in self.assigned_ids:
                raise RuntimeError(f"Task {task.task_id} was assigned twice")

            order = self.world.orders[task.order_id]
            if order.assigned_robot is not None:
                raise RuntimeError(f"Order {order.order_id} was assigned twice")

            task.status = TaskStatus.ACTIVE
            order.assigned_robot = robot_id
            robot.current_order = order.order_id
            self.assigned_ids.add(task.task_id)

            remaining_by_sku = self._remaining_requirements(order)
            self.states[robot_id] = AisleRobotState(
                task=task,
                remaining_by_sku=remaining_by_sku,
                phase=COLLECT if remaining_by_sku else FULFILL,
            )

    def _aisle_congestion(self, robot_id: int) -> Dict[int, int]:
        """Count other robots currently committed to each aisle."""
        congestion: Counter = Counter()
        for other_id in self.active_robot_ids:
            if other_id == robot_id:
                continue
            other_state = self.states[other_id]
            if (
                other_state.task is not None
                and other_state.active_aisle_id is not None
            ):
                congestion[other_state.active_aisle_id] += 1
        return dict(congestion)

    def _priority_adjacent_pallet_ids(self, robot_id: int) -> Set[int]:
        """Return unclaimed pallet choices temporarily yielded to higher priority.

        When a robot is choosing or replanning future aisle stops, any undocked
        pallet currently adjacent to an active lower-ID robot is treated as
        temporarily unavailable. This breaks the warehouse-specific case where
        two neighboring robots finish stops and immediately choose each other's
        pallet positions. The rule is only a stop-selection preference; it does
        not preempt an already-active stop or remove a pallet claim.
        """
        pallet_by_position = {
            pallet.position: pallet.pallet_id
            for pallet in self.world.pallets.values()
            if pallet.docked_to is None
        }
        unavailable: Set[int] = set()

        for higher_priority_id in self.active_robot_ids:
            if higher_priority_id >= robot_id:
                break
            if self._robot_is_permanently_idle(higher_priority_id):
                continue

            higher_priority_position = self.world.robots[higher_priority_id].position
            for adjacent in self.world.adjacent_positions(higher_priority_position):
                pallet_id = pallet_by_position.get(adjacent)
                if pallet_id is not None:
                    unavailable.add(pallet_id)

        return unavailable

    def _unavailable_pallet_ids(self, robot_id: int) -> Set[int]:
        """Return pallet ids excluded from new aisle-stop selection right now."""
        unavailable = {
            pallet_id
            for pallet_id, owner in self.pallet_claims.items()
            if owner != robot_id
        }
        unavailable.update(
            pallet.pallet_id
            for pallet in self.world.pallets.values()
            if pallet.docked_to is not None and pallet.docked_to != robot_id
        )

        priority_adjacent = self._priority_adjacent_pallet_ids(robot_id)
        active_pallet_id = self.states[robot_id].pallet_id
        if active_pallet_id is not None:
            # A higher-priority robot merely passing the pallet must never kick
            # this robot off an already-active stop.
            priority_adjacent.discard(active_pallet_id)
        unavailable.update(priority_adjacent)
        return unavailable

    def _select_new_aisle(self, robot_id: int) -> bool:
        state = self.states[robot_id]
        robot = self.world.robots[robot_id]

        plan = self.aisle_planner.choose_plan(
            robot.position,
            state.remaining_by_sku,
            congestion_by_aisle=self._aisle_congestion(robot_id),
            unavailable_pallet_ids=self._unavailable_pallet_ids(robot_id),
            blocked=self._permanent_robot_cells(robot_id),
        )
        if plan is None:
            return False

        state.active_aisle_id = plan.aisle_id
        state.aisle_plan = plan
        state.aisle_stop_index = 0
        state.pallet_id = None
        state.pickup = None
        state.remaining = 0
        return True

    def _replan_active_aisle(self, robot_id: int) -> bool:
        """Rebuild a stale plan from the robot's current position."""
        state = self.states[robot_id]
        if state.active_aisle_id is None:
            return self._select_new_aisle(robot_id)

        self._release_pallet(robot_id)
        state.pallet_id = None
        state.pickup = None
        state.remaining = 0

        plan = self.aisle_planner.plan_aisle(
            state.active_aisle_id,
            self.world.robots[robot_id].position,
            state.remaining_by_sku,
            congestion=self._aisle_congestion(robot_id).get(
                state.active_aisle_id,
                0,
            ),
            unavailable_pallet_ids=self._unavailable_pallet_ids(robot_id),
            blocked=self._permanent_robot_cells(robot_id),
        )
        if plan is None:
            state.active_aisle_id = None
            state.aisle_plan = None
            state.aisle_stop_index = 0
            return self._select_new_aisle(robot_id)

        state.aisle_plan = plan
        state.aisle_stop_index = 0
        return True

    def _extend_active_aisle_if_useful(self, robot_id: int) -> bool:
        """Rescan the current aisle before leaving it.

        A pallet that was busy when the original aisle plan was built may have
        become available while this robot serviced other stops in the aisle.
        We do not predict future releases or reserve a place in line; this is a
        single live-availability replan from the robot's current position.
        """
        state = self.states[robot_id]
        aisle_id = state.active_aisle_id
        if aisle_id is None or not state.remaining_by_sku:
            return False

        self._release_pallet(robot_id)
        state.pallet_id = None
        state.pickup = None
        state.remaining = 0
        state.row_goal = None

        plan = self.aisle_planner.plan_aisle(
            aisle_id,
            self.world.robots[robot_id].position,
            state.remaining_by_sku,
            congestion=self._aisle_congestion(robot_id).get(aisle_id, 0),
            unavailable_pallet_ids=self._unavailable_pallet_ids(robot_id),
            blocked=self._permanent_robot_cells(robot_id),
        )
        if plan is None:
            return False

        state.aisle_plan = plan
        state.aisle_stop_index = 0
        return True

    def _current_stop(self, robot_id: int) -> Optional[AisleStop]:
        state = self.states[robot_id]
        plan = state.aisle_plan
        if plan is None:
            return None
        if state.aisle_stop_index >= len(plan.stops):
            return None
        return plan.stops[state.aisle_stop_index]

    def _finish_active_aisle(self, robot_id: int) -> None:
        state = self.states[robot_id]
        self._release_pallet(robot_id)
        state.active_aisle_id = None
        state.aisle_plan = None
        state.aisle_stop_index = 0
        state.pallet_id = None
        state.pickup = None
        state.remaining = 0
        state.row_goal = None

        if state.remaining_by_sku:
            state.phase = COLLECT
        else:
            state.phase = FULFILL

    def _advance_stop(self, robot_id: int) -> None:
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
            if not self._extend_active_aisle_if_useful(robot_id):
                self._finish_active_aisle(robot_id)

    def _activate_current_stop(self, robot_id: int) -> bool:
        state = self.states[robot_id]
        stop = self._current_stop(robot_id)
        if stop is None:
            if not self._extend_active_aisle_if_useful(robot_id):
                self._finish_active_aisle(robot_id)
            return False

        quantity = state.remaining_by_sku.get(stop.sku, 0)
        if quantity <= 0:
            self._advance_stop(robot_id)
            return False

        # A stored future stop may have been planned before the local traffic
        # changed. Recheck priority adjacency at activation time so a stale plan
        # cannot send a lower-priority robot toward a pallet now occupied by a
        # higher-priority neighbor. Active stops never reach this branch because
        # they already have state.pallet_id set.
        if stop.pallet_id in self._priority_adjacent_pallet_ids(robot_id):
            return False

        pallet = self.world.pallets[stop.pallet_id]
        claim = self.pallet_claims.get(stop.pallet_id)
        if claim is not None and claim != robot_id:
            return False
        if pallet.docked_to is not None and pallet.docked_to != robot_id:
            return False
        if (
            pallet.docked_to is None
            and pallet.position != pallet.original_position
        ):
            return False
        if self.aisle_planner.aisle_for_pallet(stop.pallet_id) != state.active_aisle_id:
            return False

        self.pallet_claims[stop.pallet_id] = robot_id
        state.pallet_id = stop.pallet_id
        state.pickup = stop.pickup
        state.remaining = quantity
        return True

    def _current_stop_is_still_valid(self, robot_id: int) -> bool:
        state = self.states[robot_id]
        if state.pallet_id is None or state.pickup is None:
            return False

        pallet = self.world.pallets[state.pallet_id]
        claim = self.pallet_claims.get(state.pallet_id)
        if claim != robot_id:
            return False
        if pallet.docked_to not in (None, robot_id):
            return False
        if pallet.docked_to is None and pallet.position != pallet.original_position:
            return False

        return state.pickup in self.world.adjacent_positions(pallet.position)

    def _intent(self, robot_id: int) -> Intent:
        state = self.states[robot_id]
        robot = self.world.robots[robot_id]
        if robot_id not in self.active_robot_ids or state.task is None:
            return Intent()

        for _ in range(20):
            if state.phase == COLLECT:
                if not state.remaining_by_sku:
                    self._finish_active_aisle(robot_id)
                    continue

                if state.aisle_plan is None:
                    if not self._select_new_aisle(robot_id):
                        return Intent()
                    continue

                stop = self._current_stop(robot_id)
                if stop is None:
                    if not self._extend_active_aisle_if_useful(robot_id):
                        self._finish_active_aisle(robot_id)
                    continue
                if state.remaining_by_sku.get(stop.sku, 0) <= 0:
                    self._advance_stop(robot_id)
                    continue

                if state.pallet_id is None:
                    if not self._activate_current_stop(robot_id):
                        if state.aisle_plan is None:
                            continue
                        if not self._replan_active_aisle(robot_id):
                            return Intent()
                        continue

                if not self._current_stop_is_still_valid(robot_id):
                    if not self._replan_active_aisle(robot_id):
                        return Intent()
                    continue

                pallet = self.world.pallets[state.pallet_id]
                if robot.position != state.pickup:
                    return Intent(move_goal=state.pickup)

                if abs(robot.position[0] - pallet.position[0]) + abs(
                    robot.position[1] - pallet.position[1]
                ) != 1:
                    if not self._replan_active_aisle(robot_id):
                        return Intent()
                    continue

                # Consume stock already present before paying for a refill.
                # If the pallet empties while the SKU is still required, the
                # existing replenish subroutine runs and collection resumes in
                # this same aisle.
                if pallet.count == 0:
                    if not self._can_refill(robot_id, state.pallet_id, state.pickup):
                        if not self._replan_active_aisle(robot_id):
                            return Intent()
                        continue
                    state.refill_robot_home = robot.position
                    state.refill_pallet_home = pallet.position
                    return Intent(ActionType.DOCK, pallet.position)

                return Intent(ActionType.PICK, pallet.position)

            if state.phase == TO_REFILL:
                if robot.position[1] == self.world.replenishment_y:
                    state.phase = RETURN_REFILL
                    state.row_goal = None
                    continue

                footprint = self._footprint(robot_id)
                if state.row_goal is not None and self._goal_blocked_by_permanent_robot(
                    robot_id,
                    state.row_goal,
                    footprint,
                ):
                    state.row_goal = None

                if state.row_goal is None:
                    state.row_goal = self._best_row_goal(
                        robot_id,
                        robot.position,
                        self.world.replenishment_y,
                        footprint=footprint,
                        ignored_pallet_ids=robot.docked_pallets,
                    )
                if state.row_goal is None:
                    raise RuntimeError(
                        f"Robot {robot_id} cannot reach replenishment row"
                    )
                return Intent(move_goal=state.row_goal)

            if state.phase == RETURN_REFILL:
                if robot.position != state.refill_robot_home:
                    return Intent(move_goal=state.refill_robot_home)
                pallet = self.world.pallets[state.pallet_id]
                if pallet.position != state.refill_pallet_home:
                    raise RuntimeError("Replenished pallet did not return home")
                return Intent(ActionType.UNDOCK, pallet.position)

            if state.phase == FULFILL:
                if robot.position[1] == self.world.fulfillment_y:
                    order = self.world.orders[state.task.order_id]
                    if Counter(robot.storage) != Counter(order.skus):
                        raise RuntimeError(
                            "Robot storage does not match its queued order"
                        )
                    return Intent(ActionType.FULFILL, (0, 0))

                if state.row_goal is not None and self._goal_blocked_by_permanent_robot(
                    robot_id,
                    state.row_goal,
                    SINGLE_ROBOT_FOOTPRINT,
                ):
                    state.row_goal = None

                if state.row_goal is None:
                    state.row_goal = self._best_row_goal(
                        robot_id,
                        robot.position,
                        self.world.fulfillment_y,
                    )
                if state.row_goal is None:
                    raise RuntimeError(
                        f"Robot {robot_id} cannot reach fulfillment row"
                    )
                return Intent(move_goal=state.row_goal)

            raise RuntimeError(f"Unknown robot phase {state.phase}")

        raise RuntimeError("Aisle-aware robot state machine did not settle")

    def _post_action(self, robot_id: int, intent: Intent) -> None:
        state = self.states[robot_id]

        if intent.action == ActionType.PICK:
            stop = self._current_stop(robot_id)
            if stop is None:
                raise RuntimeError("Pick completed without an active aisle stop")
            if state.remaining <= 0:
                raise RuntimeError("Pick completed with no remaining stop quantity")

            state.remaining -= 1
            sku_remaining = state.remaining_by_sku.get(stop.sku, 0) - 1
            if sku_remaining < 0:
                raise RuntimeError("Aisle collection over-picked an SKU")
            if sku_remaining == 0:
                del state.remaining_by_sku[stop.sku]
            else:
                state.remaining_by_sku[stop.sku] = sku_remaining

            if state.remaining == 0:
                self._advance_stop(robot_id)

        elif intent.action == ActionType.DOCK:
            state.phase = TO_REFILL
            state.row_goal = None

        elif intent.action == ActionType.UNDOCK:
            # Refill is a subroutine of the current aisle. Keep the aisle
            # commitment and current stop intact while returning to collect.
            state.phase = COLLECT
            state.row_goal = None
            state.refill_robot_home = None
            state.refill_pallet_home = None

        elif intent.action == ActionType.FULFILL:
            task = state.task
            if task is None:
                raise RuntimeError("Fulfill completed without a task")
            if task.task_id in self.completed_ids:
                raise RuntimeError(f"Task {task.task_id} completed twice")
            task.status = TaskStatus.COMPLETE
            self.completed_ids.add(task.task_id)
            self.world.robots[robot_id].current_order = None
            self._release_pallet(robot_id)
            self.states[robot_id] = AisleRobotState()
