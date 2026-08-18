"""Concurrent five-robot FIFO baseline solver."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Set, Tuple

from .allocator import TaskAllocator, TaskQueue
from .models import Action, ActionType, Order, Position
from .pathfinding import Footprint, PathPlanner, SINGLE_ROBOT_FOOTPRINT
from .scheduler import ReservationTable, Scheduler, TimedPosition
from .simulator import Simulator
from .tasks import FulfillOrderTask, TaskStatus
from .world import WorldState


COLLECT = "collect"
TO_REFILL = "to_refill"
RETURN_REFILL = "return_refill"
FULFILL = "fulfill"
PATH_SLACK = 20


@dataclass
class MovementPlan:
    goal: Position
    trajectory: List[TimedPosition]


@dataclass
class RobotState:
    task: Optional[FulfillOrderTask] = None
    requirements: List[Tuple[int, int]] = field(default_factory=list)
    sku_index: int = 0
    remaining: int = 0
    phase: str = COLLECT
    pallet_id: Optional[int] = None
    pickup: Optional[Position] = None
    refill_robot_home: Optional[Position] = None
    refill_pallet_home: Optional[Position] = None
    row_goal: Optional[Position] = None
    movement: Optional[MovementPlan] = None


@dataclass(frozen=True)
class Intent:
    action: Optional[ActionType] = None
    target: Optional[Position] = None
    move_goal: Optional[Position] = None


class MultiRobotSolver:
    """FIFO order execution with deterministic prioritized collision avoidance."""

    def __init__(
        self,
        world: WorldState,
        *,
        robot_ids: Optional[Iterable[int]] = None,
        order_ids: Optional[Iterable[int]] = None,
        max_timesteps: int = 5_000_000,
    ) -> None:
        self.world = world
        self.simulator = Simulator(world)
        self.spatial = PathPlanner(world)
        self.allocator = TaskAllocator()
        self.queue = TaskQueue()
        self.actions: List[Action] = []
        self.max_timesteps = max_timesteps

        active = sorted(world.robots if robot_ids is None else set(robot_ids))
        if not active:
            raise ValueError("At least one active robot is required")
        unknown = set(active) - set(world.robots)
        if unknown:
            raise ValueError(f"Unknown robot ids: {sorted(unknown)}")
        self.active_robot_ids = active

        selected = (
            [i for i in sorted(world.orders) if not world.orders[i].fulfilled]
            if order_ids is None
            else sorted(order_ids)
        )
        if len(selected) != len(set(selected)):
            raise ValueError("order_ids contains duplicates")
        unknown_orders = set(selected) - set(world.orders)
        if unknown_orders:
            raise ValueError(f"Unknown order ids: {sorted(unknown_orders)}")
        if any(world.orders[i].fulfilled for i in selected):
            raise ValueError("Cannot queue an already fulfilled order")

        self.target_ids = set(selected)
        self.assigned_ids: Set[int] = set()
        self.completed_ids: Set[int] = set()
        for order_id in selected:
            self.queue.push(FulfillOrderTask(task_id=order_id, order_id=order_id))

        self.states = {robot_id: RobotState() for robot_id in world.robots}
        self.pallet_claims: Dict[int, int] = {}

    @staticmethod
    def _requirements(order: Order) -> List[Tuple[int, int]]:
        counts: Dict[int, int] = {}
        for sku in order.skus:
            counts[sku] = counts.get(sku, 0) + 1
        return list(counts.items())

    def _footprint(self, robot_id: int) -> Footprint:
        return self.spatial.footprint_for_robot(robot_id)

    def _footprint_cells(self, robot_id: int) -> Set[Position]:
        robot = self.world.robots[robot_id]
        return ReservationTable.footprint_cells(robot.position, self._footprint(robot_id))

    def _robot_is_permanently_idle(self, robot_id: int) -> bool:
        """Return whether a robot has no remaining chance to receive work."""
        if robot_id not in self.active_robot_ids:
            return True
        return self.states[robot_id].task is None and len(self.queue) == 0

    def _permanent_robot_cells(self, robot_id: int) -> Set[Position]:
        """Return cells occupied forever by other finished/inactive robots."""
        blocked: Set[Position] = set()
        for other_id in self.world.robots:
            if other_id != robot_id and self._robot_is_permanently_idle(other_id):
                blocked.update(self._footprint_cells(other_id))
        return blocked

    def _other_robot_cells(self, robot_id: int) -> Set[Position]:
        blocked: Set[Position] = set()
        for other_id in self.world.robots:
            if other_id != robot_id:
                blocked.update(self._footprint_cells(other_id))
        return blocked

    def _goal_blocked_by_permanent_robot(
        self,
        robot_id: int,
        goal: Position,
        footprint: Footprint,
    ) -> bool:
        goal_cells = ReservationTable.footprint_cells(goal, footprint)
        return bool(goal_cells & self._permanent_robot_cells(robot_id))

    def _assign_free_robots(self) -> None:
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

            requirements = self._requirements(order)
            self.states[robot_id] = RobotState(
                task=task,
                requirements=requirements,
                remaining=requirements[0][1] if requirements else 0,
                phase=COLLECT if requirements else FULFILL,
            )

    def _best_row_goal(
        self,
        robot_id: int,
        start: Position,
        row: int,
        *,
        footprint: Footprint = SINGLE_ROBOT_FOOTPRINT,
        ignored_pallet_ids: Iterable[int] = (),
    ) -> Optional[Position]:
        best = None
        best_length = None
        permanent_blocked = self._permanent_robot_cells(robot_id)

        for x in range(self.world.width):
            path = self.spatial.find_path(
                start,
                (x, row),
                footprint=footprint,
                blocked=permanent_blocked,
                ignored_pallet_ids=ignored_pallet_ids,
            )
            if not path:
                continue
            length = len(path)
            if best_length is None or length < best_length or (
                length == best_length and x < best[0]
            ):
                best = (x, row)
                best_length = length
        return best

    def _pickup_candidates(
        self,
        robot_id: int,
        sku: int,
    ) -> List[Tuple[int, int, Position]]:
        robot = self.world.robots[robot_id]
        permanent_blocked = self._permanent_robot_cells(robot_id)
        candidates = []
        for pallet in sorted(self.world.pallets_for_sku(sku), key=lambda p: p.pallet_id):
            claim = self.pallet_claims.get(pallet.pallet_id)
            if claim is not None and claim != robot_id:
                continue
            if pallet.docked_to is not None and pallet.docked_to != robot_id:
                continue

            for pickup in self.world.adjacent_positions(pallet.position):
                path = self.spatial.find_path(
                    robot.position,
                    pickup,
                    blocked=permanent_blocked,
                )
                if path:
                    candidates.append((len(path) - 1, pallet.pallet_id, pickup))

        candidates.sort(key=lambda item: (item[0], item[1], item[2][1], item[2][0]))
        return candidates

    def _can_refill(self, robot_id: int, pallet_id: int, pickup: Position) -> bool:
        pallet = self.world.pallets[pallet_id]
        offset = (pallet.position[0] - pickup[0], pallet.position[1] - pickup[1])
        footprint = frozenset({(0, 0), offset})
        return self._best_row_goal(
            robot_id,
            pickup,
            self.world.replenishment_y,
            footprint=footprint,
            ignored_pallet_ids=[pallet_id],
        ) is not None

    def _choose_pallet(self, robot_id: int) -> bool:
        state = self.states[robot_id]
        sku = state.requirements[state.sku_index][0]
        candidates = self._pickup_candidates(robot_id, sku)

        choice = None
        for _, pallet_id, pickup in candidates:
            if self.world.pallets[pallet_id].count >= state.remaining:
                choice = (pallet_id, pickup)
                break
        if choice is None:
            for _, pallet_id, pickup in candidates:
                if self._can_refill(robot_id, pallet_id, pickup):
                    choice = (pallet_id, pickup)
                    break
        if choice is None:
            return False

        pallet_id, pickup = choice
        self.pallet_claims[pallet_id] = robot_id
        state.pallet_id = pallet_id
        state.pickup = pickup
        state.movement = None
        return True

    def _release_pallet(self, robot_id: int) -> None:
        state = self.states[robot_id]
        if state.pallet_id is not None and self.pallet_claims.get(state.pallet_id) == robot_id:
            del self.pallet_claims[state.pallet_id]

    def _advance_sku(self, robot_id: int) -> None:
        state = self.states[robot_id]
        self._release_pallet(robot_id)
        state.pallet_id = None
        state.pickup = None
        state.movement = None
        state.row_goal = None
        state.sku_index += 1

        if state.sku_index == len(state.requirements):
            state.remaining = 0
            state.phase = FULFILL
        else:
            state.remaining = state.requirements[state.sku_index][1]
            state.phase = COLLECT

    def _intent(self, robot_id: int) -> Intent:
        state = self.states[robot_id]
        robot = self.world.robots[robot_id]
        if robot_id not in self.active_robot_ids or state.task is None:
            return Intent()

        for _ in range(6):
            if state.phase == COLLECT:
                if state.remaining == 0:
                    self._advance_sku(robot_id)
                    continue
                if state.pallet_id is None and not self._choose_pallet(robot_id):
                    return Intent()

                pallet = self.world.pallets[state.pallet_id]
                if robot.position != state.pickup:
                    return Intent(move_goal=state.pickup)

                if abs(robot.position[0] - pallet.position[0]) + abs(
                    robot.position[1] - pallet.position[1]
                ) != 1:
                    raise RuntimeError("Selected pallet is no longer adjacent")

                if pallet.count < state.remaining:
                    state.refill_robot_home = robot.position
                    state.refill_pallet_home = pallet.position
                    state.movement = None
                    return Intent(ActionType.DOCK, pallet.position)

                return Intent(ActionType.PICK, pallet.position)

            if state.phase == TO_REFILL:
                if robot.position[1] == self.world.replenishment_y:
                    state.phase = RETURN_REFILL
                    state.row_goal = None
                    state.movement = None
                    continue

                footprint = self._footprint(robot_id)
                if state.row_goal is not None and self._goal_blocked_by_permanent_robot(
                    robot_id,
                    state.row_goal,
                    footprint,
                ):
                    state.row_goal = None
                    state.movement = None

                if state.row_goal is None:
                    state.row_goal = self._best_row_goal(
                        robot_id,
                        robot.position,
                        self.world.replenishment_y,
                        footprint=footprint,
                        ignored_pallet_ids=robot.docked_pallets,
                    )
                if state.row_goal is None:
                    raise RuntimeError(f"Robot {robot_id} cannot reach replenishment row")
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
                        raise RuntimeError("Robot storage does not match its queued order")
                    return Intent(ActionType.FULFILL, (0, 0))

                if state.row_goal is not None and self._goal_blocked_by_permanent_robot(
                    robot_id,
                    state.row_goal,
                    SINGLE_ROBOT_FOOTPRINT,
                ):
                    state.row_goal = None
                    state.movement = None

                if state.row_goal is None:
                    state.row_goal = self._best_row_goal(
                        robot_id,
                        robot.position,
                        self.world.fulfillment_y,
                    )
                if state.row_goal is None:
                    raise RuntimeError(f"Robot {robot_id} cannot reach fulfillment row")
                return Intent(move_goal=state.row_goal)

            raise RuntimeError(f"Unknown robot phase {state.phase}")

        raise RuntimeError("Robot state machine did not settle")

    def _cached_plan_usable(
        self,
        robot_id: int,
        goal: Position,
        scheduler: Scheduler,
        blocked: Set[Position],
    ) -> bool:
        state = self.states[robot_id]
        plan = state.movement
        robot = self.world.robots[robot_id]
        footprint = self._footprint(robot_id)

        if (
            plan is None
            or plan.goal != goal
            or not plan.trajectory
            or plan.trajectory[0] != (self.world.timestep, robot.position)
        ):
            return False
        if len(plan.trajectory) < 2:
            return robot.position == goal

        next_timestep, next_position = plan.trajectory[1]
        if next_timestep != self.world.timestep + 1:
            return False

        next_cells = ReservationTable.footprint_cells(next_position, footprint)
        if next_cells & blocked:
            return False

        pallet_cells = {
            pallet.position
            for pallet in self.world.pallets.values()
            if pallet.docked_to is None
        }
        if next_cells & pallet_cells:
            return False

        return scheduler.reservations.transition_is_free(
            self.world.timestep,
            robot.position,
            next_position,
            footprint,
        )

    def _copy_scheduler(self, scheduler: Scheduler) -> Scheduler:
        copied = Scheduler(self.world)
        for reserved_timestep, cells in scheduler.reservations.cells.items():
            copied.reservations.cells[reserved_timestep].update(cells)
        for reserved_timestep, edges in scheduler.reservations.edges.items():
            copied.reservations.edges[reserved_timestep].update(edges)
        return copied

    def _scheduler_with_forced_moves(
        self,
        scheduler: Scheduler,
        forced_first_moves: Dict[int, Position],
        *,
        exclude_robot_id: Optional[int] = None,
    ) -> Scheduler:
        copied = self._copy_scheduler(scheduler)
        timestep = self.world.timestep
        for other_id, destination in sorted(forced_first_moves.items()):
            if other_id == exclude_robot_id:
                continue
            start = self.world.robots[other_id].position
            copied.reservations.reserve_transition(
                timestep,
                start,
                destination,
                self._footprint(other_id),
            )
        return copied

    def _planning_scheduler(
        self,
        scheduler: Scheduler,
        robot_id: int,
        forced_first_moves: Dict[int, Position],
    ) -> Scheduler:
        planning = self._scheduler_with_forced_moves(
            scheduler,
            forced_first_moves,
            exclude_robot_id=robot_id,
        )
        timestep = self.world.timestep
        permanent_blocked = self._permanent_robot_cells(robot_id)
        temporary_blocked = self._other_robot_cells(robot_id) - permanent_blocked

        # Other active robots definitely occupy their current cells at t, but
        # their future cells are left to deterministic priority planning.
        for position in temporary_blocked:
            if planning.reservations.cell_is_free(timestep, position):
                planning.reservations.reserve_cell(timestep, position)
        return planning

    def _immediate_options(
        self,
        robot_id: int,
        scheduler: Scheduler,
    ) -> Tuple[bool, List[Position]]:
        """Return whether waiting is legal and every legal one-step move."""
        check = self._planning_scheduler(scheduler, robot_id, {})
        timestep = self.world.timestep
        robot = self.world.robots[robot_id]
        footprint = self._footprint(robot_id)
        permanent_blocked = self._permanent_robot_cells(robot_id)
        docked_pallet_ids = [
            pallet.pallet_id
            for pallet in self.world.pallets.values()
            if pallet.docked_to is not None
        ]

        can_wait = check.reservations.transition_is_free(
            timestep,
            robot.position,
            robot.position,
            footprint,
        )
        moves: List[Position] = []
        for destination in self.world.adjacent_positions(robot.position):
            trajectory = check.plan_timed_path(
                robot.position,
                destination,
                start_timestep=timestep,
                footprint=footprint,
                blocked=permanent_blocked,
                ignored_pallet_ids=docked_pallet_ids,
                max_timestep=timestep + 1,
            )
            if len(trajectory) == 2:
                moves.append(destination)
        return can_wait, moves

    def _plan_with_forced_first_move(
        self,
        robot_id: int,
        goal: Position,
        scheduler: Scheduler,
        destination: Position,
    ) -> List[TimedPosition]:
        timestep = self.world.timestep
        robot = self.world.robots[robot_id]
        footprint = self._footprint(robot_id)
        permanent_blocked = self._permanent_robot_cells(robot_id)
        docked_pallet_ids = [
            pallet.pallet_id
            for pallet in self.world.pallets.values()
            if pallet.docked_to is not None
        ]

        first_step = scheduler.plan_timed_path(
            robot.position,
            destination,
            start_timestep=timestep,
            footprint=footprint,
            blocked=permanent_blocked,
            ignored_pallet_ids=docked_pallet_ids,
            max_timestep=timestep + 1,
        )
        if len(first_step) != 2:
            return []
        if destination == goal:
            return first_step

        distance = abs(goal[0] - destination[0]) + abs(goal[1] - destination[1])
        continuation = scheduler.plan_timed_path(
            destination,
            goal,
            start_timestep=timestep + 1,
            footprint=footprint,
            blocked=permanent_blocked,
            ignored_pallet_ids=docked_pallet_ids,
            max_timestep=timestep + 1 + distance + PATH_SLACK,
        )
        if not continuation:
            # A forced yield promises only this immediate, legal move.  If the
            # current reservations make the rest of the route unavailable, take
            # the escape now and replan from the new state next timestep.
            return first_step
        return first_step[:-1] + continuation

    def _plan_moves(
        self,
        intents: Dict[int, Intent],
    ) -> Tuple[List[Action], Dict[int, List[TimedPosition]]]:
        scheduler = Scheduler(self.world)
        timestep = self.world.timestep
        chosen: Dict[int, List[TimedPosition]] = {}
        forced_first_moves: Dict[int, Position] = {}
        actions: List[Action] = []

        # Non-moving robots reserve their current rigid footprint for this step.
        for robot_id in sorted(self.world.robots):
            if intents[robot_id].move_goal is None:
                position = self.world.robots[robot_id].position
                scheduler.reservations.reserve_transition(
                    timestep, position, position, self._footprint(robot_id)
                )

        for robot_id in sorted(self.world.robots):
            goal = intents[robot_id].move_goal
            state = self.states[robot_id]
            if goal is None:
                state.movement = None
                continue

            while True:
                robot = self.world.robots[robot_id]
                footprint = self._footprint(robot_id)
                blocked = self._other_robot_cells(robot_id)
                permanent_blocked = self._permanent_robot_cells(robot_id)
                planning = self._planning_scheduler(
                    scheduler,
                    robot_id,
                    forced_first_moves,
                )
                forced_destination = forced_first_moves.get(robot_id)

                if forced_destination is not None:
                    trajectory = self._plan_with_forced_first_move(
                        robot_id,
                        goal,
                        planning,
                        forced_destination,
                    )
                elif not forced_first_moves and self._cached_plan_usable(
                    robot_id,
                    goal,
                    planning,
                    blocked,
                ):
                    trajectory = list(state.movement.trajectory)
                else:
                    docked_pallet_ids = [
                        pallet.pallet_id
                        for pallet in self.world.pallets.values()
                        if pallet.docked_to is not None
                    ]
                    distance = abs(goal[0] - robot.position[0]) + abs(
                        goal[1] - robot.position[1])
                    trajectory = planning.plan_timed_path(
                        robot.position,
                        goal,
                        start_timestep=timestep,
                        footprint=footprint,
                        blocked=permanent_blocked,
                        ignored_pallet_ids=docked_pallet_ids,
                        max_timestep=timestep + distance + PATH_SLACK,
                    )

                if not trajectory:
                    if forced_destination is not None:
                        raise RuntimeError(
                            f"Robot {robot_id} could not execute forced yield move"
                        )
                    state.movement = None
                    break

                # Check the proposed priority trajectory before committing it.
                # If it would leave a lower-priority moving robot with neither
                # a legal wait nor any legal first move, preserve that robot's
                # best currently legal escape and replan around it immediately.
                trial = self._scheduler_with_forced_moves(
                    scheduler,
                    forced_first_moves,
                    exclude_robot_id=robot_id,
                )
                trial.reserve_timed_path(trajectory, footprint)

                added_force = False
                for lower_id in sorted(self.world.robots):
                    if lower_id <= robot_id:
                        continue
                    if intents[lower_id].move_goal is None:
                        continue
                    if lower_id in forced_first_moves:
                        continue

                    can_wait, legal_moves = self._immediate_options(lower_id, trial)
                    if can_wait or legal_moves:
                        continue

                    base = self._scheduler_with_forced_moves(
                        scheduler,
                        forced_first_moves,
                        exclude_robot_id=lower_id,
                    )
                    _, escape_moves = self._immediate_options(lower_id, base)
                    if not escape_moves:
                        raise RuntimeError(
                            f"Robot {lower_id} has no legal immediate yield move"
                        )

                    lower_goal = intents[lower_id].move_goal
                    escape = min(
                        escape_moves,
                        key=lambda position: (
                            abs(lower_goal[0] - position[0])
                            + abs(lower_goal[1] - position[1]),
                            position[1],
                            position[0],
                        ),
                    )
                    forced_first_moves[lower_id] = escape
                    added_force = True

                if added_force:
                    state.movement = None
                    continue

                state.movement = MovementPlan(goal, list(trajectory))
                scheduler.reserve_timed_path(trajectory, footprint)
                chosen[robot_id] = trajectory
                if forced_destination is not None:
                    del forced_first_moves[robot_id]
                if len(trajectory) >= 2 and trajectory[1][1] != robot.position:
                    actions.append(
                        Action(timestep, robot_id, ActionType.MOVE, trajectory[1][1])
                    )
                break

        return actions, chosen

    def _post_action(self, robot_id: int, intent: Intent) -> None:
        state = self.states[robot_id]

        if intent.action == ActionType.PICK:
            state.remaining -= 1
            if state.remaining == 0:
                self._advance_sku(robot_id)

        elif intent.action == ActionType.DOCK:
            state.phase = TO_REFILL
            state.row_goal = None
            state.movement = None

        elif intent.action == ActionType.UNDOCK:
            state.phase = COLLECT
            state.row_goal = None
            state.movement = None
            state.refill_robot_home = None
            state.refill_pallet_home = None

        elif intent.action == ActionType.FULFILL:
            task = state.task
            if task.task_id in self.completed_ids:
                raise RuntimeError(f"Task {task.task_id} completed twice")
            task.status = TaskStatus.COMPLETE
            self.completed_ids.add(task.task_id)
            self.world.robots[robot_id].current_order = None
            self._release_pallet(robot_id)
            self.states[robot_id] = RobotState()

    def _advance_movement_cache(
        self,
        chosen: Dict[int, List[TimedPosition]],
    ) -> None:
        for robot_id, trajectory in chosen.items():
            state = self.states[robot_id]
            if state.movement is None:
                continue
            remaining = list(trajectory[1:])
            state.movement = (
                MovementPlan(state.movement.goal, remaining)
                if remaining
                else None
            )

    def solve(self) -> List[Action]:
        if not self.target_ids:
            return []

        while self.completed_ids != self.target_ids:
            if self.world.timestep >= self.max_timesteps:
                raise RuntimeError("Multi-robot solver exceeded max_timesteps")

            self._assign_free_robots()
            intents = {
                robot_id: self._intent(robot_id)
                for robot_id in sorted(self.world.robots)
            }
            move_actions, chosen = self._plan_moves(intents)

            fixed_actions = [
                Action(self.world.timestep, robot_id, intent.action, intent.target)
                for robot_id, intent in sorted(intents.items())
                if intent.action is not None and intent.target is not None
            ]
            timestep_actions = sorted(
                fixed_actions + move_actions,
                key=lambda action: action.robot_id,
            )

            self.simulator.step(timestep_actions)
            self.actions.extend(timestep_actions)

            for robot_id, intent in intents.items():
                if intent.action is not None:
                    self._post_action(robot_id, intent)
            self._advance_movement_cache(chosen)

        if len(self.queue):
            raise RuntimeError("Queue is nonempty after all tasks completed")
        if self.assigned_ids != self.target_ids:
            raise RuntimeError("Not every queued task was assigned")
        if any(not self.world.orders[i].fulfilled for i in self.target_ids):
            raise RuntimeError("Not every requested challenge order was fulfilled")
        if self.pallet_claims:
            raise RuntimeError("Pallet claims remain after solve completion")

        for robot_id in self.active_robot_ids:
            robot = self.world.robots[robot_id]
            if robot.storage or robot.docked_pallets or self.states[robot_id].task is not None:
                raise RuntimeError(f"Robot {robot_id} did not finish cleanly")

        self.world.validate()
        return list(self.actions)