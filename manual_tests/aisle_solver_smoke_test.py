"""Generate and replay aisle-aware multi-robot solutions on BIG_ORDER."""

from argparse import ArgumentParser
from collections import Counter
import json
from pathlib import Path
import sys
import threading


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.aisle_solver import AisleAwareSolver, PERSISTENT_ADJACENCY_TIMESTEPS
from src.metrics import analyze_actions, format_metrics_report, write_metrics_json
from src.models import ActionType
from src.parser import parse_problem
from src.scheduler import ReservationTable
from src.simulator import Simulator
from src.world import WorldState
from src.writer import write_submission


BIG_ORDER_PATH = REPO_ROOT / "source_material" / "BIG_ORDER.txt"
MAX_TIMESTEPS_ERROR = "Multi-robot solver exceeded max_timesteps"


class DiagnosticAisleAwareSolver(AisleAwareSolver):
    """Aisle solver that records read-only traffic decisions for diagnosis."""

    def __init__(
        self,
        *args,
        trace_path=None,
        trace_start=None,
        trace_end=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.last_plan_timestep = None
        self.last_intents = {}
        self.last_preferred_next = {}
        self.last_committed_next = {}
        self.last_traffic_decisions = {}

        self.trace_path = Path(trace_path) if trace_path is not None else None
        self.trace_start = 0 if trace_start is None else trace_start
        self.trace_end = trace_end
        self._trace_file = None
        if self.trace_path is not None:
            self.trace_path.parent.mkdir(parents=True, exist_ok=True)
            self._trace_file = self.trace_path.open("w", encoding="utf-8", buffering=1)

    def close_trace(self) -> None:
        if self._trace_file is not None:
            self._trace_file.close()
            self._trace_file = None

    def _trace_enabled(self, timestep: int) -> bool:
        if self._trace_file is None:
            return False
        if timestep < self.trace_start:
            return False
        if self.trace_end is not None and timestep > self.trace_end:
            return False
        return True

    def _blocking_robot_ids(self, robot_id, destination):
        destination_cells = self._footprint_cells_at(robot_id, destination)
        return [
            other_id
            for other_id in sorted(self.world.robots)
            if other_id != robot_id
            and destination_cells & self._footprint_cells(other_id)
        ]

    def _traffic_decision_record(
        self,
        robot_id,
        intents,
        move_by_robot,
        committed_destinations,
        reservations,
    ):
        """Reconstruct why the normal one-step planner moved or waited."""
        robot = self.world.robots[robot_id]
        intent = intents[robot_id]
        start = robot.position
        footprint = self._footprint(robot_id)
        goal = intent.move_goal
        move_action = move_by_robot.get(robot_id)

        path = []
        candidate = None
        physical_free = None
        reservation_free = None
        blocking_robot_ids = []
        priority_blocked_cells = []

        if goal is not None:
            priority_blocked_cells = sorted(
                self._priority_blocked_cells(robot_id, committed_destinations)
            )
            path = self._preferred_path(
                robot_id,
                goal,
                committed_destinations,
            )
            if len(path) >= 2:
                candidate = path[1]
                physical_free = self._first_step_is_physically_free(
                    robot_id,
                    candidate,
                )
                reservation_free = reservations.transition_is_free(
                    self.world.timestep,
                    start,
                    candidate,
                    footprint,
                )
                blocking_robot_ids = self._blocking_robot_ids(
                    robot_id,
                    candidate,
                )

        destination = move_action.target if move_action is not None else start

        if move_action is not None:
            result = "move"
        elif intent.action is not None:
            result = f"fixed_{intent.action.value}"
        elif goal is None:
            result = "no_intent"
        elif path and len(path) == 1 and start == goal:
            result = "at_goal"
        elif not path:
            result = "no_path"
        elif candidate is None:
            result = "no_first_step"
        elif not physical_free:
            result = "blocked_current_occupancy"
        elif not reservation_free:
            result = "blocked_reservation"
        else:
            result = "wait_unexpected"

        decision = {
            "start": start,
            "goal": goal,
            "path_steps": len(path) - 1 if path else None,
            "preferred_next": candidate,
            "physical_free": physical_free,
            "reservation_free": reservation_free,
            "blocking_robot_ids": blocking_robot_ids,
            "priority_blocked_cells": priority_blocked_cells,
            "committed_next": destination,
            "result": result,
        }

        reservations.reserve_transition(
            self.world.timestep,
            start,
            destination,
            footprint,
        )
        committed_destinations[robot_id] = destination
        return decision

    @staticmethod
    def _stop_record(stop):
        if stop is None:
            return None
        return {
            "sku": stop.sku,
            "quantity": stop.quantity,
            "pallet_id": stop.pallet_id,
            "pickup": stop.pickup,
        }

    def _robot_trace_record(self, robot_id, intent, traffic):
        robot = self.world.robots[robot_id]
        state = self.states[robot_id]
        current_stop = self._current_stop(robot_id)

        remaining_plan_stops = []
        if state.aisle_plan is not None:
            remaining_plan_stops = [
                self._stop_record(stop)
                for stop in state.aisle_plan.stops[state.aisle_stop_index :]
            ]

        target_pallet = None
        if state.pallet_id is not None:
            pallet = self.world.pallets[state.pallet_id]
            target_pallet = {
                "pallet_id": pallet.pallet_id,
                "position": pallet.position,
                "sku": pallet.sku,
                "count": pallet.count,
                "max_count": pallet.max_count,
                "docked_to": pallet.docked_to,
                "claim_owner": self.pallet_claims.get(pallet.pallet_id),
            }

        adjacent_pallets = []
        for pallet in self.world.pallets.values():
            distance = (
                abs(robot.position[0] - pallet.position[0])
                + abs(robot.position[1] - pallet.position[1])
            )
            if distance == 1:
                adjacent_pallets.append(
                    {
                        "pallet_id": pallet.pallet_id,
                        "position": pallet.position,
                        "sku": pallet.sku,
                        "count": pallet.count,
                        "docked_to": pallet.docked_to,
                        "claim_owner": self.pallet_claims.get(pallet.pallet_id),
                    }
                )
        adjacent_pallets.sort(key=lambda item: item["pallet_id"])

        persistent_blocked = sorted(
            pallet_id
            for (higher_priority_id, pallet_id), streak in (
                self._priority_adjacency_streaks.items()
            )
            if higher_priority_id < robot_id
            and streak >= PERSISTENT_ADJACENCY_TIMESTEPS
        )

        return {
            "position": robot.position,
            "footprint": sorted(self._footprint_cells(robot_id)),
            "docked_pallets": list(robot.docked_pallets),
            "order_id": state.task.order_id if state.task is not None else None,
            "phase": state.phase,
            "active_aisle_id": state.active_aisle_id,
            "aisle_stop_index": state.aisle_stop_index,
            "current_stop": self._stop_record(current_stop),
            "remaining_plan_stops": remaining_plan_stops,
            "target_pallet": target_pallet,
            "pickup": state.pickup,
            "row_goal": state.row_goal,
            "refill_robot_home": state.refill_robot_home,
            "refill_pallet_home": state.refill_pallet_home,
            "remaining": state.remaining,
            "remaining_by_sku": dict(sorted(state.remaining_by_sku.items())),
            "deferred_pallet_ids": sorted(state.deferred_pallet_ids),
            "persistent_blocked_pallet_ids": persistent_blocked,
            "adjacent_pallets": adjacent_pallets,
            "intent": {
                "action": intent.action.value if intent.action is not None else None,
                "target": intent.target,
                "move_goal": intent.move_goal,
            },
            "traffic": traffic,
        }

    def _write_trace_record(self, intents, decisions) -> None:
        if not self._trace_enabled(self.world.timestep):
            return

        adjacency_streaks = [
            {
                "robot_id": robot_id,
                "pallet_id": pallet_id,
                "streak": streak,
            }
            for (robot_id, pallet_id), streak in sorted(
                self._priority_adjacency_streaks.items()
            )
        ]

        record = {
            "timestep": self.world.timestep,
            "completed_orders": len(self.completed_ids),
            "assigned_orders": len(self.assigned_ids),
            "queued_orders": len(self.queue),
            "pallet_claims": dict(sorted(self.pallet_claims.items())),
            "priority_adjacency_streaks": adjacency_streaks,
            "robots": {
                str(robot_id): self._robot_trace_record(
                    robot_id,
                    intents[robot_id],
                    decisions[robot_id],
                )
                for robot_id in sorted(self.world.robots)
            },
        }
        self._trace_file.write(json.dumps(record, separators=(",", ":")) + "\n")

    def _plan_moves(self, intents):
        """Run normal planning, then record exact per-robot traffic diagnostics."""
        actions = super()._plan_moves(intents)
        move_by_robot = {action.robot_id: action for action in actions}
        committed_destinations = {}
        reservations = ReservationTable()
        decisions = {}

        for robot_id in sorted(self.world.robots):
            decisions[robot_id] = self._traffic_decision_record(
                robot_id,
                intents,
                move_by_robot,
                committed_destinations,
                reservations,
            )

        self.last_plan_timestep = self.world.timestep
        self.last_intents = dict(intents)
        self.last_traffic_decisions = decisions
        self.last_preferred_next = {
            robot_id: decision["preferred_next"]
            for robot_id, decision in decisions.items()
        }
        self.last_committed_next = {
            robot_id: decision["committed_next"]
            for robot_id, decision in decisions.items()
        }
        self._write_trace_record(intents, decisions)
        return actions


def solve_with_progress(solver: AisleAwareSolver, label: str, stop_timestep=None):
    result = {}
    error = {}

    def run_solver() -> None:
        try:
            result["actions"] = solver.solve()
        except BaseException as exception:
            error["exception"] = exception

    thread = threading.Thread(target=run_solver, daemon=True)
    thread.start()

    while thread.is_alive():
        thread.join(timeout=2.0)
        if not thread.is_alive():
            break

        active = {}
        for robot_id in solver.active_robot_ids:
            state = solver.states[robot_id]
            if state.task is None:
                active[robot_id] = "idle"
            else:
                active[robot_id] = (
                    f"order {state.task.order_id} / {state.phase} "
                    f"/ aisle {state.active_aisle_id}"
                )

        print(
            f"[{label}] t={solver.world.timestep} "
            f"completed={len(solver.completed_ids)}/{len(solver.target_ids)} "
            f"assigned={len(solver.assigned_ids)} "
            f"queued={len(solver.queue)} "
            f"active={active}",
            flush=True,
        )

    if "exception" in error:
        exception = error["exception"]
        stopped_as_requested = (
            stop_timestep is not None
            and isinstance(exception, RuntimeError)
            and str(exception) == MAX_TIMESTEPS_ERROR
            and solver.world.timestep >= stop_timestep
        )
        if not stopped_as_requested:
            raise exception

        print(f"Reached requested stop timestep t={solver.world.timestep}.", flush=True)
        return list(solver.actions)

    return result["actions"]


def print_snapshot(solver: AisleAwareSolver) -> None:
    """Print a read-only diagnostic snapshot without invoking replanning logic."""
    print(f"Snapshot at t={solver.world.timestep}:")
    print(f"  pallet_claims={dict(sorted(solver.pallet_claims.items()))}")

    if isinstance(solver, DiagnosticAisleAwareSolver):
        print(f"  last_traffic_plan_timestep={solver.last_plan_timestep}")

    for robot_id in solver.active_robot_ids:
        robot = solver.world.robots[robot_id]
        state = solver.states[robot_id]
        order_id = state.task.order_id if state.task is not None else None
        footprint = sorted(solver._footprint_cells(robot_id))
        current_stop = solver._current_stop(robot_id)

        if current_stop is None:
            stop_description = None
        else:
            stop_description = (
                f"sku={current_stop.sku} qty={current_stop.quantity} "
                f"pallet={current_stop.pallet_id} pickup={current_stop.pickup}"
            )

        current_pallet = None
        if state.pallet_id is not None:
            pallet = solver.world.pallets[state.pallet_id]
            current_pallet = (
                f"id={pallet.pallet_id} pos={pallet.position} sku={pallet.sku} "
                f"stock={pallet.count}/{pallet.max_count} "
                f"docked_to={pallet.docked_to}"
            )

        adjacent_pallets = []
        for pallet in solver.world.pallets.values():
            distance = (
                abs(robot.position[0] - pallet.position[0])
                + abs(robot.position[1] - pallet.position[1])
            )
            if distance == 1:
                adjacent_pallets.append(
                    (
                        pallet.pallet_id,
                        pallet.position,
                        pallet.sku,
                        pallet.count,
                        pallet.docked_to,
                        solver.pallet_claims.get(pallet.pallet_id),
                    )
                )
        adjacent_pallets.sort()

        nearby_robots = {
            other_id: solver.world.robots[other_id].position
            for other_id in solver.active_robot_ids
            if other_id != robot_id
            and (
                abs(robot.position[0] - solver.world.robots[other_id].position[0])
                + abs(robot.position[1] - solver.world.robots[other_id].position[1])
                <= 3
            )
        }

        intent_description = None
        preferred_next = None
        committed_next = None
        traffic_result = None
        if isinstance(solver, DiagnosticAisleAwareSolver):
            intent = solver.last_intents.get(robot_id)
            if intent is not None:
                action_name = intent.action.value if intent.action is not None else None
                intent_description = (
                    f"action={action_name} target={intent.target} "
                    f"move_goal={intent.move_goal}"
                )
            preferred_next = solver.last_preferred_next.get(robot_id)
            committed_next = solver.last_committed_next.get(robot_id)
            decision = solver.last_traffic_decisions.get(robot_id)
            if decision is not None:
                traffic_result = decision["result"]

        print(
            f"  robot {robot_id}: pos={robot.position} footprint={footprint} "
            f"docked={list(robot.docked_pallets)}"
        )
        print(
            f"    order={order_id} phase={state.phase} "
            f"aisle={state.active_aisle_id} stop_index={state.aisle_stop_index}"
        )
        print(
            f"    current_stop={stop_description} current_pallet={current_pallet}"
        )
        print(
            f"    pickup={state.pickup} row_goal={state.row_goal} "
            f"refill_robot_home={state.refill_robot_home} "
            f"refill_pallet_home={state.refill_pallet_home}"
        )
        print(
            f"    last_intent={intent_description} "
            f"preferred_next={preferred_next} committed_next={committed_next} "
            f"traffic_result={traffic_result}"
        )
        print(
            f"    remaining={state.remaining} "
            f"remaining_by_sku={dict(sorted(state.remaining_by_sku.items()))}"
        )
        print(
            f"    adjacent_pallets={adjacent_pallets} "
            f"nearby_robots={nearby_robots}"
        )


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--robots", type=int, default=5, choices=range(1, 6))
    parser.add_argument("--orders", type=int, default=10, choices=range(1, 1001))
    parser.add_argument(
        "--stop-timestep",
        type=int,
        default=None,
        help="Stop generation at this world timestep and write the partial schedule.",
    )
    parser.add_argument(
        "--trace-start",
        type=int,
        default=None,
        help="First timestep to write to the per-timestep JSONL diagnostic trace.",
    )
    parser.add_argument(
        "--trace-end",
        type=int,
        default=None,
        help="Last timestep to write to the per-timestep JSONL diagnostic trace.",
    )
    args = parser.parse_args()

    if args.stop_timestep is not None and args.stop_timestep <= 0:
        parser.error("--stop-timestep must be positive")
    if args.trace_start is not None and args.trace_start < 0:
        parser.error("--trace-start must be nonnegative")
    if args.trace_end is not None and args.trace_end < 0:
        parser.error("--trace-end must be nonnegative")
    if (
        args.trace_start is not None
        and args.trace_end is not None
        and args.trace_end < args.trace_start
    ):
        parser.error("--trace-end must be >= --trace-start")

    robot_ids = list(range(args.robots))
    order_ids = list(range(args.orders))
    suffix = f"_{args.stop_timestep}t" if args.stop_timestep is not None else ""
    output_stem = f"aisle_v1_{args.robots}r_{args.orders}o{suffix}"
    trace_requested = args.trace_start is not None or args.trace_end is not None
    trace_path = (
        REPO_ROOT / "outputs" / f"{output_stem}_trace.jsonl"
        if trace_requested
        else None
    )

    world = WorldState(parse_problem(BIG_ORDER_PATH))
    solver_kwargs = {}
    if args.stop_timestep is not None:
        solver_kwargs["max_timesteps"] = args.stop_timestep

    solver_class = (
        DiagnosticAisleAwareSolver
        if args.stop_timestep is not None or trace_requested
        else AisleAwareSolver
    )
    if solver_class is DiagnosticAisleAwareSolver:
        solver_kwargs.update(
            trace_path=trace_path,
            trace_start=args.trace_start,
            trace_end=args.trace_end,
        )

    solver = solver_class(
        world,
        robot_ids=robot_ids,
        order_ids=order_ids,
        **solver_kwargs,
    )
    label = f"aisle-v1 / {args.robots} robots / {args.orders} orders"
    if args.stop_timestep is not None:
        label += f" / stop t={args.stop_timestep}"
    if trace_requested:
        label += f" / trace {args.trace_start or 0}..{args.trace_end or 'end'}"

    try:
        actions = solve_with_progress(solver, label, args.stop_timestep)
    finally:
        if isinstance(solver, DiagnosticAisleAwareSolver):
            solver.close_trace()

    if args.stop_timestep is not None:
        print_snapshot(solver)

    print("Generation complete. Replaying from fresh input...", flush=True)

    replay_world = WorldState(parse_problem(BIG_ORDER_PATH))
    replay_simulator = Simulator(replay_world)
    replay_simulator.run(actions)

    if args.stop_timestep is not None:
        while replay_world.timestep < args.stop_timestep:
            replay_simulator.step([])

    replay_world.validate()

    if args.stop_timestep is None and not all(
        replay_world.orders[i].fulfilled for i in order_ids
    ):
        raise RuntimeError("Fresh replay did not fulfill every requested order")

    action_keys = [(action.timestep, action.robot_id) for action in actions]
    if len(action_keys) != len(set(action_keys)):
        raise RuntimeError("Generated duplicate (timestep, robot) actions")

    output_path = REPO_ROOT / "outputs" / f"{output_stem}.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_submission(actions, output_path)

    metrics_report = analyze_actions(
        actions,
        parse_problem(BIG_ORDER_PATH),
        robot_ids=robot_ids,
        end_timestep=replay_world.timestep,
    )
    metrics_path = output_path.with_name(output_path.stem + "_metrics.json")
    write_metrics_json(metrics_report, metrics_path)

    fulfilled_count = sum(world.orders[i].fulfilled for i in order_ids)
    assignment_counts = Counter(
        world.orders[i].assigned_robot
        for i in order_ids
        if world.orders[i].assigned_robot is not None
    )
    action_counts = Counter(action.action for action in actions)

    print(f"Robots: {robot_ids}")
    print(f"Orders fulfilled: {fulfilled_count}/{args.orders}")
    print(f"World timestep reached: {solver.world.timestep}")
    print(f"Actions written: {len(actions)}")
    print(f"Assignments by robot: {dict(sorted(assignment_counts.items()))}")
    print(
        "Action counts: "
        + ", ".join(
            f"{action_type.value}={action_counts[action_type]}"
            for action_type in ActionType
        )
    )
    print("Fresh replay: valid")
    print(format_metrics_report(metrics_report))
    print(f"Wrote {output_path}")
    print(f"Wrote {metrics_path}")
    if trace_path is not None:
        print(f"Wrote {trace_path}")


if __name__ == "__main__":
    main()
