"""Generate and replay aisle-aware multi-robot solutions on BIG_ORDER."""

from argparse import ArgumentParser
from collections import Counter
from pathlib import Path
import sys
import threading


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.aisle_solver import AisleAwareSolver
from src.metrics import analyze_actions, format_metrics_report, write_metrics_json
from src.models import ActionType
from src.parser import parse_problem
from src.simulator import Simulator
from src.world import WorldState
from src.writer import write_submission


BIG_ORDER_PATH = REPO_ROOT / "source_material" / "BIG_ORDER.txt"
MAX_TIMESTEPS_ERROR = "Multi-robot solver exceeded max_timesteps"


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
    print(f"Snapshot at t={solver.world.timestep}:")
    for robot_id in solver.active_robot_ids:
        robot = solver.world.robots[robot_id]
        state = solver.states[robot_id]
        order_id = state.task.order_id if state.task is not None else None
        movement_goal = state.movement.goal if state.movement is not None else None
        print(
            f"  robot {robot_id}: pos={robot.position} order={order_id} "
            f"phase={state.phase} aisle={state.active_aisle_id} "
            f"stop={state.aisle_stop_index} pallet={state.pallet_id} "
            f"pickup={state.pickup} row_goal={state.row_goal} "
            f"move_goal={movement_goal}"
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
    args = parser.parse_args()

    if args.stop_timestep is not None and args.stop_timestep <= 0:
        parser.error("--stop-timestep must be positive")

    robot_ids = list(range(args.robots))
    order_ids = list(range(args.orders))

    world = WorldState(parse_problem(BIG_ORDER_PATH))
    solver_kwargs = {}
    if args.stop_timestep is not None:
        solver_kwargs["max_timesteps"] = args.stop_timestep

    solver = AisleAwareSolver(
        world,
        robot_ids=robot_ids,
        order_ids=order_ids,
        **solver_kwargs,
    )
    label = f"aisle-v1 / {args.robots} robots / {args.orders} orders"
    if args.stop_timestep is not None:
        label += f" / stop t={args.stop_timestep}"

    actions = solve_with_progress(solver, label, args.stop_timestep)

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

    suffix = f"_{args.stop_timestep}t" if args.stop_timestep is not None else ""
    output_path = REPO_ROOT / "outputs" / (
        f"aisle_v1_{args.robots}r_{args.orders}o{suffix}.txt"
    )
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


if __name__ == "__main__":
    main()
