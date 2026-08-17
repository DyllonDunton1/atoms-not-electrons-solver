"""Generate and replay FIFO multi-robot baselines on real BIG_ORDER orders."""

from argparse import ArgumentParser
from collections import Counter
from pathlib import Path
import sys
import threading


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models import ActionType
from src.multi_robot_solver import MultiRobotSolver
from src.parser import parse_problem
from src.simulator import Simulator
from src.world import WorldState
from src.writer import write_submission


BIG_ORDER_PATH = REPO_ROOT / "source_material" / "BIG_ORDER.txt"


def solve_with_progress(solver: MultiRobotSolver, label: str):
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
                active[robot_id] = f"order {state.task.order_id} / {state.phase}"

        print(
            f"[{label}] t={solver.world.timestep} "
            f"completed={len(solver.completed_ids)}/{len(solver.target_ids)} "
            f"assigned={len(solver.assigned_ids)} "
            f"queued={len(solver.queue)} "
            f"active={active}",
            flush=True,
        )

    if "exception" in error:
        raise error["exception"]
    return result["actions"]


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--robots", type=int, default=5, choices=range(1, 6))
    parser.add_argument("--orders", type=int, default=10, choices=range(1, 1001))
    args = parser.parse_args()

    robot_ids = list(range(args.robots))
    order_ids = list(range(args.orders))

    world = WorldState(parse_problem(BIG_ORDER_PATH))
    solver = MultiRobotSolver(
        world,
        robot_ids=robot_ids,
        order_ids=order_ids,
    )
    label = f"{args.robots} robots / {args.orders} orders"
    actions = solve_with_progress(solver, label)

    print("Generation complete. Replaying from fresh input...", flush=True)

    # Independent replay from untouched input catches schedule-generation bugs.
    replay_world = WorldState(parse_problem(BIG_ORDER_PATH))
    Simulator(replay_world).run(actions)
    replay_world.validate()

    if not all(replay_world.orders[i].fulfilled for i in order_ids):
        raise RuntimeError("Fresh replay did not fulfill every requested order")

    action_keys = [(action.timestep, action.robot_id) for action in actions]
    if len(action_keys) != len(set(action_keys)):
        raise RuntimeError("Generated duplicate (timestep, robot) actions")

    output_path = REPO_ROOT / "outputs" / (
        f"fifo_baseline_{args.robots}r_{args.orders}o.txt"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_submission(actions, output_path)

    assignment_counts = Counter(
        world.orders[i].assigned_robot
        for i in order_ids
    )
    action_counts = Counter(action.action for action in actions)
    final_timestep = max((action.timestep for action in actions), default=-1) + 1

    print(f"Robots: {robot_ids}")
    print(f"Orders fulfilled: {args.orders}")
    print(f"Elapsed timesteps: {final_timestep}")
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
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
