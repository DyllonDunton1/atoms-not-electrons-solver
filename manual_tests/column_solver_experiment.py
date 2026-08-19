"""Run the 24-column / 48-directed-route collection experiment on BIG_ORDER.

This runner keeps the existing pathfinding, traffic, simulator, and baseline aisle
strategy untouched. It swaps only the collection strategy and writes distinct
output names so the known-good aisle solutions are never overwritten.
"""

from argparse import ArgumentParser
from collections import Counter
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from manual_tests.aisle_solver_smoke_test import (
    BIG_ORDER_PATH,
    DiagnosticAisleAwareSolver,
    solve_with_progress,
)
from src.column_solver import ColumnAwareSolver
from src.metrics import analyze_actions, format_metrics_report, write_metrics_json
from src.models import ActionType
from src.parser import parse_problem
from src.simulator import Simulator
from src.world import WorldState
from src.writer import write_submission


class DiagnosticColumnAwareSolver(ColumnAwareSolver, DiagnosticAisleAwareSolver):
    """Directed-column strategy with the existing read-only trace instrumentation."""



def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--robots", type=int, default=5, choices=range(1, 6))
    parser.add_argument("--orders", type=int, default=1000, choices=range(1, 1001))
    parser.add_argument(
        "--stop-timestep",
        type=int,
        default=None,
        help="Stop generation at this world timestep and write partial outputs.",
    )
    args = parser.parse_args()

    if args.stop_timestep is not None and args.stop_timestep <= 0:
        parser.error("--stop-timestep must be positive")

    robot_ids = list(range(args.robots))
    order_ids = list(range(args.orders))

    suffix = f"_{args.stop_timestep}t" if args.stop_timestep is not None else ""
    output_stem = f"column_v1_distinct_skus_{args.robots}r_{args.orders}o{suffix}"
    outputs_dir = REPO_ROOT / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    output_path = outputs_dir / f"{output_stem}.txt"
    metrics_path = outputs_dir / f"{output_stem}_metrics.json"
    trace_path = outputs_dir / f"{output_stem}_trace.jsonl"

    world = WorldState(parse_problem(BIG_ORDER_PATH))
    solver_kwargs = {}
    if args.stop_timestep is not None:
        solver_kwargs["max_timesteps"] = args.stop_timestep

    solver = DiagnosticColumnAwareSolver(
        world,
        robot_ids=robot_ids,
        order_ids=order_ids,
        trace_path=trace_path,
        trace_start=0,
        **solver_kwargs,
    )

    label = (
        f"directed-column scoring / {args.robots} robots / "
        f"{args.orders} orders / full trace"
    )
    if args.stop_timestep is not None:
        label += f" / stop t={args.stop_timestep}"

    try:
        actions = solve_with_progress(solver, label, args.stop_timestep)
    finally:
        solver.close_trace()

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

    write_submission(actions, output_path)
    metrics_report = analyze_actions(
        actions,
        parse_problem(BIG_ORDER_PATH),
        robot_ids=robot_ids,
        end_timestep=replay_world.timestep,
    )
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
    print(f"Wrote {trace_path}")

    analyzer_path = REPO_ROOT / "manual_tests" / "analyze_traffic_trace.py"
    print("Analyzing trace...", flush=True)
    subprocess.run(
        [
            sys.executable,
            str(analyzer_path),
            str(trace_path),
            "--min-stall",
            "10",
            "--top",
            "25",
            "--context",
            "4",
        ],
        check=True,
    )
    print("Trace analysis complete.", flush=True)


if __name__ == "__main__":
    main()
