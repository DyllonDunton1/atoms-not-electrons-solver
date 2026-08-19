"""Run the independent full-horizon scheduled solver on BIG_ORDER."""

from argparse import ArgumentParser
from collections import Counter
from pathlib import Path
import sys
import time


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scheduled_solver import SchedulerConfig, ScheduledSolver
from scheduled_solver.metrics import planning_metrics, write_metrics_json as write_planner_metrics
from scheduled_solver.parser import parse_problem as parse_scheduled_problem
from scheduled_solver.writer import write_submission as write_scheduled_submission

# Deliberate boundary: legacy code is used only after planning as an independent
# replay validator / apples-to-apples metrics implementation.
from src.metrics import analyze_actions, format_metrics_report, write_metrics_json
from src.models import Action as LegacyAction
from src.models import ActionType as LegacyActionType
from src.parser import parse_problem as parse_legacy_problem
from src.simulator import Simulator
from src.world import WorldState


BIG_ORDER_PATH = REPO_ROOT / "source_material" / "BIG_ORDER.txt"


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--robots", type=int, default=5, choices=range(1, 6))
    parser.add_argument("--orders", type=int, default=1000, choices=range(1, 1001))
    parser.add_argument("--beam-width", type=int, default=8)
    parser.add_argument("--candidate-width", type=int, default=8)
    parser.add_argument("--padding", type=int, default=1)
    parser.add_argument("--path-horizon", type=int, default=512)
    parser.add_argument("--max-path-expansions", type=int, default=250_000)
    args = parser.parse_args()

    if args.beam_width <= 0:
        parser.error("--beam-width must be positive")
    if args.candidate_width <= 0:
        parser.error("--candidate-width must be positive")
    if args.padding < 0:
        parser.error("--padding must be nonnegative")

    robot_ids = list(range(args.robots))
    order_ids = list(range(args.orders))
    config = SchedulerConfig(
        beam_width=args.beam_width,
        candidate_width=args.candidate_width,
        reservation_padding=args.padding,
        path_horizon=args.path_horizon,
        max_path_expansions=args.max_path_expansions,
        max_beam_depth=64,
        require_24_columns=True,
    )

    problem = parse_scheduled_problem(BIG_ORDER_PATH)
    solver = ScheduledSolver(
        problem,
        robot_ids=robot_ids,
        order_ids=order_ids,
        config=config,
    )
    run_started = time.perf_counter()

    def progress(done, total, schedule):
        if done <= 5 or done % 10 == 0 or done == total:
            elapsed = time.perf_counter() - run_started
            print(
                f"planned {done}/{total} orders | r{schedule.robot_id} "
                f"order={schedule.order_id} finish={schedule.finish_timestep} "
                f"columns={len(schedule.column_visits)} | "
                f"wall={elapsed:.1f}s ({elapsed / done:.2f}s/order) | "
                f"A*={solver.stats.astar_seconds:.1f}s "
                f"inv={solver.stats.inventory_seconds:.1f}s "
                f"cand={solver.stats.candidate_seconds:.1f}s "
                f"compact={solver.stats.compaction_seconds:.2f}s | "
                f"fast-row={solver.stats.row_fast_path_hits} "
                f"skipped={solver.stats.candidate_expansions_skipped}",
                flush=True,
            )

    print(
        f"Scheduling {args.orders} orders with {args.robots} robots, "
        f"beam={args.beam_width}, candidates={args.candidate_width}, "
        f"padding={args.padding}...",
        flush=True,
    )
    actions = solver.solve(progress_callback=progress)

    output_stem = (
        f"scheduled_v1_full_horizon_beam{args.beam_width}_cand{args.candidate_width}_"
        f"pad{args.padding}_{args.robots}r_{args.orders}o"
    )
    outputs_dir = REPO_ROOT / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    output_path = outputs_dir / f"{output_stem}.txt"
    metrics_path = outputs_dir / f"{output_stem}_metrics.json"
    planner_metrics_path = outputs_dir / f"{output_stem}_planner_metrics.json"

    write_scheduled_submission(actions, output_path)

    legacy_actions = [
        LegacyAction(
            action.timestep,
            action.robot_id,
            LegacyActionType(action.action.value),
            action.target,
        )
        for action in actions
    ]

    print("Planning complete. Replaying with legacy authoritative simulator...", flush=True)
    replay_world = WorldState(parse_legacy_problem(BIG_ORDER_PATH))
    replay_simulator = Simulator(replay_world)
    replay_simulator.run(legacy_actions)
    replay_world.validate()
    if not all(replay_world.orders[i].fulfilled for i in order_ids):
        raise RuntimeError("Fresh legacy replay did not fulfill every requested order")

    metrics_report = analyze_actions(
        legacy_actions,
        parse_legacy_problem(BIG_ORDER_PATH),
        robot_ids=robot_ids,
        end_timestep=replay_world.timestep,
    )
    write_metrics_json(metrics_report, metrics_path)
    write_planner_metrics(planning_metrics(actions, solver.stats), planner_metrics_path)

    assignment_counts = Counter(solver.assignment.values())
    print("Fresh replay: valid")
    print(f"Orders fulfilled: {args.orders}/{args.orders}")
    print(f"Makespan: {replay_world.timestep}")
    print(f"Actions written: {len(actions)}")
    print(f"Assignments by robot: {dict(sorted(assignment_counts.items()))}")
    print(
        f"Planner: {solver.stats.planning_seconds:.2f}s, "
        f"beam expansions={solver.stats.beam_expansions}, "
        f"A* calls={solver.stats.astar_calls}, "
        f"A* expansions={solver.stats.astar_expansions}, "
        f"A* seconds={solver.stats.astar_seconds:.2f}, "
        f"inventory seconds={solver.stats.inventory_seconds:.2f}, "
        f"candidate seconds={solver.stats.candidate_seconds:.2f}, "
        f"compaction seconds={solver.stats.compaction_seconds:.2f}, "
        f"row fast paths={solver.stats.row_fast_path_hits}, "
        f"candidate expansions skipped={solver.stats.candidate_expansions_skipped}"
    )
    print(format_metrics_report(metrics_report))
    print(f"Wrote {output_path}")
    print(f"Wrote {metrics_path}")
    print(f"Wrote {planner_metrics_path}")


if __name__ == "__main__":
    main()
