"""Run the distinct-SKU aisle-scoring experiment on BIG_ORDER.

This script intentionally leaves the production aisle planner unchanged.  It
swaps in an experimental planner that scores aisle utility by the number of
distinct required SKUs available in the aisle instead of total item quantity.
It always writes a full diagnostic trace and automatically runs the streaming
trace analyzer after the solution is generated.
"""

from argparse import ArgumentParser
from collections import Counter
from dataclasses import replace
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
from src.aisles import AislePlanner
from src.metrics import analyze_actions, format_metrics_report, write_metrics_json
from src.models import ActionType
from src.parser import parse_problem
from src.simulator import Simulator
from src.world import WorldState
from src.writer import write_submission


class DistinctSkuAislePlanner(AislePlanner):
    """Experimental planner whose score rewards distinct SKU coverage."""

    def choose_plan(
        self,
        start,
        remaining_by_sku,
        *,
        congestion_by_aisle,
        unavailable_pallet_ids=(),
        blocked=(),
        excluded_aisle_ids=(),
    ):
        """Shortlist aisles by distinct-SKU coverage, then detail-plan the top set."""
        unavailable = set(unavailable_pallet_ids)
        blocked_set = set(blocked)
        excluded = set(excluded_aisle_ids)
        cheap_candidates = []

        for aisle in self.layout.aisles:
            if aisle.aisle_id in excluded:
                continue

            options_by_sku = self._options_by_sku(
                aisle,
                remaining_by_sku,
                unavailable,
            )
            if not options_by_sku:
                continue

            # Experimental change: one useful SKU contributes one unit of
            # utility regardless of how many copies of that SKU are required.
            useful_sku_count = len(options_by_sku)
            estimated_distance = min(
                abs(start[0] - option.pickup[0])
                + abs(start[1] - option.pickup[1])
                for options in options_by_sku.values()
                for option in options
            )
            congestion = congestion_by_aisle.get(aisle.aisle_id, 0)
            cheap_score = self._score(
                useful_sku_count,
                estimated_distance,
                congestion,
            )
            cheap_candidates.append((cheap_score, aisle.aisle_id))

        cheap_candidates.sort(key=lambda item: (-item[0], item[1]))

        detailed = []
        for _, aisle_id in cheap_candidates:
            plan = self.plan_aisle(
                aisle_id,
                start,
                remaining_by_sku,
                congestion=congestion_by_aisle.get(aisle_id, 0),
                unavailable_pallet_ids=unavailable,
                blocked=blocked_set,
            )
            if plan is not None:
                detailed.append(plan)
                if len(detailed) == self.top_candidates:
                    break

        if not detailed:
            return None

        detailed.sort(
            key=lambda plan: (
                -plan.score,
                plan.planned_distance,
                -plan.useful_quantity,
                plan.aisle_id,
            )
        )
        return detailed[0]

    def plan_aisle(self, *args, **kwargs):
        """Use the normal route builder, but rescore the completed route by SKU count."""
        plan = super().plan_aisle(*args, **kwargs)
        if plan is None:
            return None

        # A valid baseline aisle plan contains exactly one stop per useful SKU,
        # so len(stops) is the detailed-stage distinct-SKU utility.
        return replace(
            plan,
            score=self._score(
                len(plan.stops),
                plan.planned_distance,
                plan.congestion,
            ),
        )


class DistinctSkuDiagnosticSolver(DiagnosticAisleAwareSolver):
    """Normal diagnostic solver with only the aisle utility heuristic swapped."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.aisle_planner = DistinctSkuAislePlanner(self.world)


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--robots", type=int, default=5, choices=range(1, 6))
    parser.add_argument("--orders", type=int, default=1000, choices=range(1, 1001))
    args = parser.parse_args()

    robot_ids = list(range(args.robots))
    order_ids = list(range(args.orders))

    # Deliberately use a different stem from the known-good aisle_v1 output so
    # this experiment can never overwrite the current working solution files.
    output_stem = f"aisle_v1_distinct_skus_{args.robots}r_{args.orders}o"
    outputs_dir = REPO_ROOT / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    output_path = outputs_dir / f"{output_stem}.txt"
    metrics_path = outputs_dir / f"{output_stem}_metrics.json"
    trace_path = outputs_dir / f"{output_stem}_trace.jsonl"

    world = WorldState(parse_problem(BIG_ORDER_PATH))
    solver = DistinctSkuDiagnosticSolver(
        world,
        robot_ids=robot_ids,
        order_ids=order_ids,
        trace_path=trace_path,
        trace_start=0,
    )

    label = (
        f"distinct-SKU scoring / {args.robots} robots / "
        f"{args.orders} orders / full trace"
    )

    try:
        actions = solve_with_progress(solver, label)
    finally:
        solver.close_trace()

    print("Generation complete. Replaying from fresh input...", flush=True)
    replay_problem = parse_problem(BIG_ORDER_PATH)
    replay_world = WorldState(replay_problem)
    Simulator(replay_world).run(actions)
    replay_world.validate()

    if not all(replay_world.orders[i].fulfilled for i in order_ids):
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
