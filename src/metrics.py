"""Analyze generated schedules without changing solver behavior."""

from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Union

from .aisles import build_aisle_layout
from .models import Action, ActionType, Position, ProblemInstance
from .parser import parse_problem


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROBLEM_PATH = REPO_ROOT / "source_material" / "BIG_ORDER.txt"


@dataclass
class RobotMetrics:
    robot_id: int
    actions: int
    waits: int
    initial_idle: int
    internal_waits: int
    terminal_idle: int
    moves: int
    collection_moves: int
    refill_moves: int
    fulfillment_moves: int
    picks: int
    docks: int
    undocks: int
    fulfills: int


@dataclass
class OrderMetrics:
    robot_id: int
    order_index_for_robot: int
    completed: bool
    collection_moves: int
    refill_moves: int
    fulfillment_moves: int
    picks: int
    refill_trips: int
    aisle_visits: int
    aisle_reentries: int


def read_submission(path: Union[str, Path]) -> List[Action]:
    """Read Tutor submission text into Action objects."""
    actions: List[Action] = []
    for line_number, raw_line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(
                f"Line {line_number} must contain 5 fields, found {len(parts)}"
            )
        timestep_text, robot_text, action_text, x_text, y_text = parts
        try:
            actions.append(
                Action(
                    timestep=int(timestep_text),
                    robot_id=int(robot_text),
                    action=ActionType(action_text),
                    target=(int(x_text), int(y_text)),
                )
            )
        except (TypeError, ValueError) as exception:
            raise ValueError(f"Invalid submission line {line_number}: {line}") from exception
    return actions


def _pallet_aisles(problem: ProblemInstance):
    """Return the shared deterministic pallet-home-to-aisle mapping."""
    layout = build_aisle_layout(problem.pallets)
    return dict(layout.home_to_aisle), len(layout.aisles)


def _segment_robot_actions(actions: Sequence[Action]) -> List[List[Action]]:
    """Split one robot's action history at fulfillment actions."""
    segments: List[List[Action]] = []
    current: List[Action] = []
    for action in actions:
        current.append(action)
        if action.action == ActionType.FULFILL:
            segments.append(current)
            current = []
    if current:
        segments.append(current)
    return segments


def _analyze_order_segment(
    robot_id: int,
    order_index: int,
    segment: Sequence[Action],
    pallet_home_to_aisle: Dict[Position, int],
) -> OrderMetrics:
    completed = bool(segment and segment[-1].action == ActionType.FULFILL)
    last_pick_index = max(
        (index for index, action in enumerate(segment) if action.action == ActionType.PICK),
        default=-1,
    )

    collection_moves = 0
    refill_moves = 0
    fulfillment_moves = 0
    refill_active = False
    refill_trips = 0
    serviced_aisles: List[int] = []

    for index, action in enumerate(segment):
        if action.action == ActionType.DOCK:
            refill_active = True
            refill_trips += 1
        elif action.action == ActionType.UNDOCK:
            refill_active = False
        elif action.action == ActionType.PICK:
            aisle_id = pallet_home_to_aisle.get(action.target)
            if aisle_id is not None:
                serviced_aisles.append(aisle_id)
        elif action.action == ActionType.MOVE:
            if refill_active:
                refill_moves += 1
            elif completed and index > last_pick_index:
                fulfillment_moves += 1
            else:
                collection_moves += 1

    compressed_aisles: List[int] = []
    for aisle_id in serviced_aisles:
        if not compressed_aisles or compressed_aisles[-1] != aisle_id:
            compressed_aisles.append(aisle_id)

    seen_aisles: Set[int] = set()
    aisle_reentries = 0
    for aisle_id in compressed_aisles:
        if aisle_id in seen_aisles:
            aisle_reentries += 1
        else:
            seen_aisles.add(aisle_id)

    return OrderMetrics(
        robot_id=robot_id,
        order_index_for_robot=order_index,
        completed=completed,
        collection_moves=collection_moves,
        refill_moves=refill_moves,
        fulfillment_moves=fulfillment_moves,
        picks=sum(action.action == ActionType.PICK for action in segment),
        refill_trips=refill_trips,
        aisle_visits=len(compressed_aisles),
        aisle_reentries=aisle_reentries,
    )


def analyze_actions(
    actions: Sequence[Action],
    problem: ProblemInstance,
    *,
    robot_ids: Optional[Iterable[int]] = None,
    end_timestep: Optional[int] = None,
) -> Dict[str, object]:
    """Return deterministic schedule metrics for a generated solution.

    ``end_timestep`` is the replay world's timestep after the schedule. Pass it
    explicitly for cutoff runs because waits omitted after the final explicit
    action cannot be inferred from the text file alone.
    """
    ordered = sorted(actions, key=lambda action: (action.timestep, action.robot_id))
    seen = set()
    for action in ordered:
        if action.timestep < 0:
            raise ValueError("Action timesteps must be nonnegative")
        key = (action.timestep, action.robot_id)
        if key in seen:
            raise ValueError(f"Duplicate action for robot {action.robot_id} at t={action.timestep}")
        seen.add(key)

    if robot_ids is None:
        inferred_ids = sorted({action.robot_id for action in ordered})
        robot_ids_list = inferred_ids or sorted(robot.robot_id for robot in problem.robots)
    else:
        robot_ids_list = sorted(set(robot_ids))
    if not robot_ids_list:
        raise ValueError("At least one robot id is required for metrics")

    unknown_robot_ids = set(robot_ids_list) - {robot.robot_id for robot in problem.robots}
    if unknown_robot_ids:
        raise ValueError(f"Unknown robot ids: {sorted(unknown_robot_ids)}")

    inferred_end = max((action.timestep for action in ordered), default=-1) + 1
    if end_timestep is None:
        end_timestep = inferred_end
    if end_timestep < inferred_end:
        raise ValueError("end_timestep cannot be before the final explicit action")

    active_id_set = set(robot_ids_list)
    foreign_actions = [action for action in ordered if action.robot_id not in active_id_set]
    if foreign_actions:
        raise ValueError(
            "Submission contains actions for robots outside the requested metric set"
        )

    pallet_home_to_aisle, aisle_count = _pallet_aisles(problem)
    by_robot: Dict[int, List[Action]] = defaultdict(list)
    for action in ordered:
        by_robot[action.robot_id].append(action)

    order_metrics: List[OrderMetrics] = []
    robot_metrics: List[RobotMetrics] = []
    aisle_pick_counts: Counter = Counter()

    for robot_id in robot_ids_list:
        robot_actions = by_robot.get(robot_id, [])
        segments = _segment_robot_actions(robot_actions)
        robot_orders = [
            _analyze_order_segment(
                robot_id,
                order_index,
                segment,
                pallet_home_to_aisle,
            )
            for order_index, segment in enumerate(segments)
        ]
        order_metrics.extend(robot_orders)

        for action in robot_actions:
            if action.action == ActionType.PICK:
                aisle_id = pallet_home_to_aisle.get(action.target)
                if aisle_id is not None:
                    aisle_pick_counts[aisle_id] += 1

        first_timestep = robot_actions[0].timestep if robot_actions else end_timestep
        last_timestep = robot_actions[-1].timestep if robot_actions else -1
        initial_idle = min(first_timestep, end_timestep)
        terminal_idle = max(0, end_timestep - (last_timestep + 1))
        span = max(0, last_timestep - first_timestep + 1)
        internal_waits = max(0, span - len(robot_actions))
        waits = end_timestep - len(robot_actions)

        robot_metrics.append(
            RobotMetrics(
                robot_id=robot_id,
                actions=len(robot_actions),
                waits=waits,
                initial_idle=initial_idle,
                internal_waits=internal_waits,
                terminal_idle=terminal_idle,
                moves=sum(action.action == ActionType.MOVE for action in robot_actions),
                collection_moves=sum(order.collection_moves for order in robot_orders),
                refill_moves=sum(order.refill_moves for order in robot_orders),
                fulfillment_moves=sum(order.fulfillment_moves for order in robot_orders),
                picks=sum(action.action == ActionType.PICK for action in robot_actions),
                docks=sum(action.action == ActionType.DOCK for action in robot_actions),
                undocks=sum(action.action == ActionType.UNDOCK for action in robot_actions),
                fulfills=sum(action.action == ActionType.FULFILL for action in robot_actions),
            )
        )

    action_counts = Counter(action.action for action in ordered)
    total_robot_timesteps = end_timestep * len(robot_ids_list)
    total_waits = total_robot_timesteps - len(ordered)
    movement = {
        "collection": sum(metric.collection_moves for metric in order_metrics),
        "refill": sum(metric.refill_moves for metric in order_metrics),
        "fulfillment": sum(metric.fulfillment_moves for metric in order_metrics),
    }

    report: Dict[str, object] = {
        "end_timestep": end_timestep,
        "robot_ids": robot_ids_list,
        "robot_count": len(robot_ids_list),
        "total_robot_timesteps": total_robot_timesteps,
        "explicit_actions": len(ordered),
        "wait_timesteps": total_waits,
        "action_counts": {
            action_type.value: action_counts[action_type]
            for action_type in ActionType
        },
        "movement": movement,
        "orders_completed": action_counts[ActionType.FULFILL],
        "refill_trips": action_counts[ActionType.DOCK],
        "aisle_count": aisle_count,
        "aisle_pick_counts": {
            str(aisle_id): aisle_pick_counts[aisle_id]
            for aisle_id in range(aisle_count)
        },
        "aisle_visits": sum(metric.aisle_visits for metric in order_metrics),
        "aisle_reentries": sum(metric.aisle_reentries for metric in order_metrics),
        "robots": [asdict(metric) for metric in robot_metrics],
        "orders": [asdict(metric) for metric in order_metrics],
    }
    return report


def write_metrics_json(report: Dict[str, object], path: Union[str, Path]) -> None:
    Path(path).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _percent(numerator: int, denominator: int) -> float:
    return 0.0 if denominator == 0 else 100.0 * numerator / denominator


def format_metrics_report(report: Dict[str, object]) -> str:
    """Render a compact human-readable baseline report."""
    total = int(report["total_robot_timesteps"])
    waits = int(report["wait_timesteps"])
    action_counts = report["action_counts"]
    movement = report["movement"]
    lines = [
        "",
        "=== Solution metrics ===",
        f"Makespan: {report['end_timestep']} timesteps",
        f"Robot-time: {total} timesteps across {report['robot_count']} robots",
        f"Explicit actions: {report['explicit_actions']} ({_percent(int(report['explicit_actions']), total):.2f}%)",
        f"Waits: {waits} ({_percent(waits, total):.2f}%)",
        "Action-time ratios:",
    ]
    for action_type in ActionType:
        count = int(action_counts[action_type.value])
        lines.append(
            f"  {action_type.value:9s} {count:8d}  {_percent(count, total):6.2f}%"
        )

    total_moves = int(action_counts[ActionType.MOVE.value])
    lines.extend(
        [
            "Movement purpose:",
            f"  collection  {int(movement['collection']):8d}  {_percent(int(movement['collection']), total_moves):6.2f}% of moves",
            f"  refill      {int(movement['refill']):8d}  {_percent(int(movement['refill']), total_moves):6.2f}% of moves",
            f"  fulfillment {int(movement['fulfillment']):8d}  {_percent(int(movement['fulfillment']), total_moves):6.2f}% of moves",
            f"Orders completed: {report['orders_completed']}",
            f"Refill trips: {report['refill_trips']}",
            f"Aisle service visits: {report['aisle_visits']}",
            f"Aisle re-entries: {report['aisle_reentries']}",
            "Per robot:",
        ]
    )
    for robot in report["robots"]:
        lines.append(
            "  r{robot_id}: moves={moves} collect={collection_moves} refill={refill_moves} "
            "fulfill={fulfillment_moves} waits={waits} internal_waits={internal_waits} "
            "terminal_idle={terminal_idle} orders={fulfills}".format(**robot)
        )
    return "\n".join(lines)


def main() -> None:
    parser = ArgumentParser(description="Analyze an atoms-not-electrons submission file.")
    parser.add_argument("submission", type=Path)
    parser.add_argument("--problem", type=Path, default=DEFAULT_PROBLEM_PATH)
    parser.add_argument(
        "--robots",
        type=int,
        default=None,
        help="Analyze robot ids 0..N-1. Defaults to ids present in the submission.",
    )
    parser.add_argument(
        "--end-timestep",
        type=int,
        default=None,
        help="Exact replay world timestep. Useful when a cutoff file ends with omitted waits.",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Optional path for the machine-readable metrics report.",
    )
    args = parser.parse_args()

    if args.robots is not None and args.robots <= 0:
        parser.error("--robots must be positive")
    if args.end_timestep is not None and args.end_timestep < 0:
        parser.error("--end-timestep must be nonnegative")

    problem = parse_problem(args.problem)
    actions = read_submission(args.submission)
    robot_ids = range(args.robots) if args.robots is not None else None
    report = analyze_actions(
        actions,
        problem,
        robot_ids=robot_ids,
        end_timestep=args.end_timestep,
    )
    print(format_metrics_report(report))

    if args.json is not None:
        write_metrics_json(report, args.json)
        print(f"Wrote metrics JSON: {args.json}")


if __name__ == "__main__":
    main()
