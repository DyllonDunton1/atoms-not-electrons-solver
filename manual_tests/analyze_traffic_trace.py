"""Stream and summarize large per-timestep traffic traces.

This script is intentionally independent from solver behavior. It reads the
JSONL trace produced by ``manual_tests/aisle_solver_smoke_test.py`` one line at
a time, so even very large traces can be analyzed with bounded memory.

Outputs are small files intended for human/AI inspection:

* ``*_stalls.csv``: consecutive per-robot nonproductive intervals.
* ``*_clusters.csv``: intervals where two or more active robots are stalled.
* ``*_events.jsonl``: sparse logical state/traffic changes only.
* ``*_report.txt``: compact ranked summary of the longest stalls/clusters.
* ``*_context.jsonl``: raw trace records near the start/end of top stalls.

Optionally, ``--write-compact-csv`` also emits one compact row per robot per
traced timestep. This is much smaller than the raw JSONL but is not needed for
normal diagnosis.
"""

from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple


PRODUCTIVE_RESULTS = {"move"}


def _xy(value: Any) -> Tuple[Optional[int], Optional[int]]:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return value[0], value[1]
    return None, None


def _csv_set(values: Iterable[Any]) -> str:
    return ";".join(str(value) for value in sorted(set(values)))


def _json_key(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _is_productive(robot: Dict[str, Any]) -> bool:
    result = robot.get("traffic", {}).get("result")
    return result in PRODUCTIVE_RESULTS or (
        isinstance(result, str) and result.startswith("fixed_")
    )


def _has_outstanding_requirements(robot: Dict[str, Any]) -> bool:
    remaining_by_sku = robot.get("remaining_by_sku") or {}
    return any(quantity > 0 for quantity in remaining_by_sku.values())


def _target_pallet_id(robot: Dict[str, Any]) -> Optional[int]:
    target = robot.get("target_pallet")
    if not isinstance(target, dict):
        return None
    return target.get("pallet_id")


def _event_signature(robot: Dict[str, Any]) -> Tuple[Any, ...]:
    """Logical state signature that deliberately ignores ordinary movement.

    Position is included in emitted event payloads, but position alone does not
    create an event. This keeps the event file sparse while still recording the
    exact location whenever something meaningful changes.
    """
    intent = robot.get("intent") or {}
    traffic = robot.get("traffic") or {}
    current_stop = robot.get("current_stop") or {}
    return (
        robot.get("order_id"),
        robot.get("phase"),
        robot.get("active_aisle_id"),
        robot.get("aisle_stop_index"),
        current_stop.get("pallet_id"),
        _target_pallet_id(robot),
        _json_key(robot.get("pickup")),
        intent.get("action"),
        _json_key(intent.get("target")),
        _json_key(intent.get("move_goal")),
        traffic.get("result"),
        _json_key(traffic.get("blocking_robot_ids") or []),
        _json_key(robot.get("deferred_pallet_ids") or []),
        _json_key(robot.get("persistent_blocked_pallet_ids") or []),
        _has_outstanding_requirements(robot),
    )


def _event_record(timestep: int, robot_id: int, robot: Dict[str, Any]) -> Dict[str, Any]:
    intent = robot.get("intent") or {}
    traffic = robot.get("traffic") or {}
    return {
        "timestep": timestep,
        "robot_id": robot_id,
        "position": robot.get("position"),
        "order_id": robot.get("order_id"),
        "phase": robot.get("phase"),
        "aisle_id": robot.get("active_aisle_id"),
        "stop_index": robot.get("aisle_stop_index"),
        "target_pallet_id": _target_pallet_id(robot),
        "pickup": robot.get("pickup"),
        "remaining": robot.get("remaining"),
        "outstanding_sku_count": sum(
            1
            for quantity in (robot.get("remaining_by_sku") or {}).values()
            if quantity > 0
        ),
        "deferred_pallet_ids": robot.get("deferred_pallet_ids") or [],
        "persistent_blocked_pallet_ids": robot.get("persistent_blocked_pallet_ids") or [],
        "intent": intent,
        "traffic": {
            "result": traffic.get("result"),
            "goal": traffic.get("goal"),
            "path_steps": traffic.get("path_steps"),
            "preferred_next": traffic.get("preferred_next"),
            "physical_free": traffic.get("physical_free"),
            "reservation_free": traffic.get("reservation_free"),
            "blocking_robot_ids": traffic.get("blocking_robot_ids") or [],
            "committed_next": traffic.get("committed_next"),
        },
        "no_intent_with_outstanding": (
            robot.get("order_id") is not None
            and traffic.get("result") == "no_intent"
            and _has_outstanding_requirements(robot)
        ),
    }


def _new_stall(timestep: int, robot_id: int, robot: Dict[str, Any]) -> Dict[str, Any]:
    traffic = robot.get("traffic") or {}
    intent = robot.get("intent") or {}
    return {
        "robot_id": robot_id,
        "start": timestep,
        "end": timestep,
        "length": 1,
        "start_position": robot.get("position"),
        "end_position": robot.get("position"),
        "start_order_id": robot.get("order_id"),
        "end_order_id": robot.get("order_id"),
        "start_phase": robot.get("phase"),
        "end_phase": robot.get("phase"),
        "start_aisle_id": robot.get("active_aisle_id"),
        "end_aisle_id": robot.get("active_aisle_id"),
        "start_target_pallet_id": _target_pallet_id(robot),
        "end_target_pallet_id": _target_pallet_id(robot),
        "start_move_goal": intent.get("move_goal"),
        "end_move_goal": intent.get("move_goal"),
        "results": Counter([traffic.get("result")]),
        "blocking_robot_ids": set(traffic.get("blocking_robot_ids") or []),
        "deferred_pallet_ids": set(robot.get("deferred_pallet_ids") or []),
        "persistent_blocked_pallet_ids": set(
            robot.get("persistent_blocked_pallet_ids") or []
        ),
        "target_pallet_ids": set(
            [] if _target_pallet_id(robot) is None else [_target_pallet_id(robot)]
        ),
        "move_goals": set(
            [] if intent.get("move_goal") is None else [_json_key(intent.get("move_goal"))]
        ),
        "had_no_intent_with_outstanding": (
            traffic.get("result") == "no_intent"
            and _has_outstanding_requirements(robot)
        ),
    }


def _extend_stall(stall: Dict[str, Any], timestep: int, robot: Dict[str, Any]) -> None:
    traffic = robot.get("traffic") or {}
    intent = robot.get("intent") or {}
    stall["end"] = timestep
    stall["length"] += 1
    stall["end_position"] = robot.get("position")
    stall["end_order_id"] = robot.get("order_id")
    stall["end_phase"] = robot.get("phase")
    stall["end_aisle_id"] = robot.get("active_aisle_id")
    stall["end_target_pallet_id"] = _target_pallet_id(robot)
    stall["end_move_goal"] = intent.get("move_goal")
    stall["results"][traffic.get("result")] += 1
    stall["blocking_robot_ids"].update(traffic.get("blocking_robot_ids") or [])
    stall["deferred_pallet_ids"].update(robot.get("deferred_pallet_ids") or [])
    stall["persistent_blocked_pallet_ids"].update(
        robot.get("persistent_blocked_pallet_ids") or []
    )
    target_pallet_id = _target_pallet_id(robot)
    if target_pallet_id is not None:
        stall["target_pallet_ids"].add(target_pallet_id)
    if intent.get("move_goal") is not None:
        stall["move_goals"].add(_json_key(intent.get("move_goal")))
    stall["had_no_intent_with_outstanding"] = (
        stall["had_no_intent_with_outstanding"]
        or (
            traffic.get("result") == "no_intent"
            and _has_outstanding_requirements(robot)
        )
    )


def _stall_row(stall: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "robot_id": stall["robot_id"],
        "start_timestep": stall["start"],
        "end_timestep": stall["end"],
        "length": stall["length"],
        "start_position": _json_key(stall["start_position"]),
        "end_position": _json_key(stall["end_position"]),
        "start_order_id": stall["start_order_id"],
        "end_order_id": stall["end_order_id"],
        "start_phase": stall["start_phase"],
        "end_phase": stall["end_phase"],
        "start_aisle_id": stall["start_aisle_id"],
        "end_aisle_id": stall["end_aisle_id"],
        "start_target_pallet_id": stall["start_target_pallet_id"],
        "end_target_pallet_id": stall["end_target_pallet_id"],
        "start_move_goal": _json_key(stall["start_move_goal"]),
        "end_move_goal": _json_key(stall["end_move_goal"]),
        "result_counts": _json_key(dict(sorted(stall["results"].items()))),
        "blocking_robot_ids": _csv_set(stall["blocking_robot_ids"]),
        "target_pallet_ids_seen": _csv_set(stall["target_pallet_ids"]),
        "move_goals_seen": ";".join(sorted(stall["move_goals"])),
        "deferred_pallet_ids_seen": _csv_set(stall["deferred_pallet_ids"]),
        "persistent_blocked_pallet_ids_seen": _csv_set(
            stall["persistent_blocked_pallet_ids"]
        ),
        "no_intent_with_outstanding": stall["had_no_intent_with_outstanding"],
    }


def _compact_row(timestep: int, robot_id: int, robot: Dict[str, Any]) -> Dict[str, Any]:
    position_x, position_y = _xy(robot.get("position"))
    pickup_x, pickup_y = _xy(robot.get("pickup"))
    intent = robot.get("intent") or {}
    traffic = robot.get("traffic") or {}
    move_goal_x, move_goal_y = _xy(intent.get("move_goal"))
    preferred_x, preferred_y = _xy(traffic.get("preferred_next"))
    committed_x, committed_y = _xy(traffic.get("committed_next"))
    target_x, target_y = _xy(intent.get("target"))
    return {
        "timestep": timestep,
        "robot_id": robot_id,
        "x": position_x,
        "y": position_y,
        "order_id": robot.get("order_id"),
        "phase": robot.get("phase"),
        "aisle_id": robot.get("active_aisle_id"),
        "stop_index": robot.get("aisle_stop_index"),
        "target_pallet_id": _target_pallet_id(robot),
        "pickup_x": pickup_x,
        "pickup_y": pickup_y,
        "remaining": robot.get("remaining"),
        "outstanding_sku_count": sum(
            1
            for quantity in (robot.get("remaining_by_sku") or {}).values()
            if quantity > 0
        ),
        "intent_action": intent.get("action"),
        "intent_target_x": target_x,
        "intent_target_y": target_y,
        "move_goal_x": move_goal_x,
        "move_goal_y": move_goal_y,
        "path_steps": traffic.get("path_steps"),
        "preferred_next_x": preferred_x,
        "preferred_next_y": preferred_y,
        "physical_free": traffic.get("physical_free"),
        "reservation_free": traffic.get("reservation_free"),
        "blocking_robot_ids": _csv_set(traffic.get("blocking_robot_ids") or []),
        "committed_next_x": committed_x,
        "committed_next_y": committed_y,
        "traffic_result": traffic.get("result"),
        "deferred_pallet_ids": _csv_set(robot.get("deferred_pallet_ids") or []),
        "persistent_blocked_pallet_ids": _csv_set(
            robot.get("persistent_blocked_pallet_ids") or []
        ),
    }


def _finish_cluster(cluster: Optional[Dict[str, Any]], clusters: List[Dict[str, Any]]) -> None:
    if cluster is not None:
        clusters.append(cluster)


def analyze_trace(
    trace_path: Path,
    *,
    min_stall: int,
    min_cluster_robots: int,
    write_compact_csv: bool,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Path, Optional[Path], int, int]:
    prefix = trace_path.with_suffix("")
    events_path = prefix.with_name(prefix.name + "_events.jsonl")
    compact_path = (
        prefix.with_name(prefix.name + "_compact.csv")
        if write_compact_csv
        else None
    )

    active_stalls: Dict[int, Dict[str, Any]] = {}
    stalls: List[Dict[str, Any]] = []
    clusters: List[Dict[str, Any]] = []
    current_cluster: Optional[Dict[str, Any]] = None
    previous_signatures: Dict[int, Tuple[Any, ...]] = {}
    first_timestep: Optional[int] = None
    last_timestep: Optional[int] = None

    compact_file = None
    compact_writer = None
    if compact_path is not None:
        compact_file = compact_path.open("w", encoding="utf-8", newline="")

    try:
        with trace_path.open("r", encoding="utf-8") as trace_file, events_path.open(
            "w", encoding="utf-8"
        ) as events_file:
            for line_number, line in enumerate(trace_file, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exception:
                    raise ValueError(
                        "Invalid JSON on trace line {}: {}".format(
                            line_number, exception
                        )
                    ) from exception

                timestep = int(record["timestep"])
                if first_timestep is None:
                    first_timestep = timestep
                last_timestep = timestep
                stalled_now: Set[int] = set()

                robots = record.get("robots") or {}
                for robot_key in sorted(robots, key=lambda value: int(value)):
                    robot_id = int(robot_key)
                    robot = robots[robot_key]

                    if compact_file is not None:
                        compact_row = _compact_row(timestep, robot_id, robot)
                        if compact_writer is None:
                            compact_writer = csv.DictWriter(
                                compact_file,
                                fieldnames=list(compact_row),
                            )
                            compact_writer.writeheader()
                        compact_writer.writerow(compact_row)

                    signature = _event_signature(robot)
                    if previous_signatures.get(robot_id) != signature:
                        events_file.write(
                            json.dumps(
                                _event_record(timestep, robot_id, robot),
                                separators=(",", ":"),
                            )
                            + "\n"
                        )
                        previous_signatures[robot_id] = signature

                    active_order = robot.get("order_id") is not None
                    stalled = active_order and not _is_productive(robot)
                    if stalled:
                        stalled_now.add(robot_id)
                        if robot_id not in active_stalls:
                            active_stalls[robot_id] = _new_stall(
                                timestep, robot_id, robot
                            )
                        else:
                            _extend_stall(active_stalls[robot_id], timestep, robot)
                    else:
                        prior = active_stalls.pop(robot_id, None)
                        if prior is not None and prior["length"] >= min_stall:
                            stalls.append(prior)

                # A cluster is any continuous interval with at least N active
                # robots simultaneously stalled. The exact stalled set may vary;
                # the union and maximum simultaneous count are retained.
                if len(stalled_now) >= min_cluster_robots:
                    if current_cluster is None:
                        current_cluster = {
                            "start": timestep,
                            "end": timestep,
                            "length": 1,
                            "robots": set(stalled_now),
                            "max_stalled": len(stalled_now),
                        }
                    else:
                        current_cluster["end"] = timestep
                        current_cluster["length"] += 1
                        current_cluster["robots"].update(stalled_now)
                        current_cluster["max_stalled"] = max(
                            current_cluster["max_stalled"], len(stalled_now)
                        )
                else:
                    _finish_cluster(current_cluster, clusters)
                    current_cluster = None
    finally:
        if compact_file is not None:
            compact_file.close()

    for prior in active_stalls.values():
        if prior["length"] >= min_stall:
            stalls.append(prior)
    _finish_cluster(current_cluster, clusters)

    stalls.sort(key=lambda item: (-item["length"], item["start"], item["robot_id"]))
    clusters.sort(key=lambda item: (-item["length"], item["start"]))

    if first_timestep is None or last_timestep is None:
        raise ValueError("Trace file contained no JSON records")

    return (
        stalls,
        clusters,
        events_path,
        compact_path,
        first_timestep,
        last_timestep,
    )


def write_stalls(path: Path, stalls: Sequence[Dict[str, Any]]) -> None:
    rows = [_stall_row(stall) for stall in stalls]
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_clusters(path: Path, clusters: Sequence[Dict[str, Any]]) -> None:
    rows = [
        {
            "start_timestep": cluster["start"],
            "end_timestep": cluster["end"],
            "length": cluster["length"],
            "robots_seen": _csv_set(cluster["robots"]),
            "max_simultaneously_stalled": cluster["max_stalled"],
        }
        for cluster in clusters
    ]
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _context_timesteps(
    stalls: Sequence[Dict[str, Any]], top: int, context: int
) -> Set[int]:
    selected: Set[int] = set()
    for stall in stalls[:top]:
        for center in (stall["start"], stall["end"]):
            for timestep in range(center - context, center + context + 1):
                if timestep >= 0:
                    selected.add(timestep)
    return selected


def write_context(
    trace_path: Path,
    context_path: Path,
    stalls: Sequence[Dict[str, Any]],
    *,
    top: int,
    context: int,
) -> None:
    selected = _context_timesteps(stalls, top, context)
    with trace_path.open("r", encoding="utf-8") as source, context_path.open(
        "w", encoding="utf-8"
    ) as output:
        for line in source:
            if not line.strip():
                continue
            record = json.loads(line)
            if int(record["timestep"]) in selected:
                output.write(line if line.endswith("\n") else line + "\n")


def write_report(
    path: Path,
    trace_path: Path,
    stalls: Sequence[Dict[str, Any]],
    clusters: Sequence[Dict[str, Any]],
    *,
    top: int,
    min_stall: int,
    first_timestep: int,
    last_timestep: int,
) -> None:
    lines: List[str] = []
    lines.append("Traffic trace analysis")
    lines.append("======================")
    lines.append("trace: {}".format(trace_path))
    lines.append("timesteps: {}..{}".format(first_timestep, last_timestep))
    lines.append("stall threshold: {} consecutive timesteps".format(min_stall))
    lines.append("qualifying robot stalls: {}".format(len(stalls)))
    lines.append("multi-robot stall clusters: {}".format(len(clusters)))
    lines.append("")

    lines.append("Longest robot stalls")
    lines.append("--------------------")
    if not stalls:
        lines.append("none")
    for index, stall in enumerate(stalls[:top], start=1):
        row = _stall_row(stall)
        lines.append(
            "{}. r{} t={}..{} len={} pos {} -> {} order {} -> {} phase {} -> {} aisle {} -> {}".format(
                index,
                row["robot_id"],
                row["start_timestep"],
                row["end_timestep"],
                row["length"],
                row["start_position"],
                row["end_position"],
                row["start_order_id"],
                row["end_order_id"],
                row["start_phase"],
                row["end_phase"],
                row["start_aisle_id"],
                row["end_aisle_id"],
            )
        )
        lines.append(
            "   results={} blockers={} target_pallets={} move_goals={}".format(
                row["result_counts"],
                row["blocking_robot_ids"] or "-",
                row["target_pallet_ids_seen"] or "-",
                row["move_goals_seen"] or "-",
            )
        )
        lines.append(
            "   deferred={} persistent_blocked={} no_intent_with_outstanding={}".format(
                row["deferred_pallet_ids_seen"] or "-",
                row["persistent_blocked_pallet_ids_seen"] or "-",
                row["no_intent_with_outstanding"],
            )
        )

    lines.append("")
    lines.append("Longest multi-robot stall clusters")
    lines.append("----------------------------------")
    if not clusters:
        lines.append("none")
    for index, cluster in enumerate(clusters[:top], start=1):
        lines.append(
            "{}. t={}..{} len={} robots={} max_simultaneous={}".format(
                index,
                cluster["start"],
                cluster["end"],
                cluster["length"],
                _csv_set(cluster["robots"]),
                cluster["max_stalled"],
            )
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = ArgumentParser(
        description="Stream a large solver traffic trace into compact diagnostic files."
    )
    parser.add_argument("trace", type=Path, help="Path to *_trace.jsonl")
    parser.add_argument(
        "--min-stall",
        type=int,
        default=10,
        help="Only report per-robot stalls lasting at least this many timesteps.",
    )
    parser.add_argument(
        "--min-cluster-robots",
        type=int,
        default=2,
        help="Minimum simultaneously stalled active robots for a cluster.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=25,
        help="Number of longest stalls/clusters to rank and extract context for.",
    )
    parser.add_argument(
        "--context",
        type=int,
        default=4,
        help="Raw timesteps to retain on each side of top stall starts/ends.",
    )
    parser.add_argument(
        "--write-compact-csv",
        action="store_true",
        help="Also emit one compact CSV row per robot per traced timestep.",
    )
    args = parser.parse_args()

    if args.min_stall <= 0:
        parser.error("--min-stall must be positive")
    if args.min_cluster_robots <= 0:
        parser.error("--min-cluster-robots must be positive")
    if args.top <= 0:
        parser.error("--top must be positive")
    if args.context < 0:
        parser.error("--context must be nonnegative")
    if not args.trace.is_file():
        parser.error("trace file does not exist: {}".format(args.trace))

    prefix = args.trace.with_suffix("")
    stalls_path = prefix.with_name(prefix.name + "_stalls.csv")
    clusters_path = prefix.with_name(prefix.name + "_clusters.csv")
    report_path = prefix.with_name(prefix.name + "_report.txt")
    context_path = prefix.with_name(prefix.name + "_context.jsonl")

    (
        stalls,
        clusters,
        events_path,
        compact_path,
        first_timestep,
        last_timestep,
    ) = analyze_trace(
        args.trace,
        min_stall=args.min_stall,
        min_cluster_robots=args.min_cluster_robots,
        write_compact_csv=args.write_compact_csv,
    )

    write_stalls(stalls_path, stalls)
    write_clusters(clusters_path, clusters)
    write_report(
        report_path,
        args.trace,
        stalls,
        clusters,
        top=args.top,
        min_stall=args.min_stall,
        first_timestep=first_timestep,
        last_timestep=last_timestep,
    )
    write_context(
        args.trace,
        context_path,
        stalls,
        top=args.top,
        context=args.context,
    )

    print("Processed {} through {}.".format(first_timestep, last_timestep))
    print("Found {} robot stalls >= {} timesteps.".format(len(stalls), args.min_stall))
    print("Found {} multi-robot stall clusters.".format(len(clusters)))
    print("Wrote {}".format(report_path))
    print("Wrote {}".format(stalls_path))
    print("Wrote {}".format(clusters_path))
    print("Wrote {}".format(events_path))
    print("Wrote {}".format(context_path))
    if compact_path is not None:
        print("Wrote {}".format(compact_path))


if __name__ == "__main__":
    main()
