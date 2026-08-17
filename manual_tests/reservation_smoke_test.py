"""Generate a visual multi-robot reservation demo for the Tutor testbench.

The script uses the real BIG_ORDER warehouse and writes a move-only submission
that demonstrates three traffic cases in the open center rows:

1. Two robots on the same row swap positions. Waiting alone cannot solve the
   encounter, so the lower-priority robot must spatially detour around the
   higher-priority robot.
2. Two robots approach a perpendicular intersection at the same time. The
   lower-priority robot waits, then continues after the intersection clears.
3. Four robots share one row as two simultaneous head-on pairs. The two
   lower-priority robots must get out of the row so both pairs can pass.

Staging moves between demonstrations are intentionally sequential. The actual
traffic demonstrations are planned concurrently with Scheduler reservations.
The entire generated schedule is replayed through the local Simulator before
being written for the Tutor testbench.
"""

from pathlib import Path
import sys
from typing import Dict, Iterable, List, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models import Action, ActionType, Position
from src.parser import parse_problem
from src.pathfinding import PathPlanner
from src.scheduler import Scheduler, TimedPosition
from src.simulator import Simulator
from src.world import WorldState
from src.writer import write_submission


BIG_ORDER_PATH = REPO_ROOT / "source_material" / "BIG_ORDER.txt"
OUTPUT_PATH = REPO_ROOT / "outputs" / "reservation_smoke_test.txt"
CENTER_ROW = 19
PAUSE_TIMESTEPS = 3


def trajectory_wait_count(trajectory: Sequence[TimedPosition]) -> int:
    """Count repeated-position transitions in a timed trajectory."""
    return sum(
        previous_position == current_position
        for (_, previous_position), (_, current_position) in zip(
            trajectory,
            trajectory[1:],
        )
    )


def trajectory_to_actions(
    robot_id: int,
    trajectory: Sequence[TimedPosition],
) -> List[Action]:
    """Convert timed positions to MOVE actions, omitting waits."""
    actions: List[Action] = []

    for previous, current in zip(trajectory, trajectory[1:]):
        previous_timestep, previous_position = previous
        current_timestep, current_position = current

        if current_timestep != previous_timestep + 1:
            raise RuntimeError("Timed trajectory contains a timestep gap")

        if current_position == previous_position:
            continue

        actions.append(
            Action(
                timestep=previous_timestep,
                robot_id=robot_id,
                action=ActionType.MOVE,
                target=current_position,
            )
        )

    return actions


def pause(simulator: Simulator, timesteps: int = PAUSE_TIMESTEPS) -> None:
    """Advance several all-robot wait timesteps for visual separation."""
    for _ in range(timesteps):
        simulator.step([])


def stage_robot(
    world: WorldState,
    simulator: Simulator,
    actions: List[Action],
    robot_id: int,
    goal: Position,
) -> None:
    """Move one robot to a staging cell while every other robot stays still."""
    robot = world.robots[robot_id]
    planner = PathPlanner(world)
    blocked_robots = {
        other.position
        for other_id, other in world.robots.items()
        if other_id != robot_id
    }

    path = planner.find_path(
        robot.position,
        goal,
        blocked=blocked_robots,
    )
    if not path:
        raise RuntimeError(
            f"Could not stage robot {robot_id} from {robot.position} to {goal}"
        )

    for target in path[1:]:
        action = Action(
            timestep=world.timestep,
            robot_id=robot_id,
            action=ActionType.MOVE,
            target=target,
        )
        actions.append(action)
        simulator.step([action])


def stage_robots(
    world: WorldState,
    simulator: Simulator,
    actions: List[Action],
    positions: Dict[int, Position],
) -> None:
    """Sequentially place robots at deterministic scenario starting cells."""
    for robot_id in sorted(positions):
        stage_robot(
            world,
            simulator,
            actions,
            robot_id,
            positions[robot_id],
        )


def run_reserved_scenario(
    name: str,
    world: WorldState,
    simulator: Simulator,
    actions: List[Action],
    goals: Dict[int, Position],
    priority_order: Iterable[int],
) -> Tuple[int, int, Dict[int, List[TimedPosition]]]:
    """Plan, reserve, and locally execute one concurrent traffic scenario."""
    start_timestep = world.timestep
    active_ids = set(goals)
    idle_positions = {
        robot.position
        for robot_id, robot in world.robots.items()
        if robot_id not in active_ids
    }

    scheduler = Scheduler(world)
    trajectories: Dict[int, List[TimedPosition]] = {}

    for robot_id in priority_order:
        if robot_id not in goals:
            raise RuntimeError(
                f"Priority order for {name} contains inactive robot {robot_id}"
            )

        robot = world.robots[robot_id]
        trajectory = scheduler.plan_and_reserve(
            robot.position,
            goals[robot_id],
            start_timestep=start_timestep,
            blocked=idle_positions,
            max_timestep=start_timestep + 120,
        )
        if not trajectory:
            raise RuntimeError(
                f"No reservation-safe trajectory found for robot {robot_id} "
                f"during {name}"
            )
        trajectories[robot_id] = trajectory

    scenario_actions: List[Action] = []
    for robot_id, trajectory in trajectories.items():
        scenario_actions.extend(
            trajectory_to_actions(robot_id, trajectory)
        )

    scenario_actions.sort(key=lambda action: (action.timestep, action.robot_id))
    actions.extend(scenario_actions)

    end_timestep = max(
        trajectory[-1][0]
        for trajectory in trajectories.values()
    )

    if scenario_actions:
        simulator.run(scenario_actions)
    while world.timestep < end_timestep:
        simulator.step([])

    for robot_id, goal in goals.items():
        if world.robots[robot_id].position != goal:
            raise RuntimeError(
                f"Robot {robot_id} ended {name} at "
                f"{world.robots[robot_id].position}, expected {goal}"
            )

    print(f"{name}: timesteps {start_timestep} through {end_timestep - 1}")
    for robot_id in priority_order:
        trajectory = trajectories[robot_id]
        print(
            f"  robot {robot_id}: "
            f"{trajectory[0][1]} -> {trajectory[-1][1]}, "
            f"{len(trajectory) - 1} elapsed steps, "
            f"{trajectory_wait_count(trajectory)} waits"
        )

    return start_timestep, end_timestep, trajectories


def main() -> None:
    problem = parse_problem(BIG_ORDER_PATH)
    world = WorldState(problem)
    simulator = Simulator(world)
    actions: List[Action] = []

    # ------------------------------------------------------------------
    # Scenario 1: two robots swap positions on one completely open row.
    # Robot 0 plans first and takes the direct route. Robot 1 cannot solve
    # the swap by waiting forever, so its space-time A* must leave y=19.
    # ------------------------------------------------------------------
    stage_robots(
        world,
        simulator,
        actions,
        {
            0: (20, CENTER_ROW),
            1: (40, CENTER_ROW),
        },
    )
    pause(simulator)

    _, _, swap_paths = run_reserved_scenario(
        "Scenario 1 - two-robot same-row swap",
        world,
        simulator,
        actions,
        {
            0: (40, CENTER_ROW),
            1: (20, CENTER_ROW),
        },
        priority_order=[0, 1],
    )

    if not any(position[1] != CENTER_ROW for _, position in swap_paths[1]):
        raise RuntimeError(
            "Scenario 1 did not force robot 1 to spatially detour off the row"
        )

    pause(simulator)

    # ------------------------------------------------------------------
    # Scenario 2: perpendicular paths are chosen so both direct routes would
    # reach (30, 19) at the same state timestep. Robot 0 gets priority.
    # With the short temporary conflict, robot 1 should wait and continue.
    # ------------------------------------------------------------------
    stage_robots(
        world,
        simulator,
        actions,
        {
            0: (30, 17),
            1: (28, CENTER_ROW),
        },
    )
    pause(simulator)

    _, _, intersection_paths = run_reserved_scenario(
        "Scenario 2 - crossing intersection",
        world,
        simulator,
        actions,
        {
            0: (30, 21),
            1: (32, CENTER_ROW),
        },
        priority_order=[0, 1],
    )

    if trajectory_wait_count(intersection_paths[1]) == 0:
        raise RuntimeError(
            "Scenario 2 did not demonstrate a reservation-induced wait"
        )

    pause(simulator)

    # ------------------------------------------------------------------
    # Scenario 3: four robots share y=19 as two simultaneous head-on pairs.
    # Robots 0 and 2 are the direct-path leaders of their local encounters;
    # robots 1 and 3 must leave the row to let the pairs exchange sides.
    # ------------------------------------------------------------------
    stage_robots(
        world,
        simulator,
        actions,
        {
            0: (18, CENTER_ROW),
            1: (26, CENTER_ROW),
            2: (34, CENTER_ROW),
            3: (42, CENTER_ROW),
        },
    )
    pause(simulator)

    _, _, four_robot_paths = run_reserved_scenario(
        "Scenario 3 - four robots passing on one row",
        world,
        simulator,
        actions,
        {
            0: (26, CENTER_ROW),
            1: (18, CENTER_ROW),
            2: (42, CENTER_ROW),
            3: (34, CENTER_ROW),
        },
        priority_order=[0, 1, 2, 3],
    )

    for robot_id in (1, 3):
        if not any(
            position[1] != CENTER_ROW
            for _, position in four_robot_paths[robot_id]
        ):
            raise RuntimeError(
                f"Scenario 3 did not force robot {robot_id} off the row"
            )

    # Independently replay the complete file from the original challenge state.
    replay_problem = parse_problem(BIG_ORDER_PATH)
    replay_world = WorldState(replay_problem)
    replay_simulator = Simulator(replay_world)
    replay_simulator.run(actions)
    replay_world.validate()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_submission(actions, OUTPUT_PATH)

    print()
    print(f"Total MOVE actions written: {len(actions)}")
    print(f"Final replay timestep: {replay_world.timestep}")
    print(f"Wrote testbench submission to {OUTPUT_PATH}")
    print("Upload that file with BIG_ORDER.txt in the Tutor Testbench.")


if __name__ == "__main__":
    main()
