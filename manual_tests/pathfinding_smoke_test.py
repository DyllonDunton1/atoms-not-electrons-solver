"""Generate a real BIG_ORDER pathfinding submission for the Tutor testbench.

This is intentionally temporary/manual glue code rather than solver logic.
It plans a route for robot 0 from its real starting position to y=0 at the
same x coordinate, then from there to y=39. Pallets and the other four robots
are treated as static obstacles. The resulting move actions are written using
writer.py so the output can be uploaded directly to the Tutor testbench.
"""

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models import Action, ActionType
from src.parser import parse_problem
from src.pathfinding import PathPlanner
from src.world import WorldState
from src.writer import write_submission


BIG_ORDER_PATH = REPO_ROOT / "source_material" / "BIG_ORDER.txt"
OUTPUT_PATH = REPO_ROOT / "outputs" / "pathfinding_smoke_test.txt"
ROBOT_ID = 0


def main() -> None:
    problem = parse_problem(BIG_ORDER_PATH)
    world = WorldState(problem)
    planner = PathPlanner(world)

    robot = world.robots[ROBOT_ID]
    start = robot.position
    top_goal = (start[0], world.fulfillment_y)
    bottom_goal = (start[0], world.replenishment_y)

    # PathPlanner intentionally does not treat robots as permanent obstacles.
    # For this manual test, however, the four idle robots never move, so their
    # current cells must be added to the static blocked set.
    other_robot_positions = {
        other.position
        for other_id, other in world.robots.items()
        if other_id != ROBOT_ID
    }

    path_to_top = planner.find_path(
        start,
        top_goal,
        blocked=other_robot_positions,
    )
    if not path_to_top:
        raise RuntimeError(f"No path found from {start} to {top_goal}")

    path_to_bottom = planner.find_path(
        top_goal,
        bottom_goal,
        blocked=other_robot_positions,
    )
    if not path_to_bottom:
        raise RuntimeError(f"No path found from {top_goal} to {bottom_goal}")

    # Each path includes its starting position. Drop the second path's first
    # position so the y=0 waypoint is not duplicated.
    full_path = path_to_top + path_to_bottom[1:]

    # The first action moves from full_path[0] to full_path[1] at timestep 0.
    # This is temporary action-generation code until the scheduler exists.
    actions = [
        Action(
            timestep=timestep,
            robot_id=ROBOT_ID,
            action=ActionType.MOVE,
            target=position,
        )
        for timestep, position in enumerate(full_path[1:])
    ]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_submission(actions, OUTPUT_PATH)

    print(f"Robot {ROBOT_ID} starts at {start}")
    print(
        f"Top route: {start} -> {top_goal} "
        f"({len(path_to_top) - 1} moves)"
    )
    print(
        f"Bottom route: {top_goal} -> {bottom_goal} "
        f"({len(path_to_bottom) - 1} moves)"
    )
    print(f"Total moves written: {len(actions)}")
    print(f"Wrote testbench submission to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
