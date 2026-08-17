"""Generate a real BIG_ORDER replenishment round trip for Tutor Testbench.

This is temporary/manual glue code for validating Step 10 against the official
browser simulator. Robot 0 walks beside the real pallet at (25, 23), picks five
items to visibly lower its stock, docks the pallet on its left side, carries it
to a reachable robot-center cell on y=39, lets automatic replenishment restore
it to full capacity, returns it to its original position, and undocks it.

The actions are executed through the local Simulator as they are generated, so
the output file is only written if the complete local sequence succeeds.
"""

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models import Action, ActionType, Pallet
from src.parser import parse_problem
from src.pathfinding import PathPlanner
from src.simulator import Simulator
from src.world import WorldState
from src.writer import write_submission


BIG_ORDER_PATH = REPO_ROOT / "source_material" / "BIG_ORDER.txt"
OUTPUT_PATH = REPO_ROOT / "outputs" / "replenishment_smoke_test.txt"
ROBOT_ID = 0
PALLET_POSITION = (25, 23)
DOCK_ROBOT_POSITION = (26, 23)
PICKS_BEFORE_REFILL = 5


def execute_move_path(path, world, simulator, actions):
    """Append and locally execute MOVE actions for a planned center path."""
    for position in path[1:]:
        action = Action(
            timestep=world.timestep,
            robot_id=ROBOT_ID,
            action=ActionType.MOVE,
            target=position,
        )
        actions.append(action)
        simulator.step([action])


def shortest_replenishment_path(planner, world, blocked_robots):
    """Find the shortest reachable docked-footprint path to robot y=39."""
    robot = world.robots[ROBOT_ID]
    footprint = planner.footprint_for_robot(ROBOT_ID)
    ignored_pallet_ids = robot.docked_pallets

    best_path = None
    for x in range(world.width):
        goal = (x, world.replenishment_y)
        path = planner.find_path(
            robot.position,
            goal,
            footprint=footprint,
            blocked=blocked_robots,
            ignored_pallet_ids=ignored_pallet_ids,
        )
        if path and (best_path is None or len(path) < len(best_path)):
            best_path = path

    if best_path is None:
        raise RuntimeError("No reachable replenishment-row goal for docked footprint")

    return best_path


def main() -> None:
    problem = parse_problem(BIG_ORDER_PATH)
    world = WorldState(problem)
    planner = PathPlanner(world)
    simulator = Simulator(world)
    actions = []

    robot = world.robots[ROBOT_ID]
    pallet = world.entity_at(PALLET_POSITION)
    if not isinstance(pallet, Pallet):
        raise RuntimeError(f"Expected a pallet at {PALLET_POSITION}, found {pallet}")

    pallet_id = pallet.pallet_id
    original_pallet_position = pallet.original_position
    starting_count = pallet.count
    max_count = pallet.max_count

    if starting_count < PICKS_BEFORE_REFILL:
        raise RuntimeError(
            f"Pallet {pallet_id} only has {starting_count} items; "
            f"need at least {PICKS_BEFORE_REFILL} for this smoke test"
        )

    blocked_robots = {
        other.position
        for other_id, other in world.robots.items()
        if other_id != ROBOT_ID
    }

    # Move beside the selected real pallet so it will dock on robot 0's left.
    path_to_pallet = planner.find_path(
        robot.position,
        DOCK_ROBOT_POSITION,
        blocked=blocked_robots,
    )
    if not path_to_pallet:
        raise RuntimeError(
            f"No path from {robot.position} to {DOCK_ROBOT_POSITION}"
        )
    execute_move_path(path_to_pallet, world, simulator, actions)

    # Drain a few items so the browser visibly shows a lower count before the
    # replenishment trip. Each PICK is a separate robot action/timestep.
    for _ in range(PICKS_BEFORE_REFILL):
        action = Action(
            timestep=world.timestep,
            robot_id=ROBOT_ID,
            action=ActionType.PICK,
            target=PALLET_POSITION,
        )
        actions.append(action)
        simulator.step([action])

    depleted_count = world.pallets[pallet_id].count
    expected_depleted_count = starting_count - PICKS_BEFORE_REFILL
    if depleted_count != expected_depleted_count:
        raise RuntimeError(
            f"Expected depleted count {expected_depleted_count}, got {depleted_count}"
        )

    dock_action = Action(
        timestep=world.timestep,
        robot_id=ROBOT_ID,
        action=ActionType.DOCK,
        target=PALLET_POSITION,
    )
    actions.append(dock_action)
    simulator.step([dock_action])

    if world.pallets[pallet_id].docked_offset != (-1, 0):
        raise RuntimeError(
            f"Expected pallet to dock on left at offset (-1, 0), got "
            f"{world.pallets[pallet_id].docked_offset}"
        )

    # Carry the docked footprint to any shortest reachable robot cell on y=39.
    path_to_replenishment = shortest_replenishment_path(
        planner,
        world,
        blocked_robots,
    )
    execute_move_path(path_to_replenishment, world, simulator, actions)

    if world.robots[ROBOT_ID].position[1] != world.replenishment_y:
        raise RuntimeError("Robot did not reach the replenishment row")
    if world.pallets[pallet_id].count != max_count:
        raise RuntimeError(
            f"Pallet did not refill: count={world.pallets[pallet_id].count}, "
            f"max={max_count}"
        )

    replenishment_robot_position = world.robots[ROBOT_ID].position

    # Return with the same rigid footprint. Because the pallet stayed on the
    # robot's left, returning the robot center to (26, 23) returns the pallet
    # exactly to its original cell at (25, 23).
    footprint = planner.footprint_for_robot(ROBOT_ID)
    path_home = planner.find_path(
        world.robots[ROBOT_ID].position,
        DOCK_ROBOT_POSITION,
        footprint=footprint,
        blocked=blocked_robots,
        ignored_pallet_ids=world.robots[ROBOT_ID].docked_pallets,
    )
    if not path_home:
        raise RuntimeError("No docked-footprint path back to the pallet's home")
    execute_move_path(path_home, world, simulator, actions)

    if world.pallets[pallet_id].position != original_pallet_position:
        raise RuntimeError(
            f"Pallet returned to {world.pallets[pallet_id].position}, expected "
            f"{original_pallet_position}"
        )

    undock_action = Action(
        timestep=world.timestep,
        robot_id=ROBOT_ID,
        action=ActionType.UNDOCK,
        target=original_pallet_position,
    )
    actions.append(undock_action)
    simulator.step([undock_action])

    final_pallet = world.pallets[pallet_id]
    if final_pallet.docked_to is not None:
        raise RuntimeError("Pallet remained docked after returning home")
    if final_pallet.position != original_pallet_position:
        raise RuntimeError("Pallet moved during undocking")
    if final_pallet.count != max_count:
        raise RuntimeError("Pallet lost its replenished stock on the return trip")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_submission(actions, OUTPUT_PATH)

    print(f"Pallet {pallet_id} at {original_pallet_position}")
    print(f"Starting stock: {starting_count}/{max_count}")
    print(
        f"After {PICKS_BEFORE_REFILL} picks: "
        f"{depleted_count}/{max_count}"
    )
    print(
        f"Robot reached replenishment row at {replenishment_robot_position}; "
        f"pallet refilled to {max_count}/{max_count}."
    )
    print(
        f"Returned pallet to {final_pallet.position} and undocked with "
        f"{final_pallet.count}/{final_pallet.max_count} stock."
    )
    print(f"Total actions: {len(actions)}")
    print("Local simulator validation passed.")
    print(f"Wrote testbench submission to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
