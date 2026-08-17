"""Generate a complex real-order submission for manual Tutor validation.

This is intentionally temporary/manual planning glue rather than production
solver logic. Robot 0 fulfills one large order from BIG_ORDER.txt. For each
SKU in that order, this script greedily chooses the closest reachable pallet
with that SKU, finds the closest reachable adjacent pickup cell, moves there,
and picks one item. The other four robots are treated as permanent obstacles.

After all items are collected, robot 0 moves to y=0 and fulfills the order.
The generated actions are replayed through the local Simulator before writer.py
produces the uploadable text file.
"""

from collections import Counter
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models import Action, ActionType
from src.parser import parse_problem
from src.pathfinding import PathPlanner
from src.simulator import Simulator
from src.world import WorldState
from src.writer import write_submission


BIG_ORDER_PATH = REPO_ROOT / "source_material" / "BIG_ORDER.txt"
OUTPUT_PATH = REPO_ROOT / "outputs" / "real_order_3_smoke_test.txt"
ROBOT_ID = 0
ORDER_ID = 3


def choose_pickup_route(planner, world, start, sku, blocked, remaining_stock):
    """Choose the nearest reachable pallet and adjacent pickup cell for a SKU.

    This helper exists only for this manual test. Later solver/task code will
    own pallet selection and pickup-position planning.
    """
    best_path = None
    best_pallet = None

    pallets = sorted(world.pallets_for_sku(sku), key=lambda pallet: pallet.pallet_id)
    for pallet in pallets:
        if remaining_stock[pallet.pallet_id] <= 0:
            continue

        for pickup_position in world.adjacent_positions(pallet.position):
            # PathPlanner already treats every pallet cell as blocked. The
            # explicit blocked set contains the four idle robots.
            if pickup_position in blocked:
                continue

            path = planner.find_path(
                start,
                pickup_position,
                blocked=blocked,
            )
            if not path:
                continue

            if best_path is None or len(path) < len(best_path):
                best_path = path
                best_pallet = pallet

    if best_path is None or best_pallet is None:
        raise RuntimeError(
            f"No reachable pallet with stock found for SKU {sku} from {start}"
        )

    return best_pallet, best_path


def main() -> None:
    problem = parse_problem(BIG_ORDER_PATH)
    world = WorldState(problem)
    planner = PathPlanner(world)

    robot = world.robots[ROBOT_ID]
    order = world.orders[ORDER_ID]

    # The other robots do not receive any actions in this smoke test, so make
    # their starting cells permanent obstacles during all path planning.
    blocked_robots = {
        other.position
        for other_id, other in world.robots.items()
        if other_id != ROBOT_ID
    }

    # Keep a planning-side stock count because the real WorldState is not
    # mutated until the complete schedule is replayed through Simulator.
    remaining_stock = {
        pallet_id: pallet.count
        for pallet_id, pallet in world.pallets.items()
    }

    actions = []
    timestep = 0
    current_position = robot.position
    move_count = 0

    # Deliberately process the order in its original SKU sequence. This is not
    # intended to be globally efficient; it is a large mechanics smoke test.
    for item_number, sku in enumerate(order.skus, start=1):
        pallet, path = choose_pickup_route(
            planner,
            world,
            current_position,
            sku,
            blocked_robots,
            remaining_stock,
        )

        for position in path[1:]:
            actions.append(
                Action(
                    timestep=timestep,
                    robot_id=ROBOT_ID,
                    action=ActionType.MOVE,
                    target=position,
                )
            )
            timestep += 1
            move_count += 1

        current_position = path[-1]

        actions.append(
            Action(
                timestep=timestep,
                robot_id=ROBOT_ID,
                action=ActionType.PICK,
                target=pallet.position,
            )
        )
        timestep += 1
        remaining_stock[pallet.pallet_id] -= 1

        print(
            f"Item {item_number:>2}/{len(order.skus)}: SKU {sku:>2} "
            f"from pallet {pallet.pallet_id:>3} at {pallet.position}; "
            f"robot at {current_position}"
        )

    # Finish at the fulfillment row. Using the robot's current x coordinate
    # keeps this final target simple while still letting A* route around pallets.
    fulfillment_position = (current_position[0], world.fulfillment_y)
    path_to_fulfillment = planner.find_path(
        current_position,
        fulfillment_position,
        blocked=blocked_robots,
    )
    if not path_to_fulfillment:
        raise RuntimeError(
            f"No route from {current_position} to {fulfillment_position}"
        )

    for position in path_to_fulfillment[1:]:
        actions.append(
            Action(
                timestep=timestep,
                robot_id=ROBOT_ID,
                action=ActionType.MOVE,
                target=position,
            )
        )
        timestep += 1
        move_count += 1

    actions.append(
        Action(
            timestep=timestep,
            robot_id=ROBOT_ID,
            action=ActionType.FULFILL,
            target=(0, 0),
        )
    )

    # Validate the complete generated schedule locally before producing the
    # testbench file. This mutates world, but planning is already complete.
    simulator = Simulator(world)
    simulator.run(actions)

    if not world.orders[ORDER_ID].fulfilled:
        raise RuntimeError(
            f"Local simulation did not mark intended order {ORDER_ID} fulfilled"
        )
    if world.robots[ROBOT_ID].storage:
        raise RuntimeError("Robot storage was not cleared after fulfillment")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_submission(actions, OUTPUT_PATH)

    counts = Counter(order.skus)
    repeated_skus = sum(1 for quantity in counts.values() if quantity > 1)

    print()
    print(f"Order {ORDER_ID}: {len(order.skus)} total items")
    print(f"Distinct SKUs: {len(counts)}")
    print(f"SKUs appearing more than once: {repeated_skus}")
    print(f"Movement actions: {move_count}")
    print(f"Pick actions: {len(order.skus)}")
    print("Fulfill actions: 1")
    print(f"Total actions: {len(actions)}")
    print(f"Final robot position: {world.robots[ROBOT_ID].position}")
    print("Local simulator validation passed.")
    print(f"Wrote testbench submission to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
