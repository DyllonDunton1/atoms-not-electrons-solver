"""Generate a real ten-order autonomous schedule for Tutor Testbench.

Robot 0 solves BIG_ORDER orders 0 through 9 sequentially using the Step 11
single-robot baseline. The other four robots remain stationary obstacles. Every
generated action is immediately validated by the local Simulator inside Solver,
including any replenish-and-return trips that become necessary.
"""

from collections import Counter
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models import ActionType
from src.parser import parse_problem
from src.solver import Solver
from src.world import WorldState
from src.writer import write_submission


BIG_ORDER_PATH = REPO_ROOT / "source_material" / "BIG_ORDER.txt"
OUTPUT_PATH = REPO_ROOT / "outputs" / "one_robot_10_orders.txt"
ORDER_COUNT = 10


def main() -> None:
    problem = parse_problem(BIG_ORDER_PATH)
    world = WorldState(problem)
    solver = Solver(world, robot_id=0)

    actions = solver.solve_orders(range(ORDER_COUNT))

    fulfilled_ids = [
        order_id
        for order_id in range(ORDER_COUNT)
        if world.orders[order_id].fulfilled
    ]
    if fulfilled_ids != list(range(ORDER_COUNT)):
        raise RuntimeError(
            f"Expected orders 0-{ORDER_COUNT - 1} fulfilled, got {fulfilled_ids}"
        )
    if world.robots[0].storage:
        raise RuntimeError("Robot 0 still has inventory after the final order")
    if world.robots[0].docked_pallets:
        raise RuntimeError("Robot 0 still has docked pallets after the final order")

    world.validate()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_submission(actions, OUTPUT_PATH)

    action_counts = Counter(action.action for action in actions)
    total_items = sum(len(world.orders[order_id].skus) for order_id in range(ORDER_COUNT))

    print(f"Solved orders: 0 through {ORDER_COUNT - 1}")
    print(f"Total ordered items collected: {total_items}")
    print(f"MOVE actions: {action_counts[ActionType.MOVE]}")
    print(f"PICK actions: {action_counts[ActionType.PICK]}")
    print(f"DOCK actions: {action_counts[ActionType.DOCK]}")
    print(f"UNDOCK actions: {action_counts[ActionType.UNDOCK]}")
    print(f"FULFILL actions: {action_counts[ActionType.FULFILL]}")
    print(f"Total actions: {len(actions)}")
    print(f"Final timestep: {world.timestep}")
    print(f"Final robot position: {world.robots[0].position}")
    print("Local simulator validation passed for the complete schedule.")
    print(f"Wrote testbench submission to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
