"""Generate a short real BIG_ORDER docking sequence for the Tutor testbench.

Robot 0 starts at (25, 22), directly above the pallet at (25, 23). This manual
smoke test docks that pallet on the robot's bottom side, moves the combined
footprint five cells upward through the open middle aisle, then undocks it.
The complete sequence is validated by the local Simulator before writer.py
creates the uploadable text file.
"""

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models import Action, ActionType, Pallet
from src.parser import parse_problem
from src.simulator import Simulator
from src.world import WorldState
from src.writer import write_submission


BIG_ORDER_PATH = REPO_ROOT / "source_material" / "BIG_ORDER.txt"
OUTPUT_PATH = REPO_ROOT / "outputs" / "docking_smoke_test.txt"
ROBOT_ID = 0
START_PALLET_POSITION = (25, 23)


def main() -> None:
    problem = parse_problem(BIG_ORDER_PATH)
    world = WorldState(problem)
    simulator = Simulator(world)

    robot = world.robots[ROBOT_ID]
    if robot.position != (25, 22):
        raise RuntimeError(
            f"Expected robot 0 at (25, 22), found {robot.position}"
        )

    pallet = world.entity_at(START_PALLET_POSITION)
    if not isinstance(pallet, Pallet):
        raise RuntimeError(
            f"Expected a pallet at {START_PALLET_POSITION}, found {pallet}"
        )
    pallet_id = pallet.pallet_id

    actions = [
        Action(0, ROBOT_ID, ActionType.DOCK, START_PALLET_POSITION),
        Action(1, ROBOT_ID, ActionType.MOVE, (25, 21)),
        Action(2, ROBOT_ID, ActionType.MOVE, (25, 20)),
        Action(3, ROBOT_ID, ActionType.MOVE, (25, 19)),
        Action(4, ROBOT_ID, ActionType.MOVE, (25, 18)),
        Action(5, ROBOT_ID, ActionType.MOVE, (25, 17)),
        Action(6, ROBOT_ID, ActionType.UNDOCK, (25, 18)),
    ]

    simulator.run(actions)

    moved_pallet = world.pallets[pallet_id]
    if world.robots[ROBOT_ID].position != (25, 17):
        raise RuntimeError("Robot 0 did not finish at the expected position")
    if moved_pallet.position != (25, 18):
        raise RuntimeError("Docked pallet did not move with robot 0 as expected")
    if moved_pallet.docked_to is not None:
        raise RuntimeError("Pallet remained docked after the undock action")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_submission(actions, OUTPUT_PATH)

    print(f"Robot 0 started at (25, 22)")
    print(
        f"Docked pallet {pallet_id} at {START_PALLET_POSITION}, "
        "moved five cells upward, and undocked it at (25, 18)."
    )
    print("Local simulator validation passed.")
    print(f"Wrote testbench submission to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
