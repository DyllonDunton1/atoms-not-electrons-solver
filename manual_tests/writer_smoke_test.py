"""Generate a small writer smoke-test submission for the Tutor testbench.

This is intentionally not a unit test. Run it manually from the repository
root, then upload the generated outputs/writer_smoke_test.txt file to the
Atoms Not Electrons testbench.
"""

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models import Action, ActionType
from src.writer import write_submission


OUTPUT_PATH = REPO_ROOT / "outputs" / "writer_smoke_test.txt"


def main() -> None:
    # These moves start from the official BIG_ORDER robot positions and stay in
    # open aisle / middle-row cells. The list is intentionally out of order so
    # writer.py also has to sort it before writing the submission.
    actions = [
        # Timestep 5
        Action(5, 4, ActionType.MOVE, (7, 18)),
        Action(5, 2, ActionType.MOVE, (20, 18)),
        Action(5, 0, ActionType.MOVE, (24, 21)),
        Action(5, 3, ActionType.MOVE, (35, 23)),
        Action(5, 1, ActionType.MOVE, (34, 21)),

        # Timestep 2
        Action(2, 3, ActionType.MOVE, (35, 26)),
        Action(2, 0, ActionType.MOVE, (22, 22)),
        Action(2, 4, ActionType.MOVE, (9, 17)),
        Action(2, 1, ActionType.MOVE, (34, 18)),
        Action(2, 2, ActionType.MOVE, (21, 20)),

        # Timestep 0
        Action(0, 2, ActionType.MOVE, (21, 22)),
        Action(0, 4, ActionType.MOVE, (9, 19)),
        Action(0, 1, ActionType.MOVE, (34, 16)),
        Action(0, 3, ActionType.MOVE, (35, 28)),
        Action(0, 0, ActionType.MOVE, (24, 22)),

        # Timestep 4
        Action(4, 1, ActionType.MOVE, (34, 20)),
        Action(4, 3, ActionType.MOVE, (35, 24)),
        Action(4, 0, ActionType.MOVE, (23, 21)),
        Action(4, 2, ActionType.MOVE, (20, 19)),
        Action(4, 4, ActionType.MOVE, (7, 17)),

        # Timestep 1
        Action(1, 4, ActionType.MOVE, (9, 18)),
        Action(1, 0, ActionType.MOVE, (23, 22)),
        Action(1, 2, ActionType.MOVE, (21, 21)),
        Action(1, 1, ActionType.MOVE, (34, 17)),
        Action(1, 3, ActionType.MOVE, (35, 27)),

        # Timestep 3
        Action(3, 2, ActionType.MOVE, (20, 20)),
        Action(3, 1, ActionType.MOVE, (34, 19)),
        Action(3, 4, ActionType.MOVE, (8, 17)),
        Action(3, 0, ActionType.MOVE, (22, 21)),
        Action(3, 3, ActionType.MOVE, (35, 25)),
    ]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_submission(actions, OUTPUT_PATH)

    print(f"Wrote {len(actions)} actions to {OUTPUT_PATH}")
    print("Upload that .txt file to the Tutor testbench for a manual smoke test.")


if __name__ == "__main__":
    main()
