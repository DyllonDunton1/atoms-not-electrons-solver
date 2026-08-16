"""Write generated actions in the challenge submission format."""

from pathlib import Path
from typing import List, Set, Tuple, Union

from .models import Action


def write_submission(
    actions: List[Action],
    path: Union[str, Path],
) -> None:
    """Validate, sort, and write actions in the challenge submission format."""
    seen_actions: Set[Tuple[int, int]] = set()

    for action in actions:
        key = (action.timestep, action.robot_id)
        if key in seen_actions:
            raise ValueError(
                f"Robot {action.robot_id} has multiple actions at timestep "
                f"{action.timestep}"
            )
        seen_actions.add(key)

    ordered = sorted(
        actions,
        key=lambda action: (action.timestep, action.robot_id),
    )

    lines = [
        f"{action.timestep} {action.robot_id} {action.action.value} "
        f"{action.target[0]} {action.target[1]}"
        for action in ordered
    ]

    output_path = Path(path)
    output_path.write_text(
        "\n".join(lines) + ("\n" if lines else ""),
        encoding="utf-8",
    )
