"""Write scheduled-solver actions in challenge submission format."""

from pathlib import Path
from typing import Iterable, Set, Tuple, Union

from .models import Action


def write_submission(actions: Iterable[Action], path: Union[str, Path]) -> None:
    ordered = sorted(actions, key=lambda action: (action.timestep, action.robot_id))
    seen: Set[Tuple[int, int]] = set()
    lines = []
    for action in ordered:
        key = (action.timestep, action.robot_id)
        if key in seen:
            raise ValueError(f"Duplicate action for robot {action.robot_id} at t={action.timestep}")
        seen.add(key)
        lines.append(
            f"{action.timestep} {action.robot_id} {action.action.value} "
            f"{action.target[0]} {action.target[1]}"
        )
    Path(path).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
