"""Write generated actions in the challenge submission format."""

from __future__ import annotations

from pathlib import Path

from .models import Action


def write_submission(actions: list[Action], path: str | Path) -> None:
    """Write actions sorted by timestep and robot id."""
    output_path = Path(path)
    ordered = sorted(actions, key=lambda action: (action.timestep, action.robot_id))

    lines = [
        f"{action.timestep} {action.robot_id} {action.action.value} "
        f"{action.target[0]} {action.target[1]}"
        for action in ordered
    ]
    output_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
