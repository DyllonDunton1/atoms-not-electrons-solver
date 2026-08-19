"""Small planning-metrics helpers kept independent from the legacy metrics code."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Union

from .models import Action, ActionType, PlannerStats


def planning_metrics(actions: Iterable[Action], stats: PlannerStats) -> dict:
    action_list = list(actions)
    counts = {action_type.value: 0 for action_type in ActionType}
    for action in action_list:
        counts[action.action.value] += 1
    return {
        "planner": asdict(stats),
        "action_counts": counts,
        "actions": len(action_list),
        "makespan": max((action.timestep for action in action_list), default=-1) + 1,
    }


def write_metrics_json(metrics: dict, path: Union[str, Path]) -> None:
    Path(path).write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
