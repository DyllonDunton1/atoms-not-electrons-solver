"""Internal structural checks for committed schedules."""

from __future__ import annotations

from typing import Dict, Iterable

from .geometry import WarehouseGeometry
from .models import Action, ActionType, CommittedOrderSchedule


def validate_action_uniqueness(actions: Iterable[Action]) -> None:
    seen = set()
    for action in actions:
        key = (action.timestep, action.robot_id)
        if key in seen:
            raise ValueError(f"Duplicate action for robot {action.robot_id} at t={action.timestep}")
        seen.add(key)


def validate_schedule_structure(
    schedule: CommittedOrderSchedule,
    geometry: WarehouseGeometry,
) -> None:
    if not schedule.poses:
        raise ValueError("Schedule has no poses")
    if schedule.poses[0].timestep != schedule.start_timestep:
        raise ValueError("Schedule start pose timestep mismatch")
    if schedule.poses[0].center != schedule.start_position:
        raise ValueError("Schedule start position mismatch")
    if schedule.poses[-1].timestep != schedule.finish_timestep:
        raise ValueError("Schedule finish timestep mismatch")
    if schedule.poses[-1].center != schedule.end_position:
        raise ValueError("Schedule end position mismatch")

    for previous, current in zip(schedule.poses, schedule.poses[1:]):
        if current.timestep != previous.timestep + 1:
            raise ValueError("Pose timeline is not contiguous")
        distance = abs(current.center[0] - previous.center[0]) + abs(
            current.center[1] - previous.center[1]
        )
        if distance > 1:
            raise ValueError("Pose timeline contains a multi-cell jump")
        if not geometry.pose_is_statically_valid(
            current.center,
            current.footprint_offsets,
            current.exemptions,
        ):
            raise ValueError(f"Invalid scheduled pose at t={current.timestep}")

    validate_action_uniqueness(schedule.actions)
    pose_by_time = {pose.timestep: pose for pose in schedule.poses}
    for action in schedule.actions:
        if action.robot_id != schedule.robot_id:
            raise ValueError("Schedule contains another robot's action")
        if action.timestep not in pose_by_time or action.timestep + 1 not in pose_by_time:
            raise ValueError("Action lacks adjacent poses")
        before = pose_by_time[action.timestep]
        after = pose_by_time[action.timestep + 1]
        if action.action == ActionType.MOVE:
            if after.center != action.target:
                raise ValueError("MOVE target does not match next pose")
            if before.footprint_offsets != after.footprint_offsets:
                raise ValueError("MOVE changed footprint")
        elif after.center != before.center:
            raise ValueError("Fixed action moved robot center")

    if not schedule.actions or schedule.actions[-1].action != ActionType.FULFILL:
        raise ValueError("Schedule does not end with FULFILL")
