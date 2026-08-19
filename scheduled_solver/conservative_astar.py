"""Conservative point fast paths for the scheduled solver.

The generic cached-static-path shortcut is excellent for speed, but choosing an
arbitrary shortest geometric route can change the reservation history compared
with space-time A* when several equal-length routes exist.  With beam width one,
that tie-breaking difference can matter many orders later.

This wrapper keeps the safe optimization only when the point-to-point shortest
route is unique: a completely clear straight horizontal or vertical segment
with no extra waiting required by ``min_goal_time``.  All other point requests
fall through to the original space-time A* implementation.  Row fast paths are
unchanged.
"""

from __future__ import annotations

from typing import FrozenSet, Mapping, Optional, Tuple

from .models import Offset, Position
from .space_time_astar import SpaceTimeAStar


class ConservativeFastPathSpaceTimeAStar(SpaceTimeAStar):
    """Use point fast paths only when they cannot alter shortest-path ties."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._allow_straight_fast_path = False

    def _straight_static_path(
        self,
        start: Position,
        goal: Position,
        footprint_offsets: FrozenSet[Offset],
        exemptions: Mapping[Offset, Position],
    ) -> Optional[Tuple[Position, ...]]:
        if start[0] != goal[0] and start[1] != goal[1]:
            return None

        if start == goal:
            return (start,)

        dx = 0 if start[0] == goal[0] else (1 if goal[0] > start[0] else -1)
        dy = 0 if start[1] == goal[1] else (1 if goal[1] > start[1] else -1)
        position = start
        path = [start]

        while position != goal:
            position = (position[0] + dx, position[1] + dy)
            if not self.geometry.pose_is_statically_valid(
                position,
                footprint_offsets,
                exemptions,
            ):
                return None
            path.append(position)

        return tuple(path)

    def _static_shortest_path(
        self,
        start: Position,
        goal: Position,
        footprint_offsets: FrozenSet[Offset],
        exemptions: Mapping[Offset, Position],
    ) -> Optional[Tuple[Position, ...]]:
        # ``SpaceTimeAStar.find_path`` calls this before its complete search.
        # Returning None disables only the point shortcut; the normal A* search
        # then runs exactly as before.
        if not self._allow_straight_fast_path:
            return None
        return self._straight_static_path(
            start,
            goal,
            footprint_offsets,
            exemptions,
        )

    def find_path(
        self,
        start: Position,
        start_time: int,
        goal: Position,
        *,
        owner: int,
        footprint_offsets: FrozenSet[Offset] = frozenset({(0, 0)}),
        static_exemptions: Mapping[Offset, Position] = {},
        min_goal_time: Optional[int] = None,
        goal_hold_steps: int = 0,
        max_expansions: Optional[int] = None,
        context: str = "point",
    ):
        direct_steps = abs(goal[0] - start[0]) + abs(goal[1] - start[1])
        natural_arrival = start_time + direct_steps

        # Do not fast-path a request that needs deliberate waiting.  A* may
        # choose a different place to wait, and that timing choice changes the
        # reservation footprint even when the geometric route is straight.
        needs_wait = min_goal_time is not None and min_goal_time > natural_arrival
        aligned = start[0] == goal[0] or start[1] == goal[1]
        self._allow_straight_fast_path = aligned and not needs_wait
        try:
            return super().find_path(
                start,
                start_time,
                goal,
                owner=owner,
                footprint_offsets=footprint_offsets,
                static_exemptions=static_exemptions,
                min_goal_time=min_goal_time,
                goal_hold_steps=goal_hold_steps,
                max_expansions=max_expansions,
                context=context,
            )
        finally:
            self._allow_straight_fast_path = False
