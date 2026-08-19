"""Budget-aware wrapper around the full-horizon beam planner.

Candidate pathfinding uses a deliberately smaller A* expansion budget for
speed. This wrapper makes that budget fail closed: if any A* call inside a
directed-column expansion hits the candidate cap, the whole candidate is
rejected rather than allowing the base planner to keep a partial column visit.
If capped planning still cannot complete an order, the order is retried once
with the full pathfinding budget before being declared unschedulable.

The planner also swaps in a conservative point fast-path implementation. Only
clear straight horizontal/vertical routes can bypass point A*, which preserves
the original A* tie-breaking whenever multiple equal-length routes exist.
"""

from __future__ import annotations

from typing import Optional

from .conservative_astar import ConservativeFastPathSpaceTimeAStar
from .geometry import ServiceColumn
from .models import CommittedOrderSchedule, OrderSpec, Position
from .planner import FullHorizonBeamPlanner, _BeamState


_UNSCHEDULED_PREFIX = "Full-horizon beam search could not schedule order "


class BudgetAwareFullHorizonBeamPlanner(FullHorizonBeamPlanner):
    """Full-horizon planner with safe low-budget candidate rejection."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._force_full_order_budget = False

        # The generic static shortest-path shortcut can choose a different
        # equally-short route than A*, changing later reservations. Keep the
        # shortcut only for unique clear straight paths; row fast paths remain
        # unchanged in the inherited search implementation.
        self.astar = ConservativeFastPathSpaceTimeAStar(
            self.geometry,
            self.reservations,
            path_horizon=self.config.path_horizon,
            max_expansions=self.config.max_path_expansions,
            counters=self._astar_counters,
        )

    def _expand_column(
        self,
        state: _BeamState,
        column: ServiceColumn,
        direction: str,
        robot_id: int,
        order_id: int,
        *,
        candidate_budget: bool = True,
    ) -> Optional[_BeamState]:
        # During the whole-order rescue, every candidate gets the normal full
        # A* budget. Otherwise preserve the base planner's explicit rescue
        # calls (candidate_budget=False).
        effective_candidate_budget = (
            candidate_budget and not self._force_full_order_budget
        )
        capped_before = self._astar_counters.capped_calls
        candidate = super()._expand_column(
            state,
            column,
            direction,
            robot_id,
            order_id,
            candidate_budget=effective_candidate_budget,
        )

        # A capped pallet/path search is not evidence that the pallet is truly
        # unavailable. The base column expansion may otherwise skip that
        # pallet and return a misleading partial candidate. Reject the entire
        # low-budget directed-column candidate so preselection can try another
        # route or invoke its existing full-budget rescue.
        if (
            effective_candidate_budget
            and self._astar_counters.capped_calls > capped_before
        ):
            self.stats.capped_candidate_rejections += 1
            return None
        return candidate

    def plan_order(
        self,
        robot_id: int,
        order: OrderSpec,
        start_position: Position,
        start_timestep: int,
    ) -> CommittedOrderSchedule:
        try:
            return super().plan_order(
                robot_id,
                order,
                start_position,
                start_timestep,
            )
        except RuntimeError as exc:
            # Only the base planner's explicit "could not schedule" result is
            # eligible for a budget rescue; unrelated runtime errors must still
            # surface immediately instead of being hidden by a retry.
            if not str(exc).startswith(_UNSCHEDULED_PREFIX):
                raise

            # The low cap is purely a speed heuristic. Nothing from a failed
            # order attempt has been committed yet, so it is safe to rerun the
            # same order against the identical reservation/inventory state.
            if (
                self._force_full_order_budget
                or self.config.candidate_max_path_expansions
                >= self.config.max_path_expansions
            ):
                raise

            self.stats.order_full_budget_rescues += 1
            self._force_full_order_budget = True
            try:
                return super().plan_order(
                    robot_id,
                    order,
                    start_position,
                    start_timestep,
                )
            finally:
                self._force_full_order_budget = False
