"""Time-expanded cell, edge, pallet, and terminal reservations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from .models import PalletReservation, Position


Edge = Tuple[Position, Position]


class ReservationConflict(RuntimeError):
    pass


@dataclass(frozen=True)
class _StoredPalletInterval:
    start: int
    end: int
    robot_id: int
    order_id: int


class ReservationTable:
    """Prioritized time-expanded reservations with configurable time padding."""

    def __init__(self, padding: int = 1) -> None:
        if padding < 0:
            raise ValueError("padding must be nonnegative")
        self.padding = padding
        self._vertices: Dict[int, Dict[Position, int]] = {}
        self._edges: Dict[int, Dict[Edge, int]] = {}
        self._pallets: Dict[int, List[_StoredPalletInterval]] = {}
        self._permanent_vertices: Dict[Position, int] = {}
        self._terminal_vertices: Dict[int, Tuple[Position, int]] = {}
        self._finite_horizon = 0

    def _times(self, timestep: int) -> range:
        return range(timestep - self.padding, timestep + self.padding + 1)

    def reserve_permanent_cell(self, position: Position, owner: int) -> None:
        existing = self._permanent_vertices.get(position)
        if existing not in (None, owner):
            raise ReservationConflict(
                f"Permanent cell {position} already reserved by robot {existing}"
            )
        self._permanent_vertices[position] = owner

    def _terminal_owner(self, position: Position, timestep: int) -> Optional[int]:
        for robot_id, (held_position, start_timestep) in self._terminal_vertices.items():
            if held_position == position and timestep >= start_timestep:
                return robot_id
        return None

    def terminal_hold(self, owner: int) -> Optional[Tuple[Position, int]]:
        """Return the owner's current indefinite post-schedule hold, if any."""
        return self._terminal_vertices.get(owner)

    def terminal_hold_is_free(
        self,
        position: Position,
        start_timestep: int,
        owner: Optional[int] = None,
    ) -> bool:
        """Return whether ``position`` can be occupied indefinitely from start."""
        permanent = self._permanent_vertices.get(position)
        if permanent not in (None, owner):
            return False

        for other_id, (other_position, _) in self._terminal_vertices.items():
            if other_id != owner and other_position == position:
                return False

        for timestep, reservations in self._vertices.items():
            if timestep < start_timestep:
                continue
            if reservations.get(position) not in (None, owner):
                return False
        return True

    def set_terminal_hold(self, position: Position, start_timestep: int, owner: int) -> None:
        """Replace ``owner``'s old idle hold with a new one after this schedule."""
        if not self.terminal_hold_is_free(position, start_timestep, owner):
            raise ReservationConflict(
                f"Terminal cell {position} from t={start_timestep} conflicts with "
                "an existing reservation"
            )
        self._terminal_vertices[owner] = (position, start_timestep)

    def vertex_owner(self, position: Position, timestep: int) -> Optional[int]:
        permanent = self._permanent_vertices.get(position)
        if permanent is not None:
            return permanent
        terminal = self._terminal_owner(position, timestep)
        if terminal is not None:
            return terminal
        return self._vertices.get(timestep, {}).get(position)

    def vertex_is_free(
        self,
        positions: Iterable[Position],
        timestep: int,
        owner: Optional[int] = None,
    ) -> bool:
        reservations = self._vertices.get(timestep, {})
        for position in positions:
            permanent = self._permanent_vertices.get(position)
            if permanent not in (None, owner):
                return False
            terminal = self._terminal_owner(position, timestep)
            if terminal not in (None, owner):
                return False
            if reservations.get(position) not in (None, owner):
                return False
        return True

    def vertex_reservation_is_free(
        self,
        positions: Iterable[Position],
        timestep: int,
        owner: Optional[int] = None,
    ) -> bool:
        cells = tuple(positions)
        return all(
            self.vertex_is_free(cells, time, owner)
            for time in self._times(timestep)
        )

    def edge_is_free(
        self,
        edges: Iterable[Edge],
        timestep: int,
        owner: Optional[int] = None,
    ) -> bool:
        reservations = self._edges.get(timestep, {})
        for edge in edges:
            existing = reservations.get(edge)
            reverse = reservations.get((edge[1], edge[0]))
            if existing not in (None, owner) or reverse not in (None, owner):
                return False
        return True

    def edge_reservation_is_free(
        self,
        edges: Iterable[Edge],
        timestep: int,
        owner: Optional[int] = None,
    ) -> bool:
        edge_list = tuple(edges)
        return all(
            self.edge_is_free(edge_list, time, owner)
            for time in self._times(timestep)
        )

    def reserve_pose(
        self,
        positions: Iterable[Position],
        timestep: int,
        owner: int,
    ) -> None:
        cells = tuple(positions)
        for time in self._times(timestep):
            bucket = self._vertices.setdefault(time, {})
            for cell in cells:
                existing = bucket.get(cell)
                if existing not in (None, owner):
                    raise ReservationConflict(
                        f"Cell {cell} at t={time} already reserved by robot {existing}"
                    )
                terminal = self._terminal_owner(cell, time)
                if terminal not in (None, owner):
                    raise ReservationConflict(
                        f"Cell {cell} at t={time} is held by idle robot {terminal}"
                    )
            for cell in cells:
                bucket[cell] = owner
        self._finite_horizon = max(self._finite_horizon, timestep + self.padding)

    def reserve_edges(
        self,
        edges: Iterable[Edge],
        timestep: int,
        owner: int,
    ) -> None:
        edge_list = tuple(edges)
        for time in self._times(timestep):
            bucket = self._edges.setdefault(time, {})
            for edge in edge_list:
                existing = bucket.get(edge)
                reverse = bucket.get((edge[1], edge[0]))
                if existing not in (None, owner) or reverse not in (None, owner):
                    blocker = existing if existing not in (None, owner) else reverse
                    raise ReservationConflict(
                        f"Edge {edge} at t={time} conflicts with robot {blocker}"
                    )
            for edge in edge_list:
                bucket[edge] = owner
        self._finite_horizon = max(self._finite_horizon, timestep + self.padding)

    def first_pallet_conflict(
        self,
        pallet_id: int,
        start: int,
        end: int,
        owner: Optional[int] = None,
    ) -> Optional[_StoredPalletInterval]:
        candidate_start = start - self.padding
        candidate_end = end + self.padding
        for interval in self._pallets.get(pallet_id, []):
            if interval.robot_id == owner:
                continue
            if candidate_start <= interval.end and interval.start <= candidate_end:
                return interval
        return None

    def pallet_is_free(
        self,
        pallet_id: int,
        start: int,
        end: int,
        owner: Optional[int] = None,
    ) -> bool:
        return self.first_pallet_conflict(pallet_id, start, end, owner) is None

    def reserve_pallet(self, reservation: PalletReservation) -> None:
        if reservation.end_timestep < reservation.start_timestep:
            raise ValueError("Pallet reservation end precedes start")
        conflict = self.first_pallet_conflict(
            reservation.pallet_id,
            reservation.start_timestep,
            reservation.end_timestep,
            reservation.robot_id,
        )
        if conflict is not None:
            raise ReservationConflict(
                f"Pallet {reservation.pallet_id} reservation conflicts with robot "
                f"{conflict.robot_id} order {conflict.order_id}"
            )
        stored = _StoredPalletInterval(
            reservation.start_timestep - self.padding,
            reservation.end_timestep + self.padding,
            reservation.robot_id,
            reservation.order_id,
        )
        bucket = self._pallets.setdefault(reservation.pallet_id, [])
        bucket.append(stored)
        bucket.sort(key=lambda item: (item.start, item.end, item.robot_id))
        self._finite_horizon = max(self._finite_horizon, stored.end)

    def pallet_intervals(self, pallet_id: int) -> Tuple[Tuple[int, int, int, int], ...]:
        return tuple(
            (item.start, item.end, item.robot_id, item.order_id)
            for item in self._pallets.get(pallet_id, [])
        )

    def compact_before(self, frontier_timestep: int) -> int:
        """Discard finite reservations that no future plan can query.

        The scheduler's planning frontier is monotonic.  A candidate beginning
        at ``frontier_timestep`` can inspect times as early as one padding width
        before it, so retain that boundary and everything after it.

        Returns the number of time buckets / pallet intervals removed.
        """
        cutoff = frontier_timestep - self.padding
        removed = 0

        old_vertex_count = len(self._vertices)
        self._vertices = {
            timestep: bucket
            for timestep, bucket in self._vertices.items()
            if timestep >= cutoff
        }
        removed += old_vertex_count - len(self._vertices)

        old_edge_count = len(self._edges)
        self._edges = {
            timestep: bucket
            for timestep, bucket in self._edges.items()
            if timestep >= cutoff
        }
        removed += old_edge_count - len(self._edges)

        for pallet_id in list(self._pallets):
            intervals = self._pallets[pallet_id]
            retained = [interval for interval in intervals if interval.end >= cutoff]
            removed += len(intervals) - len(retained)
            if retained:
                self._pallets[pallet_id] = retained
            else:
                del self._pallets[pallet_id]

        return removed

    def reservation_horizon(self) -> int:
        # Cached rather than rescanning every retained time bucket.  The cache
        # is monotonic; a stale value can only be in the past after compaction,
        # in which case fulfillment ignores it because it is <= arrival time.
        return self._finite_horizon
