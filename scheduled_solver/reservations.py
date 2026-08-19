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
        # A terminal hold begins only when a robot finishes its currently
        # committed schedule.  Unlike the old fixed parking cells, this lets a
        # robot fulfill at the closest legal point on y=0 while still making
        # its idle post-order occupancy visible to later planners.
        self._terminal_vertices: Dict[int, Tuple[Position, int]] = {}

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
        """Return whether ``position`` can be occupied indefinitely from start.

        Finite reservations are already stored with their configured time
        padding, so scanning those buckets from ``start_timestep`` onward also
        respects the safety margin.  Existing terminal holds are indefinite and
        therefore conflict whenever another owner holds the same cell.
        """
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

    def pallet_intervals(self, pallet_id: int) -> Tuple[Tuple[int, int, int, int], ...]:
        return tuple(
            (item.start, item.end, item.robot_id, item.order_id)
            for item in self._pallets.get(pallet_id, [])
        )

    def reservation_horizon(self) -> int:
        # Terminal holds are deliberately omitted: they are indefinite.  This
        # method answers how far existing finite schedules extend so a newly
        # finishing robot can prove its chosen row cell is safe through all
        # already-committed future movement.
        times = list(self._vertices) + list(self._edges)
        for intervals in self._pallets.values():
            times.extend(interval.end for interval in intervals)
        return max(times, default=0)
