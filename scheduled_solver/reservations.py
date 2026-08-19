"""Time-expanded cell, edge, and pallet reservations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

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

    def _times(self, timestep: int) -> range:
        return range(timestep - self.padding, timestep + self.padding + 1)

    def reserve_permanent_cell(self, position: Position, owner: int) -> None:
        existing = self._permanent_vertices.get(position)
        if existing not in (None, owner):
            raise ReservationConflict(
                f"Permanent cell {position} already reserved by robot {existing}"
            )
        self._permanent_vertices[position] = owner

    def vertex_owner(self, position: Position, timestep: int) -> Optional[int]:
        permanent = self._permanent_vertices.get(position)
        if permanent is not None:
            return permanent
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
        times = list(self._vertices) + list(self._edges)
        for intervals in self._pallets.values():
            times.extend(interval.end for interval in intervals)
        return max(times, default=0)
