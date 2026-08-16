"""Multi-robot scheduling and time-based collision reservations."""

from __future__ import annotations

from collections import defaultdict

from .models import Position


class ReservationTable:
    """Track cells and directed edges reserved by robots over time."""

    def __init__(self) -> None:
        self.cells: dict[int, set[Position]] = defaultdict(set)
        self.edges: dict[int, set[tuple[Position, Position]]] = defaultdict(set)

    def reserve_cell(self, timestep: int, position: Position) -> None:
        self.cells[timestep].add(position)

    def reserve_edge(
        self,
        timestep: int,
        start: Position,
        end: Position,
    ) -> None:
        self.edges[timestep].add((start, end))

    def cell_is_free(self, timestep: int, position: Position) -> bool:
        return position not in self.cells[timestep]

    def edge_is_free(
        self,
        timestep: int,
        start: Position,
        end: Position,
    ) -> bool:
        """Reject both duplicate and head-on edge conflicts."""
        reservations = self.edges[timestep]
        return (start, end) not in reservations and (end, start) not in reservations


class Scheduler:
    """Coordinate planned actions for multiple robots."""

    def __init__(self) -> None:
        self.reservations = ReservationTable()
