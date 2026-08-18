"""One-timestep collision reservations for prioritized fleet movement."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Set, Tuple

from .models import Position
from .pathfinding import Footprint


class ReservationTable:
    """Track already-committed moves for the current fleet timestep.

    The fleet solver replans spatial routes from the real world state every
    timestep. Reservations therefore exist only to make the first moves chosen
    by lower-ID robots authoritative while higher-ID robots are planned.
    """

    def __init__(self) -> None:
        self.cells: Dict[int, Set[Position]] = defaultdict(set)
        self.edges: Dict[int, Set[Tuple[Position, Position]]] = defaultdict(set)

    @staticmethod
    def footprint_cells(center: Position, footprint: Footprint) -> Set[Position]:
        """Return the absolute cells occupied by a robot-relative footprint."""
        center_x, center_y = center
        return {
            (center_x + offset_x, center_y + offset_y)
            for offset_x, offset_y in footprint
        }

    def reserve_cell(self, timestep: int, position: Position) -> None:
        self.cells[timestep].add(position)

    def reserve_edge(
        self,
        timestep: int,
        start: Position,
        end: Position,
    ) -> None:
        self.edges[timestep].add((start, end))

    def reserve_footprint(
        self,
        timestep: int,
        center: Position,
        footprint: Footprint,
    ) -> None:
        for position in self.footprint_cells(center, footprint):
            self.reserve_cell(timestep, position)

    def reserve_transition(
        self,
        timestep: int,
        start: Position,
        end: Position,
        footprint: Footprint,
    ) -> None:
        """Reserve one wait/move transition for a complete rigid footprint.

        Destination cells are reserved both at the action-start state and at
        the resulting state. This matches the simulator's conservative rule:
        another robot may not occupy a cell at the start of a timestep if a
        lower-ID robot intends to enter that cell during the timestep.
        """
        distance = abs(end[0] - start[0]) + abs(end[1] - start[1])
        if distance not in (0, 1):
            raise ValueError(
                f"Transition from {start} to {end} is not a wait or one-cell move"
            )

        self.reserve_footprint(timestep, start, footprint)
        self.reserve_footprint(timestep, end, footprint)
        self.reserve_footprint(timestep + 1, end, footprint)

        delta = (end[0] - start[0], end[1] - start[1])
        for offset_x, offset_y in footprint:
            entity_start = (start[0] + offset_x, start[1] + offset_y)
            entity_end = (
                entity_start[0] + delta[0],
                entity_start[1] + delta[1],
            )
            self.reserve_edge(timestep, entity_start, entity_end)

    def cell_is_free(self, timestep: int, position: Position) -> bool:
        return position not in self.cells.get(timestep, set())

    def footprint_is_free(
        self,
        timestep: int,
        center: Position,
        footprint: Footprint,
    ) -> bool:
        return all(
            self.cell_is_free(timestep, position)
            for position in self.footprint_cells(center, footprint)
        )

    def edge_is_free(
        self,
        timestep: int,
        start: Position,
        end: Position,
    ) -> bool:
        """Reject duplicate and head-on use of an already-reserved edge."""
        reservations = self.edges.get(timestep, set())
        return (start, end) not in reservations and (end, start) not in reservations

    def transition_is_free(
        self,
        timestep: int,
        start: Position,
        end: Position,
        footprint: Footprint,
    ) -> bool:
        """Return whether a proposed first move respects lower-ID commitments."""
        distance = abs(end[0] - start[0]) + abs(end[1] - start[1])
        if distance not in (0, 1):
            return False

        if not self.footprint_is_free(timestep, end, footprint):
            return False
        if not self.footprint_is_free(timestep + 1, end, footprint):
            return False

        delta = (end[0] - start[0], end[1] - start[1])
        for offset_x, offset_y in footprint:
            entity_start = (start[0] + offset_x, start[1] + offset_y)
            entity_end = (
                entity_start[0] + delta[0],
                entity_start[1] + delta[1],
            )
            if not self.edge_is_free(timestep, entity_start, entity_end):
                return False

        return True
