"""Static warehouse geometry and the independent 24-column representation."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, FrozenSet, Iterable, List, Mapping, Sequence, Set, Tuple

from .models import Offset, PalletSpec, Position


UP = "up"
DOWN = "down"
DIRECTIONS = (UP, DOWN)


@dataclass(frozen=True)
class ServiceColumn:
    column_id: int
    pallet_x: int
    service_x: int
    pallet_ids: Tuple[int, ...]
    homes: Tuple[Position, ...]

    @property
    def service_cells(self) -> Tuple[Position, ...]:
        return tuple((self.service_x, y) for _, y in self.homes)


@dataclass(frozen=True)
class WarehouseGeometry:
    width: int
    height: int
    fulfillment_y: int
    replenishment_y: int
    static_blocked: FrozenSet[Position]
    columns: Tuple[ServiceColumn, ...]
    pallet_to_column: Mapping[int, int]

    def in_bounds(self, position: Position) -> bool:
        x, y = position
        return 0 <= x < self.width and 0 <= y < self.height

    def footprint_cells(self, center: Position, offsets: Iterable[Offset]) -> FrozenSet[Position]:
        return frozenset((center[0] + dx, center[1] + dy) for dx, dy in offsets)

    def pose_is_statically_valid(
        self,
        center: Position,
        offsets: Iterable[Offset],
        exemptions: Mapping[Offset, Position] = {},
    ) -> bool:
        for offset in offsets:
            cell = (center[0] + offset[0], center[1] + offset[1])
            if not self.in_bounds(cell):
                return False
            if cell in self.static_blocked and exemptions.get(offset) != cell:
                return False
        return True


def _components(homes: Set[Position]) -> List[Set[Position]]:
    remaining = set(homes)
    result = []
    while remaining:
        start = min(remaining, key=lambda p: (p[1], p[0]))
        component = {start}
        remaining.remove(start)
        queue = deque([start])
        while queue:
            x, y = queue.popleft()
            for neighbor in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    component.add(neighbor)
                    queue.append(neighbor)
        result.append(component)
    return result


def build_geometry(
    pallets: Sequence[PalletSpec],
    *,
    width: int = 60,
    height: int = 40,
    fulfillment_y: int = 0,
    replenishment_y: int = 39,
    require_24_columns: bool = True,
) -> WarehouseGeometry:
    """Build columns from rectangular 2xN connected pallet islands.

    Pallet home cells are permanently static obstacles, even while a pallet is
    physically docked and moved elsewhere by a robot.
    """
    homes = {pallet.home for pallet in pallets}
    if len(homes) != len(pallets):
        raise ValueError("Pallet home positions must be unique")
    by_home = {pallet.home: pallet for pallet in pallets}

    raw: List[Tuple[int, int, int, Tuple[int, ...], Tuple[Position, ...]]] = []
    for component in _components(homes):
        xs = sorted({x for x, _ in component})
        ys = sorted({y for _, y in component})
        if len(xs) != 2:
            raise ValueError(f"Pallet island must be two columns wide; got x={xs}")
        expected = {(x, y) for x in xs for y in ys}
        if component != expected:
            raise ValueError("Pallet island must be a solid 2xN rectangle")
        left_x, right_x = xs
        for pallet_x in xs:
            column_homes = tuple((pallet_x, y) for y in ys)
            pallet_ids = tuple(by_home[home].pallet_id for home in column_homes)
            service_x = pallet_x - 1 if pallet_x == left_x else pallet_x + 1
            raw.append((ys[0], pallet_x, service_x, pallet_ids, column_homes))

    raw.sort(key=lambda item: (item[0], item[1]))
    columns = []
    pallet_to_column: Dict[int, int] = {}
    for column_id, (_, pallet_x, service_x, pallet_ids, column_homes) in enumerate(raw):
        if not 0 <= service_x < width:
            raise ValueError("Service lane lies outside warehouse")
        column = ServiceColumn(column_id, pallet_x, service_x, pallet_ids, column_homes)
        columns.append(column)
        for pallet_id in pallet_ids:
            pallet_to_column[pallet_id] = column_id

    if require_24_columns and len(columns) != 24:
        raise ValueError(f"Expected 24 service columns, found {len(columns)}")

    return WarehouseGeometry(
        width=width,
        height=height,
        fulfillment_y=fulfillment_y,
        replenishment_y=replenishment_y,
        static_blocked=frozenset(homes),
        columns=tuple(columns),
        pallet_to_column=pallet_to_column,
    )
