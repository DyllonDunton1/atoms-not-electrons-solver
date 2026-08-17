"""Warehouse aisle geometry and aisle-aware collection planning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from .models import Pallet, Position
from .pathfinding import PathPlanner
from .world import WorldState


CONGESTION_DISTANCE_PENALTY = 8
TOP_AISLE_CANDIDATES = 3


@dataclass(frozen=True)
class Aisle:
    """One connected 2 x 10 pallet island in the warehouse."""

    aisle_id: int
    pallet_ids: Tuple[int, ...]
    home_positions: Tuple[Position, ...]
    service_cells: Tuple[Position, ...]


@dataclass(frozen=True)
class AisleLayout:
    """Deterministic pallet-to-aisle mappings shared by solver and metrics."""

    aisles: Tuple[Aisle, ...]
    pallet_to_aisle: Dict[int, int]
    home_to_aisle: Dict[Position, int]


@dataclass(frozen=True)
class AisleStop:
    """One SKU service target within a selected aisle."""

    sku: int
    quantity: int
    pallet_id: int
    pickup: Position


@dataclass(frozen=True)
class AislePlan:
    """Concrete partial route that services all currently useful SKUs in an aisle."""

    aisle_id: int
    stops: Tuple[AisleStop, ...]
    useful_quantity: int
    planned_distance: int
    congestion: int
    score: float


@dataclass(frozen=True)
class _ServiceOption:
    sku: int
    quantity: int
    pallet_id: int
    pickup: Position
    stock_sufficient: bool


def build_aisle_layout(pallets: Iterable[Pallet]) -> AisleLayout:
    """Group orthogonally connected pallet homes into deterministic aisle ids."""
    pallet_list = list(pallets)
    home_to_pallet = {pallet.original_position: pallet for pallet in pallet_list}
    if len(home_to_pallet) != len(pallet_list):
        raise ValueError("Pallet original positions must be unique")

    remaining = set(home_to_pallet)
    components: List[Set[Position]] = []

    while remaining:
        start = min(remaining)
        stack = [start]
        component: Set[Position] = set()
        remaining.remove(start)

        while stack:
            position = stack.pop()
            component.add(position)
            x, y = position
            for neighbor in (
                (x - 1, y),
                (x + 1, y),
                (x, y - 1),
                (x, y + 1),
            ):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    stack.append(neighbor)

        components.append(component)

    components.sort(
        key=lambda component: (
            min(y for _, y in component),
            min(x for x, _ in component),
        )
    )

    all_homes = set(home_to_pallet)
    aisles: List[Aisle] = []
    pallet_to_aisle: Dict[int, int] = {}
    home_to_aisle: Dict[Position, int] = {}

    for aisle_id, component in enumerate(components):
        service_cells: Set[Position] = set()
        for x, y in component:
            for neighbor in (
                (x - 1, y),
                (x + 1, y),
                (x, y - 1),
                (x, y + 1),
            ):
                if neighbor not in all_homes:
                    service_cells.add(neighbor)

        pallet_ids = tuple(
            pallet.pallet_id
            for pallet in sorted(
                (home_to_pallet[position] for position in component),
                key=lambda pallet: (
                    pallet.original_position[1],
                    pallet.original_position[0],
                    pallet.pallet_id,
                ),
            )
        )
        home_positions = tuple(
            sorted(component, key=lambda position: (position[1], position[0]))
        )

        aisles.append(
            Aisle(
                aisle_id=aisle_id,
                pallet_ids=pallet_ids,
                home_positions=home_positions,
                service_cells=tuple(
                    sorted(service_cells, key=lambda position: (position[1], position[0]))
                ),
            )
        )
        for pallet_id in pallet_ids:
            pallet_to_aisle[pallet_id] = aisle_id
        for position in component:
            home_to_aisle[position] = aisle_id

    return AisleLayout(
        aisles=tuple(aisles),
        pallet_to_aisle=pallet_to_aisle,
        home_to_aisle=home_to_aisle,
    )


class AislePlanner:
    """Choose and concretely route one useful aisle at a time."""

    def __init__(
        self,
        world: WorldState,
        *,
        congestion_distance_penalty: int = CONGESTION_DISTANCE_PENALTY,
        top_candidates: int = TOP_AISLE_CANDIDATES,
    ) -> None:
        if congestion_distance_penalty < 0:
            raise ValueError("congestion_distance_penalty must be nonnegative")
        if top_candidates <= 0:
            raise ValueError("top_candidates must be positive")

        self.world = world
        self.spatial = PathPlanner(world)
        self.layout = build_aisle_layout(world.pallets.values())
        self.congestion_distance_penalty = congestion_distance_penalty
        self.top_candidates = top_candidates

    def aisle_for_pallet(self, pallet_id: int) -> int:
        try:
            return self.layout.pallet_to_aisle[pallet_id]
        except KeyError as exception:
            raise ValueError(f"Unknown pallet id {pallet_id}") from exception

    def choose_plan(
        self,
        start: Position,
        remaining_by_sku: Mapping[int, int],
        *,
        congestion_by_aisle: Mapping[int, int],
        unavailable_pallet_ids: Iterable[int] = (),
        blocked: Iterable[Position] = (),
    ) -> Optional[AislePlan]:
        """Shortlist all aisles cheaply, then fully plan the best three."""
        unavailable = set(unavailable_pallet_ids)
        blocked_set = set(blocked)
        cheap_candidates: List[Tuple[float, int]] = []

        for aisle in self.layout.aisles:
            options_by_sku = self._options_by_sku(
                aisle,
                remaining_by_sku,
                unavailable,
            )
            if not options_by_sku:
                continue

            useful_quantity = sum(
                remaining_by_sku[sku] for sku in options_by_sku
            )
            estimated_distance = min(
                abs(start[0] - option.pickup[0]) + abs(start[1] - option.pickup[1])
                for options in options_by_sku.values()
                for option in options
            )
            congestion = congestion_by_aisle.get(aisle.aisle_id, 0)
            cheap_score = self._score(
                useful_quantity,
                estimated_distance,
                congestion,
            )
            cheap_candidates.append((cheap_score, aisle.aisle_id))

        cheap_candidates.sort(key=lambda item: (-item[0], item[1]))

        detailed: List[AislePlan] = []
        for _, aisle_id in cheap_candidates:
            plan = self.plan_aisle(
                aisle_id,
                start,
                remaining_by_sku,
                congestion=congestion_by_aisle.get(aisle_id, 0),
                unavailable_pallet_ids=unavailable,
                blocked=blocked_set,
            )
            if plan is not None:
                detailed.append(plan)
                if len(detailed) == self.top_candidates:
                    break

        if not detailed:
            return None

        detailed.sort(
            key=lambda plan: (
                -plan.score,
                plan.planned_distance,
                -plan.useful_quantity,
                plan.aisle_id,
            )
        )
        return detailed[0]

    def plan_aisle(
        self,
        aisle_id: int,
        start: Position,
        remaining_by_sku: Mapping[int, int],
        *,
        congestion: int = 0,
        unavailable_pallet_ids: Iterable[int] = (),
        blocked: Iterable[Position] = (),
    ) -> Optional[AislePlan]:
        """Build the best multi-start nearest-neighbor route for one aisle."""
        if aisle_id < 0 or aisle_id >= len(self.layout.aisles):
            raise ValueError(f"Unknown aisle id {aisle_id}")

        aisle = self.layout.aisles[aisle_id]
        unavailable = set(unavailable_pallet_ids)
        blocked_set = set(blocked)
        options_by_sku = self._options_by_sku(
            aisle,
            remaining_by_sku,
            unavailable,
        )
        if not options_by_sku:
            return None

        useful_quantity = sum(
            remaining_by_sku[sku] for sku in options_by_sku
        )
        distance_cache: Dict[Tuple[Position, Position], Optional[int]] = {}

        first_options: List[Tuple[int, _ServiceOption]] = []
        for sku in sorted(options_by_sku):
            for option in self._preferred_options(options_by_sku[sku]):
                distance = self._distance(
                    start,
                    option.pickup,
                    blocked_set,
                    distance_cache,
                )
                if distance is not None:
                    first_options.append((distance, option))

        if not first_options:
            return None

        best_stops: Optional[Tuple[AisleStop, ...]] = None
        best_distance: Optional[int] = None

        for first_distance, first_option in first_options:
            remaining_skus = set(options_by_sku)
            remaining_skus.remove(first_option.sku)
            stops = [
                AisleStop(
                    sku=first_option.sku,
                    quantity=first_option.quantity,
                    pallet_id=first_option.pallet_id,
                    pickup=first_option.pickup,
                )
            ]
            current = first_option.pickup
            total_distance = first_distance
            valid = True

            while remaining_skus:
                next_choice = None
                next_key = None

                for sku in sorted(remaining_skus):
                    for option in self._preferred_options(options_by_sku[sku]):
                        distance = self._distance(
                            current,
                            option.pickup,
                            blocked_set,
                            distance_cache,
                        )
                        if distance is None:
                            continue
                        key = (
                            distance,
                            option.pallet_id,
                            option.pickup[1],
                            option.pickup[0],
                            sku,
                        )
                        if next_key is None or key < next_key:
                            next_key = key
                            next_choice = option

                if next_choice is None or next_key is None:
                    valid = False
                    break

                total_distance += next_key[0]
                stops.append(
                    AisleStop(
                        sku=next_choice.sku,
                        quantity=next_choice.quantity,
                        pallet_id=next_choice.pallet_id,
                        pickup=next_choice.pickup,
                    )
                )
                current = next_choice.pickup
                remaining_skus.remove(next_choice.sku)

            if not valid:
                continue

            stop_tuple = tuple(stops)
            if (
                best_distance is None
                or total_distance < best_distance
                or (
                    total_distance == best_distance
                    and self._stop_key(stop_tuple) < self._stop_key(best_stops)
                )
            ):
                best_distance = total_distance
                best_stops = stop_tuple

        if best_stops is None or best_distance is None:
            return None

        return AislePlan(
            aisle_id=aisle_id,
            stops=best_stops,
            useful_quantity=useful_quantity,
            planned_distance=best_distance,
            congestion=congestion,
            score=self._score(useful_quantity, best_distance, congestion),
        )

    def _options_by_sku(
        self,
        aisle: Aisle,
        remaining_by_sku: Mapping[int, int],
        unavailable_pallet_ids: Set[int],
    ) -> Dict[int, List[_ServiceOption]]:
        result: Dict[int, List[_ServiceOption]] = {}

        for pallet_id in aisle.pallet_ids:
            pallet = self.world.pallets[pallet_id]
            quantity = remaining_by_sku.get(pallet.sku, 0)
            if quantity <= 0:
                continue
            if pallet_id in unavailable_pallet_ids:
                continue
            if pallet.docked_to is not None:
                continue
            if pallet.position != pallet.original_position:
                continue

            options = result.setdefault(pallet.sku, [])
            for pickup in self.world.adjacent_positions(pallet.position):
                if pickup in self.layout.home_to_aisle:
                    continue
                # If this stop will require a refill, do not choose a pickup
                # above the pallet. Docking from above puts the pallet below
                # the robot, which would place it at y=40 when the robot reaches
                # replenishment row y=39.
                if (
                    quantity > pallet.count
                    and pickup[0] == pallet.position[0]
                    and pickup[1] == pallet.position[1] - 1
                ):
                    continue
                options.append(
                    _ServiceOption(
                        sku=pallet.sku,
                        quantity=quantity,
                        pallet_id=pallet_id,
                        pickup=pickup,
                        stock_sufficient=pallet.count >= quantity,
                    )
                )

        return result

    @staticmethod
    def _preferred_options(options: Sequence[_ServiceOption]) -> Sequence[_ServiceOption]:
        stocked = [option for option in options if option.stock_sufficient]
        return stocked if stocked else options

    def _distance(
        self,
        start: Position,
        goal: Position,
        blocked: Set[Position],
        cache: Dict[Tuple[Position, Position], Optional[int]],
    ) -> Optional[int]:
        key = (start, goal)
        if key in cache:
            return cache[key]

        path = self.spatial.find_path(start, goal, blocked=blocked)
        distance = len(path) - 1 if path else None
        cache[key] = distance
        cache[(goal, start)] = distance
        return distance

    def _score(self, useful_quantity: int, distance: int, congestion: int) -> float:
        denominator = (
            distance
            + 1
            + congestion * self.congestion_distance_penalty
        )
        return float(useful_quantity) / float(denominator)

    @staticmethod
    def _stop_key(stops: Optional[Tuple[AisleStop, ...]]) -> Tuple[Tuple[int, int, Position], ...]:
        if stops is None:
            return tuple()
        return tuple(
            (stop.sku, stop.pallet_id, stop.pickup)
            for stop in stops
        )
