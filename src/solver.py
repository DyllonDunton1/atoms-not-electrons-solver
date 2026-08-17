"""Autonomous single-robot baseline solver."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .models import Action, ActionType, Order, Pallet, Position
from .parser import parse_problem
from .pathfinding import Footprint, PathPlanner, SINGLE_ROBOT_FOOTPRINT
from .simulator import Simulator
from .world import WorldState
from .writer import write_submission


@dataclass(frozen=True)
class PickupPlan:
    """A reachable pallet, pickup cell, and optional replenishment route."""

    pallet_id: int
    path_to_pickup: List[Position]
    replenishment_path: Optional[List[Position]] = None


class Solver:
    """Generate a legal baseline schedule using one robot at a time.

    Step 11 intentionally uses only one active robot. The remaining robots stay
    at their starting cells and are treated as permanent obstacles. Orders are
    solved sequentially; task allocation and time-based reservations are added
    in later milestones.
    """

    def __init__(self, world: WorldState, robot_id: int = 0) -> None:
        if robot_id not in world.robots:
            raise ValueError(f"Unknown robot id {robot_id}")

        self.world = world
        self.robot_id = robot_id
        self.planner = PathPlanner(world)
        self.simulator = Simulator(world)
        self.actions: List[Action] = []

    @property
    def robot(self):
        return self.world.robots[self.robot_id]

    def _blocked_robots(self) -> Set[Position]:
        """Return the cells occupied by every inactive robot."""
        return {
            robot.position
            for other_id, robot in self.world.robots.items()
            if other_id != self.robot_id
        }

    def _emit(self, action_type: ActionType, target: Position) -> Action:
        """Append one action and immediately validate/apply it locally."""
        action = Action(
            timestep=self.world.timestep,
            robot_id=self.robot_id,
            action=action_type,
            target=target,
        )
        self.actions.append(action)
        self.simulator.step([action])
        return action

    def _move_along(self, path: Sequence[Position]) -> None:
        """Execute every MOVE after the first cell of a planned path."""
        if not path:
            raise RuntimeError("Cannot move along an empty path")
        if path[0] != self.robot.position:
            raise RuntimeError(
                f"Path starts at {path[0]}, but robot {self.robot_id} is at "
                f"{self.robot.position}"
            )

        for position in path[1:]:
            self._emit(ActionType.MOVE, position)

    @staticmethod
    def _required_items(order: Order) -> List[Tuple[int, int]]:
        """Return (SKU, quantity) pairs in first-appearance order."""
        quantities: Dict[int, int] = {}
        for sku in order.skus:
            quantities[sku] = quantities.get(sku, 0) + 1
        return list(quantities.items())

    def _shortest_path_to_row(
        self,
        start: Position,
        row: int,
        *,
        footprint: Footprint = SINGLE_ROBOT_FOOTPRINT,
        ignored_pallet_ids: Iterable[int] = (),
    ) -> List[Position]:
        """Return the shortest reachable path from start to any x on a row."""
        best_path: Optional[List[Position]] = None
        best_x: Optional[int] = None
        blocked = self._blocked_robots()

        for x in range(self.world.width):
            path = self.planner.find_path(
                start,
                (x, row),
                footprint=footprint,
                blocked=blocked,
                ignored_pallet_ids=ignored_pallet_ids,
            )
            if not path:
                continue

            if (
                best_path is None
                or len(path) < len(best_path)
                or (len(path) == len(best_path) and (best_x is None or x < best_x))
            ):
                best_path = path
                best_x = x

        return best_path or []

    def _reachable_pickup_candidates(self, sku: int) -> List[Tuple[int, int, Position, List[Position]]]:
        """Return reachable pallet pickup choices sorted deterministically."""
        candidates: List[Tuple[int, int, Position, List[Position]]] = []
        blocked = self._blocked_robots()

        for pallet in sorted(
            self.world.pallets_for_sku(sku),
            key=lambda candidate: candidate.pallet_id,
        ):
            for pickup_position in self.world.adjacent_positions(pallet.position):
                if pickup_position in blocked:
                    continue

                path = self.planner.find_path(
                    self.robot.position,
                    pickup_position,
                    blocked=blocked,
                )
                if not path:
                    continue

                candidates.append(
                    (
                        len(path) - 1,
                        pallet.pallet_id,
                        pickup_position,
                        path,
                    )
                )

        candidates.sort(
            key=lambda candidate: (
                candidate[0],
                candidate[1],
                candidate[2][1],
                candidate[2][0],
            )
        )
        return candidates

    def _find_pickup_plan(self, sku: int, quantity: int) -> PickupPlan:
        """Choose the nearest usable pallet for the remaining SKU quantity.

        A pallet already holding enough stock is preferred. If no reachable
        pallet has enough, choose the nearest candidate whose docked footprint
        can reach the replenishment row, and pre-plan that replenishment route.
        """
        if quantity <= 0:
            raise ValueError("Pickup quantity must be positive")

        candidates = self._reachable_pickup_candidates(sku)
        if not candidates:
            raise RuntimeError(
                f"No reachable pallet found for SKU {sku} from {self.robot.position}"
            )

        for _, pallet_id, _, path in candidates:
            pallet = self.world.pallets[pallet_id]
            if pallet.count >= quantity:
                return PickupPlan(pallet_id, path)

        for _, pallet_id, pickup_position, path in candidates:
            pallet = self.world.pallets[pallet_id]
            offset = (
                pallet.position[0] - pickup_position[0],
                pallet.position[1] - pickup_position[1],
            )
            footprint = frozenset({(0, 0), offset})
            replenishment_path = self._shortest_path_to_row(
                pickup_position,
                self.world.replenishment_y,
                footprint=footprint,
                ignored_pallet_ids=[pallet_id],
            )
            if replenishment_path:
                return PickupPlan(
                    pallet_id,
                    path,
                    replenishment_path=replenishment_path,
                )

        raise RuntimeError(
            f"Reachable pallets for SKU {sku} exist, but none can be replenished"
        )

    def _replenish_pallet(
        self,
        pallet_id: int,
        path_to_replenishment: Optional[Sequence[Position]] = None,
    ) -> None:
        """Dock, refill, return a pallet to its current home cell, and undock."""
        pallet = self.world.pallets[pallet_id]
        robot_home = self.robot.position
        pallet_home = pallet.position

        if (
            abs(robot_home[0] - pallet_home[0])
            + abs(robot_home[1] - pallet_home[1])
            != 1
        ):
            raise RuntimeError(
                f"Robot {self.robot_id} must be adjacent to pallet {pallet_id} "
                "before replenishing it"
            )

        self._emit(ActionType.DOCK, pallet_home)

        footprint = self.planner.footprint_for_robot(self.robot_id)
        if path_to_replenishment is None:
            path_to_replenishment = self._shortest_path_to_row(
                self.robot.position,
                self.world.replenishment_y,
                footprint=footprint,
                ignored_pallet_ids=self.robot.docked_pallets,
            )

        path_to_replenishment = list(path_to_replenishment)
        if not path_to_replenishment:
            raise RuntimeError(
                f"No replenishment route found for pallet {pallet_id}"
            )
        if path_to_replenishment[0] != self.robot.position:
            raise RuntimeError(
                "Preplanned replenishment route does not start at the docking cell"
            )

        self._move_along(path_to_replenishment)

        if self.robot.position[1] != self.world.replenishment_y:
            raise RuntimeError("Replenishment route did not reach y=39")
        if self.world.pallets[pallet_id].count != self.world.pallets[pallet_id].max_count:
            raise RuntimeError(f"Pallet {pallet_id} did not refill at y=39")

        # The warehouse is static in the one-robot baseline, so the reverse of
        # the legal outward footprint path is a legal deterministic return path.
        return_path = list(reversed(path_to_replenishment))
        self._move_along(return_path)

        if self.robot.position != robot_home:
            raise RuntimeError(
                f"Robot returned to {self.robot.position}, expected {robot_home}"
            )
        if self.world.pallets[pallet_id].position != pallet_home:
            raise RuntimeError(
                f"Pallet {pallet_id} returned to "
                f"{self.world.pallets[pallet_id].position}, expected {pallet_home}"
            )

        self._emit(ActionType.UNDOCK, pallet_home)

    def _pick_quantity(self, pallet_id: int, quantity: int) -> None:
        """Pick a positive quantity from the currently adjacent pallet."""
        pallet = self.world.pallets[pallet_id]
        if quantity <= 0:
            raise ValueError("Pick quantity must be positive")
        if pallet.count < quantity:
            raise RuntimeError(
                f"Pallet {pallet_id} has {pallet.count} items, cannot pick {quantity}"
            )

        for _ in range(quantity):
            self._emit(ActionType.PICK, pallet.position)

    def _collect_sku(self, sku: int, quantity: int) -> None:
        """Collect one SKU, replenishing and repeating if necessary."""
        remaining = quantity

        while remaining > 0:
            plan = self._find_pickup_plan(sku, remaining)
            self._move_along(plan.path_to_pickup)
            pallet = self.world.pallets[plan.pallet_id]

            if pallet.count < remaining:
                self._replenish_pallet(
                    pallet.pallet_id,
                    path_to_replenishment=plan.replenishment_path,
                )
                pallet = self.world.pallets[plan.pallet_id]

            quantity_to_pick = min(remaining, pallet.count)
            if quantity_to_pick <= 0:
                raise RuntimeError(
                    f"Pallet {pallet.pallet_id} still has no stock after replenishment"
                )

            self._pick_quantity(pallet.pallet_id, quantity_to_pick)
            remaining -= quantity_to_pick

    def solve_order(self, order_id: int) -> None:
        """Autonomously collect and fulfill one unfulfilled order."""
        order = self.world.orders.get(order_id)
        if order is None:
            raise ValueError(f"Unknown order id {order_id}")
        if order.fulfilled:
            raise ValueError(f"Order {order_id} is already fulfilled")
        if self.robot.storage:
            raise RuntimeError(
                f"Robot {self.robot_id} has nonempty storage before order {order_id}"
            )
        if self.robot.docked_pallets:
            raise RuntimeError(
                f"Robot {self.robot_id} still has docked pallets before order {order_id}"
            )

        self.robot.current_order = order_id

        for sku, quantity in self._required_items(order):
            self._collect_sku(sku, quantity)

        path_to_fulfillment = self._shortest_path_to_row(
            self.robot.position,
            self.world.fulfillment_y,
        )
        if not path_to_fulfillment:
            raise RuntimeError(
                f"No path to fulfillment row for order {order_id}"
            )
        self._move_along(path_to_fulfillment)
        self._emit(ActionType.FULFILL, (0, 0))

        if not self.world.orders[order_id].fulfilled:
            raise RuntimeError(
                f"Fulfill action completed a different order instead of {order_id}"
            )
        if self.robot.storage:
            raise RuntimeError(
                f"Robot {self.robot_id} storage was not cleared after order {order_id}"
            )

        self.robot.current_order = None

    def solve_orders(self, order_ids: Iterable[int]) -> List[Action]:
        """Solve the requested orders sequentially with the configured robot."""
        for order_id in order_ids:
            self.solve_order(order_id)
        return list(self.actions)

    def solve(self) -> List[Action]:
        """Solve every currently unfulfilled order in increasing ID order."""
        order_ids = [
            order_id
            for order_id in sorted(self.world.orders)
            if not self.world.orders[order_id].fulfilled
        ]
        return self.solve_orders(order_ids)


def solve_file(input_path: str | Path, output_path: str | Path) -> None:
    """Parse a challenge file, solve it, and write a submission file."""
    problem = parse_problem(input_path)
    world = WorldState(problem)
    actions = Solver(world).solve()
    write_submission(actions, output_path)
