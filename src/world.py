"""Warehouse world state and rule checks."""

from __future__ import annotations

from typing import List, Optional, Set, Union

from .models import Pallet, Position, ProblemInstance, Robot


WorldEntity = Union[Robot, Pallet]


class WorldState:
    """Mutable state of the warehouse at a particular timestep."""

    width = 60
    height = 40
    fulfillment_y = 0
    replenishment_y = 39

    def __init__(self, problem: ProblemInstance) -> None:
        self.timestep = 0
        self.robots = {robot.robot_id: robot for robot in problem.robots}
        self.pallets = {pallet.pallet_id: pallet for pallet in problem.pallets}
        self.orders = {order.order_id: order for order in problem.orders}
        self.sku_capacities = list(problem.sku_capacities)

    def in_bounds(self, position: Position) -> bool:
        """Return whether a position is inside the 60 x 40 warehouse grid."""
        x, y = position
        return 0 <= x < self.width and 0 <= y < self.height

    def adjacent_positions(self, position: Position) -> List[Position]:
        """Return in-bounds orthogonal neighbors of a position."""
        x, y = position
        candidates = [
            (x - 1, y),
            (x + 1, y),
            (x, y - 1),
            (x, y + 1),
        ]
        return [candidate for candidate in candidates if self.in_bounds(candidate)]

    def occupied_positions(self) -> Set[Position]:
        """Return cells currently occupied by robots or pallets."""
        robot_positions = {robot.position for robot in self.robots.values()}
        pallet_positions = {pallet.position for pallet in self.pallets.values()}
        return robot_positions | pallet_positions

    def entity_at(self, position: Position) -> Optional[WorldEntity]:
        """Return the robot or pallet occupying a position, if one exists."""
        for robot in self.robots.values():
            if robot.position == position:
                return robot

        for pallet in self.pallets.values():
            if pallet.position == position:
                return pallet

        return None

    def pallets_for_sku(self, sku: int) -> List[Pallet]:
        """Return every pallet that stores the requested SKU."""
        return [pallet for pallet in self.pallets.values() if pallet.sku == sku]

    def validate(self) -> None:
        """Validate basic invariants of the current warehouse state.

        This checks the state itself, not whether a proposed action is legal.
        Action validation belongs in the simulator.
        """
        seen_positions = {}
        cardinal_offsets = {(-1, 0), (1, 0), (0, -1), (0, 1)}

        for robot_id, robot in self.robots.items():
            if robot.robot_id != robot_id:
                raise ValueError(
                    f"Robot dictionary key {robot_id} does not match robot id {robot.robot_id}"
                )

            if not self.in_bounds(robot.position):
                raise ValueError(
                    f"Robot {robot.robot_id} is out of bounds at {robot.position}"
                )

            if robot.position in seen_positions:
                other = seen_positions[robot.position]
                raise ValueError(
                    f"Robot {robot.robot_id} overlaps {other} at {robot.position}"
                )

            if len(robot.docked_pallets) > 4:
                raise ValueError(
                    f"Robot {robot.robot_id} has more than four docked pallets"
                )

            if len(set(robot.docked_pallets)) != len(robot.docked_pallets):
                raise ValueError(
                    f"Robot {robot.robot_id} lists a docked pallet more than once"
                )

            used_offsets = set()
            for pallet_id in robot.docked_pallets:
                pallet = self.pallets.get(pallet_id)
                if pallet is None:
                    raise ValueError(
                        f"Robot {robot.robot_id} references unknown docked pallet {pallet_id}"
                    )

                if pallet.docked_to != robot.robot_id:
                    raise ValueError(
                        f"Robot {robot.robot_id} lists pallet {pallet_id} as docked, "
                        f"but pallet owner is {pallet.docked_to}"
                    )

                if pallet.docked_offset not in cardinal_offsets:
                    raise ValueError(
                        f"Pallet {pallet_id} has invalid docked offset "
                        f"{pallet.docked_offset}"
                    )

                if pallet.docked_offset in used_offsets:
                    raise ValueError(
                        f"Robot {robot.robot_id} has multiple pallets on side "
                        f"{pallet.docked_offset}"
                    )
                used_offsets.add(pallet.docked_offset)

                offset_x, offset_y = pallet.docked_offset
                expected_position = (
                    robot.position[0] + offset_x,
                    robot.position[1] + offset_y,
                )
                if pallet.position != expected_position:
                    raise ValueError(
                        f"Docked pallet {pallet_id} is at {pallet.position}; expected "
                        f"{expected_position} beside robot {robot.robot_id}"
                    )

            seen_positions[robot.position] = f"robot {robot.robot_id}"

        for pallet_id, pallet in self.pallets.items():
            if pallet.pallet_id != pallet_id:
                raise ValueError(
                    f"Pallet dictionary key {pallet_id} does not match pallet id {pallet.pallet_id}"
                )

            if not self.in_bounds(pallet.position):
                raise ValueError(
                    f"Pallet {pallet.pallet_id} is out of bounds at {pallet.position}"
                )

            if not self.in_bounds(pallet.original_position):
                raise ValueError(
                    f"Pallet {pallet.pallet_id} has out-of-bounds original position "
                    f"{pallet.original_position}"
                )

            if pallet.position in seen_positions:
                other = seen_positions[pallet.position]
                raise ValueError(
                    f"Pallet {pallet.pallet_id} overlaps {other} at {pallet.position}"
                )

            if not 0 <= pallet.sku < len(self.sku_capacities):
                raise ValueError(
                    f"Pallet {pallet.pallet_id} references invalid SKU {pallet.sku}"
                )

            expected_capacity = self.sku_capacities[pallet.sku]
            if pallet.max_count != expected_capacity:
                raise ValueError(
                    f"Pallet {pallet.pallet_id} max count {pallet.max_count} does not "
                    f"match SKU {pallet.sku} capacity {expected_capacity}"
                )

            if not 0 <= pallet.count <= pallet.max_count:
                raise ValueError(
                    f"Pallet {pallet.pallet_id} has invalid count {pallet.count}; "
                    f"expected 0 through {pallet.max_count}"
                )

            if pallet.docked_to is None:
                if pallet.docked_offset is not None:
                    raise ValueError(
                        f"Undocked pallet {pallet.pallet_id} still has docked offset "
                        f"{pallet.docked_offset}"
                    )
            else:
                owner = self.robots.get(pallet.docked_to)
                if owner is None:
                    raise ValueError(
                        f"Pallet {pallet.pallet_id} is docked to unknown robot "
                        f"{pallet.docked_to}"
                    )
                if pallet.pallet_id not in owner.docked_pallets:
                    raise ValueError(
                        f"Pallet {pallet.pallet_id} says it is docked to robot "
                        f"{pallet.docked_to}, but the robot does not list it"
                    )
                if pallet.docked_offset not in cardinal_offsets:
                    raise ValueError(
                        f"Pallet {pallet.pallet_id} has invalid docked offset "
                        f"{pallet.docked_offset}"
                    )

            seen_positions[pallet.position] = f"pallet {pallet.pallet_id}"
