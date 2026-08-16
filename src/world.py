"""Warehouse world state and rule checks."""

from __future__ import annotations

from .models import Pallet, Position, ProblemInstance, Robot


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
        x, y = position
        return 0 <= x < self.width and 0 <= y < self.height

    def occupied_positions(self) -> set[Position]:
        """Return cells currently occupied by robots or pallets."""
        robot_positions = {robot.position for robot in self.robots.values()}
        pallet_positions = {pallet.position for pallet in self.pallets.values()}
        return robot_positions | pallet_positions

    def validate(self) -> None:
        """Validate the current world state against challenge rules."""
        raise NotImplementedError
