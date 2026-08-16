"""Parsing utilities for BIG_ORDER.txt."""

from pathlib import Path
from typing import List, Union

from .models import Order, Pallet, ProblemInstance, Robot


def parse_problem(path: Union[str, Path]) -> ProblemInstance:
    """Parse an Atoms Not Electrons worklist into a ProblemInstance."""
    lines = [
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    index = 0

    def next_line() -> str:
        nonlocal index
        if index >= len(lines):
            raise ValueError("Unexpected end of BIG_ORDER file")
        line = lines[index]
        index += 1
        return line

    # Robots
    num_robots = int(next_line())
    robots: List[Robot] = []
    for robot_id in range(num_robots):
        x, y = map(int, next_line().split())
        robots.append(Robot(robot_id=robot_id, position=(x, y)))

    # SKU capacities
    num_skus = int(next_line())
    sku_capacities = [int(next_line()) for _ in range(num_skus)]

    # Pallets
    num_pallets = int(next_line())
    pallets: List[Pallet] = []
    for pallet_id in range(num_pallets):
        x, y, sku = map(int, next_line().split())
        if not 0 <= sku < num_skus:
            raise ValueError(f"Pallet {pallet_id} references invalid SKU {sku}")

        position = (x, y)
        capacity = sku_capacities[sku]
        pallets.append(
            Pallet(
                pallet_id=pallet_id,
                position=position,
                sku=sku,
                count=capacity,
                max_count=capacity,
                original_position=position,
            )
        )

    # Orders: each remaining order is one line of SKU ids.
    num_orders = int(next_line())
    orders: List[Order] = []
    for order_id in range(num_orders):
        skus = [int(sku) for sku in next_line().split()]
        if any(sku < 0 or sku >= num_skus for sku in skus):
            raise ValueError(f"Order {order_id} contains an invalid SKU")
        orders.append(Order(order_id=order_id, skus=skus))

    if index != len(lines):
        raise ValueError(f"Unexpected trailing data after order {num_orders - 1}")

    return ProblemInstance(
        robots=robots,
        sku_capacities=sku_capacities,
        pallets=pallets,
        orders=orders,
    )
