"""Independent parser for Atoms Not Electrons problem files."""

from pathlib import Path
from typing import Union

from .models import OrderSpec, PalletSpec, ProblemInstance, RobotSpec


def parse_problem(path: Union[str, Path]) -> ProblemInstance:
    lines = [
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    index = 0

    def next_line() -> str:
        nonlocal index
        if index >= len(lines):
            raise ValueError("Unexpected end of problem file")
        result = lines[index]
        index += 1
        return result

    num_robots = int(next_line())
    robots = tuple(
        RobotSpec(robot_id, tuple(map(int, next_line().split())))
        for robot_id in range(num_robots)
    )

    num_skus = int(next_line())
    sku_capacities = tuple(int(next_line()) for _ in range(num_skus))

    num_pallets = int(next_line())
    pallets = []
    for pallet_id in range(num_pallets):
        x, y, sku = map(int, next_line().split())
        if not 0 <= sku < num_skus:
            raise ValueError(f"Pallet {pallet_id} references invalid SKU {sku}")
        pallets.append(PalletSpec(pallet_id, (x, y), sku, sku_capacities[sku]))

    num_orders = int(next_line())
    orders = []
    for order_id in range(num_orders):
        skus = tuple(map(int, next_line().split()))
        if any(sku < 0 or sku >= num_skus for sku in skus):
            raise ValueError(f"Order {order_id} contains an invalid SKU")
        orders.append(OrderSpec(order_id, skus))

    if index != len(lines):
        raise ValueError("Unexpected trailing data")

    return ProblemInstance(
        robots=robots,
        sku_capacities=sku_capacities,
        pallets=tuple(pallets),
        orders=tuple(orders),
    )
