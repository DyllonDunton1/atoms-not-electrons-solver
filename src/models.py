"""Core data models used throughout the solver.

The classes in this module describe the problem state without implementing
solver strategy. Keeping state representation separate from planning logic
should make later optimization experiments easier to compare.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple


Position = Tuple[int, int]


class ActionType(str, Enum):
    MOVE = "move"
    PICK = "pick"
    DOCK = "dock"
    UNDOCK = "undock"
    FULFILL = "fulfill"


@dataclass(frozen=True)
class Action:
    timestep: int
    robot_id: int
    action: ActionType
    target: Position


@dataclass
class Pallet:
    pallet_id: int
    position: Position
    sku: int
    count: int
    max_count: int
    original_position: Position
    docked_to: Optional[int] = None


@dataclass
class Order:
    order_id: int
    skus: List[int]
    fulfilled: bool = False
    assigned_robot: Optional[int] = None


@dataclass
class Robot:
    robot_id: int
    position: Position
    storage: List[int] = field(default_factory=list)
    docked_pallets: List[int] = field(default_factory=list)
    current_order: Optional[int] = None


@dataclass
class ProblemInstance:
    robots: List[Robot]
    sku_capacities: List[int]
    pallets: List[Pallet]
    orders: List[Order]
