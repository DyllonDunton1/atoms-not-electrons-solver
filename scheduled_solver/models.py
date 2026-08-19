"""Independent data models for the scheduled solver.

Nothing in this package imports algorithmic code from ``src``.  The experiment
runner may convert these actions into the legacy simulator's models solely for
independent replay validation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, FrozenSet, List, Optional, Tuple


Position = Tuple[int, int]
Offset = Tuple[int, int]


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


@dataclass(frozen=True)
class RobotSpec:
    robot_id: int
    start: Position


@dataclass(frozen=True)
class PalletSpec:
    pallet_id: int
    home: Position
    sku: int
    max_count: int


@dataclass(frozen=True)
class OrderSpec:
    order_id: int
    skus: Tuple[int, ...]


@dataclass(frozen=True)
class ProblemInstance:
    robots: Tuple[RobotSpec, ...]
    sku_capacities: Tuple[int, ...]
    pallets: Tuple[PalletSpec, ...]
    orders: Tuple[OrderSpec, ...]


@dataclass(frozen=True)
class TimedPose:
    """Robot state at the start of one world timestep."""

    timestep: int
    center: Position
    footprint_offsets: FrozenSet[Offset] = frozenset({(0, 0)})
    static_exemptions: Tuple[Tuple[Offset, Position], ...] = ()

    @property
    def exemptions(self) -> Dict[Offset, Position]:
        return dict(self.static_exemptions)


@dataclass(frozen=True)
class InventoryEvent:
    """Pallet stock mutation applied at the end of ``timestep``."""

    timestep: int
    pallet_id: int
    kind: str  # "pick" or "refill"
    amount: int = 1
    robot_id: Optional[int] = None


@dataclass(frozen=True)
class PalletReservation:
    pallet_id: int
    start_timestep: int
    end_timestep: int
    robot_id: int
    order_id: int


@dataclass(frozen=True)
class ColumnVisit:
    column_id: int
    direction: str
    start_timestep: int
    end_timestep: int
    pallet_ids: Tuple[int, ...]


@dataclass(frozen=True)
class CommittedOrderSchedule:
    robot_id: int
    order_id: int
    start_timestep: int
    finish_timestep: int
    start_position: Position
    end_position: Position
    actions: Tuple[Action, ...]
    poses: Tuple[TimedPose, ...]
    inventory_events: Tuple[InventoryEvent, ...]
    pallet_reservations: Tuple[PalletReservation, ...]
    column_visits: Tuple[ColumnVisit, ...]


@dataclass
class PlannerStats:
    orders_planned: int = 0
    beam_expansions: int = 0
    beam_generated: int = 0
    beam_pruned: int = 0
    astar_calls: int = 0
    astar_expansions: int = 0
    astar_seconds: float = 0.0
    astar_capped_calls: int = 0
    astar_max_call_expansions: int = 0
    astar_max_call_seconds: float = 0.0
    astar_worst_context: str = ""
    row_fast_path_hits: int = 0
    point_fast_path_hits: int = 0
    inventory_seconds: float = 0.0
    candidate_seconds: float = 0.0
    compaction_seconds: float = 0.0
    candidate_expansions_skipped: int = 0
    candidate_full_budget_rescues: int = 0
    wait_steps: int = 0
    refill_trips: int = 0
    failed_expansions: int = 0
    planning_seconds: float = 0.0


@dataclass
class RobotScheduleState:
    robot_id: int
    available_timestep: int
    position: Position
    assigned_orders: List[int] = field(default_factory=list)
