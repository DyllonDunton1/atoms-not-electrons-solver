"""Local challenge simulator and action validator."""

from __future__ import annotations

from collections.abc import Iterable

from .models import Action
from .world import WorldState


class SimulationError(RuntimeError):
    """Raised when a generated action violates the challenge rules."""


class Simulator:
    """Execute solver actions against a local mutable world state."""

    def __init__(self, world: WorldState) -> None:
        self.world = world

    def step(self, actions: Iterable[Action]) -> None:
        """Execute one timestep worth of robot actions."""
        raise NotImplementedError

    def run(self, actions: Iterable[Action]) -> int:
        """Run a complete action schedule and return the final timestep."""
        raise NotImplementedError
