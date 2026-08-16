"""Local challenge simulator and action validator."""

from __future__ import annotations

from typing import Dict, Iterable, List, Set

from .models import Action, ActionType, Position
from .world import WorldState


class SimulationError(RuntimeError):
    """Raised when a generated action violates the challenge rules."""


class Simulator:
    """Execute solver actions against a local mutable world state."""

    def __init__(self, world: WorldState) -> None:
        self.world = world

    def step(self, actions: Iterable[Action]) -> None:
        """Execute one movement-only timestep.

        Every supplied action must belong to the world's current timestep.
        Robots omitted from ``actions`` simply wait. The complete timestep is
        validated before any robot position is changed, so an invalid action
        cannot partially apply the timestep.
        """
        actions_list = list(actions)

        try:
            self.world.validate()
        except ValueError as error:
            raise SimulationError(f"Cannot simulate invalid world: {error}") from error

        seen_robots: Set[int] = set()
        proposed_positions: Dict[int, Position] = {}
        destination_owners: Dict[Position, int] = {}

        pallet_positions = {
            pallet.position for pallet in self.world.pallets.values()
        }
        robot_positions = {
            robot.position: robot.robot_id for robot in self.world.robots.values()
        }

        for action in actions_list:
            if action.timestep != self.world.timestep:
                raise SimulationError(
                    f"Action for robot {action.robot_id} has timestep "
                    f"{action.timestep}; expected {self.world.timestep}"
                )

            if action.robot_id in seen_robots:
                raise SimulationError(
                    f"Robot {action.robot_id} has multiple actions at timestep "
                    f"{self.world.timestep}"
                )
            seen_robots.add(action.robot_id)

            robot = self.world.robots.get(action.robot_id)
            if robot is None:
                raise SimulationError(f"Unknown robot id {action.robot_id}")

            if action.action != ActionType.MOVE:
                raise SimulationError(
                    f"Action type '{action.action.value}' is not supported by "
                    "the movement-only simulator"
                )

            target = action.target
            if not self.world.in_bounds(target):
                raise SimulationError(
                    f"Robot {action.robot_id} cannot move out of bounds to {target}"
                )

            distance = (
                abs(target[0] - robot.position[0])
                + abs(target[1] - robot.position[1])
            )
            if distance != 1:
                raise SimulationError(
                    f"Robot {action.robot_id} move from {robot.position} to {target} "
                    "is not a one-cell orthogonal move"
                )

            if target in pallet_positions:
                raise SimulationError(
                    f"Robot {action.robot_id} cannot move into pallet at {target}"
                )

            occupying_robot = robot_positions.get(target)
            if occupying_robot is not None:
                raise SimulationError(
                    f"Robot {action.robot_id} cannot move into robot "
                    f"{occupying_robot} at {target}"
                )

            other_robot = destination_owners.get(target)
            if other_robot is not None:
                raise SimulationError(
                    f"Robots {other_robot} and {action.robot_id} both target "
                    f"{target} at timestep {self.world.timestep}"
                )

            destination_owners[target] = action.robot_id
            proposed_positions[action.robot_id] = target

        # Only mutate the world after the entire timestep has been accepted.
        for robot_id, target in proposed_positions.items():
            self.world.robots[robot_id].position = target

        try:
            self.world.validate()
        except ValueError as error:
            raise SimulationError(
                f"Movement produced an invalid world: {error}"
            ) from error

        self.world.timestep += 1

    def run(self, actions: Iterable[Action]) -> int:
        """Run a complete movement schedule.

        Missing timesteps are executed as waits for every robot. The return
        value is ``world.timestep`` after the schedule finishes: the next
        timestep that would be executed (and therefore the number of elapsed
        timesteps when starting from zero).
        """
        schedule: List[Action] = list(actions)
        if not schedule:
            return self.world.timestep

        for action in schedule:
            if action.timestep < self.world.timestep:
                raise SimulationError(
                    f"Action timestep {action.timestep} is before current world "
                    f"timestep {self.world.timestep}"
                )

        actions_by_timestep: Dict[int, List[Action]] = {}
        for action in schedule:
            actions_by_timestep.setdefault(action.timestep, []).append(action)

        final_action_timestep = max(actions_by_timestep)
        while self.world.timestep <= final_action_timestep:
            timestep_actions = actions_by_timestep.get(self.world.timestep, [])
            self.step(timestep_actions)

        return self.world.timestep
