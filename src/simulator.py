"""Local challenge simulator and action validator."""

from __future__ import annotations

from collections import Counter
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
        """Execute one timestep of currently supported challenge actions.

        Robots omitted from ``actions`` wait. MOVE, PICK, and FULFILL are
        supported; docking actions are added later. The complete timestep is
        validated before any state is changed so an invalid action cannot
        partially apply the timestep.
        """
        actions_list = list(actions)

        try:
            self.world.validate()
        except ValueError as error:
            raise SimulationError(f"Cannot simulate invalid world: {error}") from error

        seen_robots: Set[int] = set()
        proposed_positions: Dict[int, Position] = {}
        destination_owners: Dict[Position, int] = {}
        pick_requests: Dict[int, int] = {}
        picks_per_pallet: Dict[int, int] = {}
        fulfillment_orders: Dict[int, int] = {}
        reserved_order_ids: Set[int] = set()

        pallet_by_position = {
            pallet.position: pallet for pallet in self.world.pallets.values()
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

            if action.action == ActionType.MOVE:
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

                if target in pallet_by_position:
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

            elif action.action == ActionType.PICK:
                pallet = pallet_by_position.get(action.target)
                if pallet is None:
                    raise SimulationError(
                        f"Robot {action.robot_id} cannot pick from {action.target}; "
                        "no pallet is there"
                    )

                distance = (
                    abs(action.target[0] - robot.position[0])
                    + abs(action.target[1] - robot.position[1])
                )
                if distance != 1:
                    raise SimulationError(
                        f"Robot {action.robot_id} at {robot.position} is not adjacent "
                        f"to pallet {pallet.pallet_id} at {action.target}"
                    )

                pick_requests[action.robot_id] = pallet.pallet_id
                picks_per_pallet[pallet.pallet_id] = (
                    picks_per_pallet.get(pallet.pallet_id, 0) + 1
                )

            elif action.action == ActionType.FULFILL:
                if robot.position[1] != self.world.fulfillment_y:
                    raise SimulationError(
                        f"Robot {action.robot_id} cannot fulfill from {robot.position}; "
                        f"robot must be on y={self.world.fulfillment_y}"
                    )

                storage = Counter(robot.storage)
                matching_order_id = None
                for order_id in sorted(self.world.orders):
                    order = self.world.orders[order_id]
                    if order.fulfilled or order_id in reserved_order_ids:
                        continue
                    if Counter(order.skus) == storage:
                        matching_order_id = order_id
                        break

                if matching_order_id is None:
                    raise SimulationError(
                        f"Robot {action.robot_id} storage does not exactly match "
                        "an available unfulfilled order"
                    )

                reserved_order_ids.add(matching_order_id)
                fulfillment_orders[action.robot_id] = matching_order_id

            else:
                raise SimulationError(
                    f"Action type '{action.action.value}' is not supported yet"
                )

        # Multiple robots may pick one pallet in the same timestep, but their
        # combined picks cannot exceed the stock that existed at timestep start.
        for pallet_id, requested_picks in picks_per_pallet.items():
            pallet = self.world.pallets[pallet_id]
            if requested_picks > pallet.count:
                raise SimulationError(
                    f"Pallet {pallet_id} has {pallet.count} items but "
                    f"{requested_picks} picks were requested"
                )

        # Only mutate the world after the entire timestep has been accepted.
        for robot_id, target in proposed_positions.items():
            self.world.robots[robot_id].position = target

        for robot_id, pallet_id in pick_requests.items():
            pallet = self.world.pallets[pallet_id]
            pallet.count -= 1
            self.world.robots[robot_id].storage.append(pallet.sku)

        for robot_id, order_id in fulfillment_orders.items():
            self.world.orders[order_id].fulfilled = True
            self.world.robots[robot_id].storage.clear()

        try:
            self.world.validate()
        except ValueError as error:
            raise SimulationError(
                f"Timestep produced an invalid world: {error}"
            ) from error

        self.world.timestep += 1

    def run(self, actions: Iterable[Action]) -> int:
        """Run a complete action schedule.

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
