"""Local challenge simulator and action validator."""

from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable, List, Set, Tuple

from .models import Action, ActionType, Position
from .world import WorldState


EntityKey = Tuple[str, int]


class SimulationError(RuntimeError):
    """Raised when a generated action violates the challenge rules."""


class Simulator:
    """Execute solver actions against a local mutable world state."""

    def __init__(self, world: WorldState) -> None:
        self.world = world

    def _apply_replenishment(self) -> None:
        """Refill pallets docked to robots ending the timestep on y=39."""
        for robot in self.world.robots.values():
            if robot.position[1] != self.world.replenishment_y:
                continue

            for pallet_id in robot.docked_pallets:
                pallet = self.world.pallets[pallet_id]
                pallet.count = pallet.max_count

    def step(self, actions: Iterable[Action]) -> None:
        """Execute one timestep of supported challenge actions.

        Robots omitted from ``actions`` wait. The complete timestep is
        validated before any state is changed so an invalid action cannot
        partially apply the timestep. Automatic replenishment is applied after
        all robot action effects, matching the challenge's end-of-timestep rule.
        """
        actions_list = list(actions)

        try:
            self.world.validate()
        except ValueError as error:
            raise SimulationError(f"Cannot simulate invalid world: {error}") from error

        seen_robots: Set[int] = set()
        proposed_robot_positions: Dict[int, Position] = {}
        proposed_pallet_positions: Dict[int, Position] = {}
        proposed_cell_owners: Dict[Position, int] = {}
        pick_requests: Dict[int, int] = {}
        picks_per_pallet: Dict[int, int] = {}
        fulfillment_orders: Dict[int, int] = {}
        reserved_order_ids: Set[int] = set()
        dock_requests: Dict[int, int] = {}
        dock_offsets: Dict[int, Position] = {}
        reserved_dock_pallets: Set[int] = set()
        undock_requests: Dict[int, int] = {}

        pallet_by_position = {
            pallet.position: pallet for pallet in self.world.pallets.values()
        }
        entity_by_position: Dict[Position, EntityKey] = {}
        for robot in self.world.robots.values():
            entity_by_position[robot.position] = ("robot", robot.robot_id)
        for pallet in self.world.pallets.values():
            entity_by_position[pallet.position] = ("pallet", pallet.pallet_id)

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

                delta = (
                    target[0] - robot.position[0],
                    target[1] - robot.position[1],
                )
                distance = abs(delta[0]) + abs(delta[1])
                if distance != 1:
                    raise SimulationError(
                        f"Robot {action.robot_id} move from {robot.position} to {target} "
                        "is not a one-cell orthogonal move"
                    )

                own_entities: Set[EntityKey] = {("robot", action.robot_id)}
                footprint_targets: List[Tuple[EntityKey, Position]] = [
                    (("robot", action.robot_id), target)
                ]

                for pallet_id in robot.docked_pallets:
                    pallet = self.world.pallets[pallet_id]
                    own_entities.add(("pallet", pallet_id))
                    pallet_target = (
                        pallet.position[0] + delta[0],
                        pallet.position[1] + delta[1],
                    )
                    footprint_targets.append(
                        (("pallet", pallet_id), pallet_target)
                    )

                local_targets: Set[Position] = set()
                for entity_key, target_position in footprint_targets:
                    if not self.world.in_bounds(target_position):
                        raise SimulationError(
                            f"Robot {action.robot_id} docked footprint would leave "
                            f"the grid at {target_position}"
                        )

                    if target_position in local_targets:
                        raise SimulationError(
                            f"Robot {action.robot_id} docked footprint overlaps itself "
                            f"at {target_position}"
                        )
                    local_targets.add(target_position)

                    occupant = entity_by_position.get(target_position)
                    if occupant is not None and occupant not in own_entities:
                        occupant_type, occupant_id = occupant
                        if entity_key[0] == "robot":
                            if occupant_type == "pallet":
                                raise SimulationError(
                                    f"Robot {action.robot_id} cannot move into pallet "
                                    f"{occupant_id} at {target_position}"
                                )
                            raise SimulationError(
                                f"Robot {action.robot_id} cannot move into robot "
                                f"{occupant_id} at {target_position}"
                            )

                        raise SimulationError(
                            f"Robot {action.robot_id} docked pallet {entity_key[1]} "
                            f"would collide with {occupant_type} {occupant_id} at "
                            f"{target_position}"
                        )

                    other_robot = proposed_cell_owners.get(target_position)
                    if (
                        other_robot is not None
                        and other_robot != action.robot_id
                    ):
                        raise SimulationError(
                            f"Robots {other_robot} and {action.robot_id} both target "
                            f"{target_position} at timestep {self.world.timestep}"
                        )

                for entity_key, target_position in footprint_targets:
                    proposed_cell_owners[target_position] = action.robot_id
                    if entity_key[0] == "robot":
                        proposed_robot_positions[action.robot_id] = target_position
                    else:
                        proposed_pallet_positions[entity_key[1]] = target_position

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

            elif action.action == ActionType.DOCK:
                pallet = pallet_by_position.get(action.target)
                if pallet is None:
                    raise SimulationError(
                        f"Robot {action.robot_id} cannot dock at {action.target}; "
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

                if pallet.docked_to is not None:
                    raise SimulationError(
                        f"Pallet {pallet.pallet_id} is already docked to robot "
                        f"{pallet.docked_to}"
                    )

                if pallet.pallet_id in reserved_dock_pallets:
                    raise SimulationError(
                        f"Pallet {pallet.pallet_id} is requested by multiple robots "
                        f"at timestep {self.world.timestep}"
                    )

                if len(robot.docked_pallets) >= 4:
                    raise SimulationError(
                        f"Robot {action.robot_id} already has four docked pallets"
                    )

                offset = (
                    action.target[0] - robot.position[0],
                    action.target[1] - robot.position[1],
                )
                occupied_offsets = {
                    self.world.pallets[pallet_id].docked_offset
                    for pallet_id in robot.docked_pallets
                }
                if offset in occupied_offsets:
                    raise SimulationError(
                        f"Robot {action.robot_id} already has a pallet docked on "
                        f"side {offset}"
                    )

                reserved_dock_pallets.add(pallet.pallet_id)
                dock_requests[action.robot_id] = pallet.pallet_id
                dock_offsets[action.robot_id] = offset

            elif action.action == ActionType.UNDOCK:
                pallet = pallet_by_position.get(action.target)
                if pallet is None:
                    raise SimulationError(
                        f"Robot {action.robot_id} cannot undock at {action.target}; "
                        "no pallet is there"
                    )

                if pallet.docked_to != action.robot_id:
                    raise SimulationError(
                        f"Pallet {pallet.pallet_id} is not docked to robot "
                        f"{action.robot_id}"
                    )

                undock_requests[action.robot_id] = pallet.pallet_id

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
        for robot_id, target in proposed_robot_positions.items():
            self.world.robots[robot_id].position = target

        for pallet_id, target in proposed_pallet_positions.items():
            self.world.pallets[pallet_id].position = target

        for robot_id, pallet_id in pick_requests.items():
            pallet = self.world.pallets[pallet_id]
            pallet.count -= 1
            self.world.robots[robot_id].storage.append(pallet.sku)

        for robot_id, pallet_id in dock_requests.items():
            robot = self.world.robots[robot_id]
            pallet = self.world.pallets[pallet_id]
            pallet.docked_to = robot_id
            pallet.docked_offset = dock_offsets[robot_id]
            robot.docked_pallets.append(pallet_id)

        for robot_id, pallet_id in undock_requests.items():
            robot = self.world.robots[robot_id]
            pallet = self.world.pallets[pallet_id]
            robot.docked_pallets.remove(pallet_id)
            pallet.docked_to = None
            pallet.docked_offset = None

        for robot_id, order_id in fulfillment_orders.items():
            self.world.orders[order_id].fulfilled = True
            self.world.robots[robot_id].storage.clear()

        # Replenishment is automatic and happens after all action effects. A
        # robot that ends the timestep on y=39 refills every pallet still
        # docked to it, regardless of the pallets' own y coordinates.
        self._apply_replenishment()

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
