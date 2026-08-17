"""Tests for docking, docked footprints, and pallet movement."""

import unittest

from src.models import Action, ActionType, Pallet, ProblemInstance, Robot
from src.pathfinding import PathPlanner
from src.simulator import SimulationError, Simulator
from src.world import WorldState


def make_pallet(pallet_id, position, sku=0):
    """Create a full synthetic pallet for docking tests."""
    return Pallet(
        pallet_id=pallet_id,
        position=position,
        sku=sku,
        count=10,
        max_count=10,
        original_position=position,
    )


def make_world(robots=None, pallets=None):
    """Create a small valid world for docking tests."""
    return WorldState(
        ProblemInstance(
            robots=robots or [],
            sku_capacities=[10],
            pallets=pallets or [],
            orders=[],
        )
    )


class TestDocking(unittest.TestCase):
    def test_can_dock_one_pallet_on_each_side(self):
        robot = Robot(0, (5, 5))
        pallets = [
            make_pallet(0, (4, 5)),
            make_pallet(1, (6, 5)),
            make_pallet(2, (5, 4)),
            make_pallet(3, (5, 6)),
        ]
        world = make_world([robot], pallets)
        simulator = Simulator(world)

        for timestep, pallet in enumerate(pallets):
            simulator.step(
                [
                    Action(
                        timestep,
                        0,
                        ActionType.DOCK,
                        pallet.position,
                    )
                ]
            )

        self.assertEqual(world.robots[0].docked_pallets, [0, 1, 2, 3])
        self.assertEqual(world.pallets[0].docked_offset, (-1, 0))
        self.assertEqual(world.pallets[1].docked_offset, (1, 0))
        self.assertEqual(world.pallets[2].docked_offset, (0, -1))
        self.assertEqual(world.pallets[3].docked_offset, (0, 1))
        for pallet in world.pallets.values():
            self.assertEqual(pallet.docked_to, 0)

        world.validate()

    def test_docked_pallet_moves_with_robot_and_preserves_offset(self):
        world = make_world(
            [Robot(0, (5, 5))],
            [make_pallet(0, (6, 5))],
        )
        simulator = Simulator(world)

        simulator.step([Action(0, 0, ActionType.DOCK, (6, 5))])
        simulator.step([Action(1, 0, ActionType.MOVE, (6, 5))])

        self.assertEqual(world.robots[0].position, (6, 5))
        self.assertEqual(world.pallets[0].position, (7, 5))
        self.assertEqual(world.pallets[0].docked_offset, (1, 0))
        self.assertEqual(world.pallets[0].docked_to, 0)

    def test_pallet_part_of_footprint_can_cause_collision(self):
        world = make_world(
            [Robot(0, (5, 5))],
            [
                make_pallet(0, (6, 5)),
                make_pallet(1, (7, 5)),
            ],
        )
        simulator = Simulator(world)

        simulator.step([Action(0, 0, ActionType.DOCK, (6, 5))])

        with self.assertRaisesRegex(
            SimulationError,
            "docked pallet 0 would collide with pallet 1",
        ):
            simulator.step([Action(1, 0, ActionType.MOVE, (6, 5))])

        self.assertEqual(world.robots[0].position, (5, 5))
        self.assertEqual(world.pallets[0].position, (6, 5))
        self.assertEqual(world.pallets[1].position, (7, 5))
        self.assertEqual(world.timestep, 1)

    def test_docked_footprint_cannot_leave_grid(self):
        world = make_world(
            [Robot(0, (1, 1))],
            [make_pallet(0, (0, 1))],
        )
        simulator = Simulator(world)

        simulator.step([Action(0, 0, ActionType.DOCK, (0, 1))])

        with self.assertRaisesRegex(SimulationError, "footprint would leave the grid"):
            simulator.step([Action(1, 0, ActionType.MOVE, (0, 1))])

        self.assertEqual(world.robots[0].position, (1, 1))
        self.assertEqual(world.pallets[0].position, (0, 1))
        self.assertEqual(world.timestep, 1)

    def test_undock_leaves_pallet_in_place(self):
        world = make_world(
            [Robot(0, (5, 5))],
            [make_pallet(0, (6, 5))],
        )
        simulator = Simulator(world)

        simulator.step([Action(0, 0, ActionType.DOCK, (6, 5))])
        simulator.step([Action(1, 0, ActionType.MOVE, (5, 4))])
        simulator.step([Action(2, 0, ActionType.UNDOCK, (6, 4))])

        self.assertEqual(world.robots[0].position, (5, 4))
        self.assertEqual(world.robots[0].docked_pallets, [])
        self.assertEqual(world.pallets[0].position, (6, 4))
        self.assertIsNone(world.pallets[0].docked_to)
        self.assertIsNone(world.pallets[0].docked_offset)
        world.validate()

    def test_rejects_non_adjacent_dock(self):
        world = make_world(
            [Robot(0, (5, 5))],
            [make_pallet(0, (8, 5))],
        )
        simulator = Simulator(world)

        with self.assertRaisesRegex(SimulationError, "not adjacent"):
            simulator.step([Action(0, 0, ActionType.DOCK, (8, 5))])

    def test_rejects_undocking_pallet_owned_by_another_robot(self):
        world = make_world(
            [Robot(0, (5, 5)), Robot(1, (7, 5))],
            [make_pallet(0, (6, 5))],
        )
        simulator = Simulator(world)

        simulator.step([Action(0, 0, ActionType.DOCK, (6, 5))])

        with self.assertRaisesRegex(SimulationError, "not docked to robot 1"):
            simulator.step([Action(1, 1, ActionType.UNDOCK, (6, 5))])


class TestDockedPathfinding(unittest.TestCase):
    def test_planner_builds_footprint_from_docked_offsets(self):
        world = make_world(
            [Robot(0, (5, 5))],
            [
                make_pallet(0, (4, 5)),
                make_pallet(1, (5, 4)),
            ],
        )
        simulator = Simulator(world)
        simulator.step([Action(0, 0, ActionType.DOCK, (4, 5))])
        simulator.step([Action(1, 0, ActionType.DOCK, (5, 4))])

        planner = PathPlanner(world)
        self.assertEqual(
            planner.footprint_for_robot(0),
            frozenset({(0, 0), (-1, 0), (0, -1)}),
        )

    def test_ignored_docked_pallet_home_still_blocks_robot_center(self):
        world = make_world(
            [Robot(0, (1, 2))],
            [make_pallet(0, (2, 2))],
        )
        world.width = 3
        world.height = 5
        simulator = Simulator(world)
        simulator.step([Action(0, 0, ActionType.DOCK, (2, 2))])
        planner = PathPlanner(world)

        path = planner.find_path(
            (1, 2),
            (2, 2),
            ignored_pallet_ids=world.robots[0].docked_pallets,
        )

        self.assertEqual(path, [])

    def test_pallet_part_of_planner_footprint_respects_obstacles(self):
        world = make_world(
            [Robot(0, (1, 2))],
            [
                make_pallet(0, (2, 2)),
                make_pallet(1, (2, 1)),
            ],
        )
        simulator = Simulator(world)
        simulator.step([Action(0, 0, ActionType.DOCK, (2, 2))])
        planner = PathPlanner(world)

        single_path = planner.find_path(
            (1, 2),
            (1, 1),
            ignored_pallet_ids=[0],
        )
        docked_path = planner.find_path(
            (1, 2),
            (1, 1),
            footprint=planner.footprint_for_robot(0),
            ignored_pallet_ids=[0],
        )

        self.assertEqual(single_path, [(1, 2), (1, 1)])
        self.assertEqual(docked_path, [])

    def test_planner_finds_clear_path_for_docked_robot(self):
        world = make_world(
            [Robot(0, (5, 5))],
            [make_pallet(0, (6, 5))],
        )
        simulator = Simulator(world)
        simulator.step([Action(0, 0, ActionType.DOCK, (6, 5))])
        planner = PathPlanner(world)

        path = planner.find_path(
            (5, 5),
            (5, 2),
            footprint=planner.footprint_for_robot(0),
            ignored_pallet_ids=[0],
        )

        self.assertEqual(path[0], (5, 5))
        self.assertEqual(path[-1], (5, 2))
        self.assertEqual(len(path) - 1, 3)


if __name__ == "__main__":
    unittest.main()
