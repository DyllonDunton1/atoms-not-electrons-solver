"""Unit tests for warehouse world state queries and validation."""

import unittest

from src.models import Pallet, ProblemInstance, Robot
from src.world import WorldState


DEFAULT_CAPACITIES = [5, 3, 8]


def make_problem(
    robots=None,
    pallets=None,
    orders=None,
    sku_capacities=None,
):
    """Create a synthetic problem for world-state tests."""
    return ProblemInstance(
        robots=[] if robots is None else robots,
        sku_capacities=(
            DEFAULT_CAPACITIES.copy()
            if sku_capacities is None
            else sku_capacities
        ),
        pallets=[] if pallets is None else pallets,
        orders=[] if orders is None else orders,
    )


def make_pallet(
    pallet_id,
    position,
    sku=0,
    count=None,
    max_count=None,
    original_position=None,
):
    """Create a pallet using the default synthetic SKU capacities."""
    expected_capacity = DEFAULT_CAPACITIES[sku]
    capacity = expected_capacity if max_count is None else max_count
    current_count = capacity if count is None else count
    start_position = position if original_position is None else original_position

    return Pallet(
        pallet_id=pallet_id,
        position=position,
        sku=sku,
        count=current_count,
        max_count=capacity,
        original_position=start_position,
    )


def make_busy_valid_world():
    """Create a larger valid world with varied inventory and edge positions."""
    robots = [
        Robot(robot_id=0, position=(0, 0)),
        Robot(robot_id=1, position=(59, 39)),
        Robot(robot_id=2, position=(20, 20)),
        Robot(robot_id=3, position=(40, 5)),
        Robot(robot_id=4, position=(7, 35)),
    ]

    pallets = [
        make_pallet(0, (1, 0), sku=0, count=0),
        make_pallet(1, (58, 39), sku=1, count=1),
        make_pallet(2, (20, 21), sku=2, count=8),
        make_pallet(3, (40, 6), sku=0, count=2),
        make_pallet(4, (7, 34), sku=1, count=3),
        make_pallet(5, (10, 10), sku=2, count=4),
        make_pallet(6, (11, 10), sku=2, count=7),
        make_pallet(7, (12, 10), sku=0, count=5),
        make_pallet(8, (30, 30), sku=1, count=0),
        make_pallet(9, (31, 30), sku=0, count=1),
        make_pallet(10, (32, 30), sku=2, count=3),
        make_pallet(11, (59, 0), sku=0, count=4),
    ]

    return WorldState(make_problem(robots=robots, pallets=pallets))


class TestWorldQueries(unittest.TestCase):
    def setUp(self):
        problem = make_problem(
            robots=[
                Robot(robot_id=0, position=(2, 2)),
                Robot(robot_id=1, position=(10, 10)),
            ],
            pallets=[
                make_pallet(0, (3, 2), sku=0),
                make_pallet(1, (7, 7), sku=1),
                make_pallet(2, (8, 7), sku=1),
            ],
        )
        self.world = WorldState(problem)

    def test_in_bounds(self):
        self.assertTrue(self.world.in_bounds((0, 0)))
        self.assertTrue(self.world.in_bounds((59, 39)))
        self.assertFalse(self.world.in_bounds((-1, 0)))
        self.assertFalse(self.world.in_bounds((60, 0)))
        self.assertFalse(self.world.in_bounds((0, 40)))

    def test_adjacent_positions_in_middle(self):
        self.assertEqual(
            set(self.world.adjacent_positions((5, 5))),
            {(4, 5), (6, 5), (5, 4), (5, 6)},
        )

    def test_adjacent_positions_at_all_corners(self):
        corner_cases = {
            (0, 0): {(1, 0), (0, 1)},
            (59, 0): {(58, 0), (59, 1)},
            (0, 39): {(1, 39), (0, 38)},
            (59, 39): {(58, 39), (59, 38)},
        }

        for corner, expected in corner_cases.items():
            with self.subTest(corner=corner):
                self.assertEqual(
                    set(self.world.adjacent_positions(corner)),
                    expected,
                )

    def test_occupied_positions(self):
        self.assertEqual(
            self.world.occupied_positions(),
            {(2, 2), (10, 10), (3, 2), (7, 7), (8, 7)},
        )

    def test_occupied_positions_on_busy_world(self):
        world = make_busy_valid_world()

        expected = {
            (0, 0),
            (59, 39),
            (20, 20),
            (40, 5),
            (7, 35),
            (1, 0),
            (58, 39),
            (20, 21),
            (40, 6),
            (7, 34),
            (10, 10),
            (11, 10),
            (12, 10),
            (30, 30),
            (31, 30),
            (32, 30),
            (59, 0),
        }

        self.assertEqual(world.occupied_positions(), expected)
        self.assertEqual(len(world.occupied_positions()), 17)

    def test_occupied_positions_reflect_current_state(self):
        world = make_busy_valid_world()

        world.robots[2].position = (21, 20)
        world.pallets[5].position = (10, 11)

        occupied = world.occupied_positions()
        self.assertIn((21, 20), occupied)
        self.assertIn((10, 11), occupied)
        self.assertNotIn((20, 20), occupied)
        self.assertNotIn((10, 10), occupied)

    def test_entity_at(self):
        self.assertIs(self.world.entity_at((2, 2)), self.world.robots[0])
        self.assertIs(self.world.entity_at((7, 7)), self.world.pallets[1])
        self.assertIsNone(self.world.entity_at((20, 20)))

    def test_entity_at_in_busy_world(self):
        world = make_busy_valid_world()

        self.assertIs(world.entity_at((59, 39)), world.robots[1])
        self.assertIs(world.entity_at((32, 30)), world.pallets[10])
        self.assertIsNone(world.entity_at((15, 15)))

    def test_pallets_for_sku(self):
        pallets = self.world.pallets_for_sku(1)
        self.assertEqual([pallet.pallet_id for pallet in pallets], [1, 2])
        self.assertEqual(self.world.pallets_for_sku(2), [])

    def test_pallets_for_sku_in_busy_world(self):
        world = make_busy_valid_world()

        sku_0_ids = [pallet.pallet_id for pallet in world.pallets_for_sku(0)]
        sku_1_ids = [pallet.pallet_id for pallet in world.pallets_for_sku(1)]
        sku_2_ids = [pallet.pallet_id for pallet in world.pallets_for_sku(2)]

        self.assertEqual(sku_0_ids, [0, 3, 7, 9, 11])
        self.assertEqual(sku_1_ids, [1, 4, 8])
        self.assertEqual(sku_2_ids, [2, 5, 6, 10])


class TestWorldValidation(unittest.TestCase):
    def test_multiple_valid_worlds_pass(self):
        worlds = {
            "empty world": WorldState(make_problem()),
            "single entity world": WorldState(
                make_problem(robots=[Robot(robot_id=0, position=(30, 20))])
            ),
            "busy world": make_busy_valid_world(),
            "zero inventory world": WorldState(
                make_problem(
                    robots=[Robot(robot_id=0, position=(5, 5))],
                    pallets=[make_pallet(0, (6, 5), sku=2, count=0)],
                )
            ),
        }

        for name, world in worlds.items():
            with self.subTest(world=name):
                world.validate()

    def test_rejects_out_of_bounds_robot(self):
        invalid_positions = [(-1, 5), (60, 5), (5, -1), (5, 40)]

        for position in invalid_positions:
            with self.subTest(position=position):
                world = WorldState(
                    make_problem(robots=[Robot(robot_id=0, position=position)])
                )
                with self.assertRaisesRegex(ValueError, "out of bounds"):
                    world.validate()

    def test_rejects_out_of_bounds_pallet(self):
        invalid_positions = [(-1, 5), (60, 5), (5, -1), (5, 40)]

        for position in invalid_positions:
            with self.subTest(position=position):
                world = WorldState(
                    make_problem(pallets=[make_pallet(0, position, sku=0)])
                )
                with self.assertRaisesRegex(ValueError, "out of bounds"):
                    world.validate()

    def test_rejects_out_of_bounds_original_pallet_position(self):
        pallet = make_pallet(
            0,
            (5, 5),
            sku=0,
            original_position=(-1, 5),
        )
        world = WorldState(make_problem(pallets=[pallet]))

        with self.assertRaisesRegex(ValueError, "original position"):
            world.validate()

    def test_rejects_robot_robot_overlap(self):
        world = WorldState(
            make_problem(
                robots=[
                    Robot(robot_id=0, position=(5, 5)),
                    Robot(robot_id=1, position=(5, 5)),
                ]
            )
        )

        with self.assertRaisesRegex(ValueError, "overlaps"):
            world.validate()

    def test_rejects_robot_pallet_overlap(self):
        world = WorldState(
            make_problem(
                robots=[Robot(robot_id=0, position=(5, 5))],
                pallets=[make_pallet(0, (5, 5), sku=0)],
            )
        )

        with self.assertRaisesRegex(ValueError, "overlaps"):
            world.validate()

    def test_rejects_pallet_pallet_overlap(self):
        world = WorldState(
            make_problem(
                pallets=[
                    make_pallet(0, (5, 5), sku=0),
                    make_pallet(1, (5, 5), sku=1),
                ]
            )
        )

        with self.assertRaisesRegex(ValueError, "overlaps"):
            world.validate()

    def test_rejects_invalid_inventory_counts(self):
        invalid_counts = [-1, 6, 100]

        for count in invalid_counts:
            with self.subTest(count=count):
                world = WorldState(
                    make_problem(
                        pallets=[make_pallet(0, (5, 5), sku=0, count=count)]
                    )
                )
                with self.assertRaisesRegex(ValueError, "invalid count"):
                    world.validate()

    def test_accepts_inventory_boundaries(self):
        for count in [0, DEFAULT_CAPACITIES[0]]:
            with self.subTest(count=count):
                world = WorldState(
                    make_problem(
                        pallets=[make_pallet(0, (5, 5), sku=0, count=count)]
                    )
                )
                world.validate()

    def test_rejects_invalid_sku(self):
        invalid_skus = [-1, 3, 99]

        for sku in invalid_skus:
            with self.subTest(sku=sku):
                pallet = make_pallet(0, (5, 5), sku=0)
                pallet.sku = sku
                world = WorldState(make_problem(pallets=[pallet]))

                with self.assertRaisesRegex(ValueError, "invalid SKU"):
                    world.validate()

    def test_rejects_capacity_mismatch(self):
        for bad_capacity in [0, 4, 6, 100]:
            with self.subTest(max_count=bad_capacity):
                pallet = make_pallet(
                    0,
                    (5, 5),
                    sku=0,
                    count=0,
                    max_count=bad_capacity,
                )
                world = WorldState(make_problem(pallets=[pallet]))

                with self.assertRaisesRegex(ValueError, "does not match"):
                    world.validate()

    def test_rejects_robot_id_mismatch(self):
        world = WorldState(
            make_problem(robots=[Robot(robot_id=0, position=(5, 5))])
        )
        world.robots[0].robot_id = 7

        with self.assertRaisesRegex(ValueError, "does not match robot id"):
            world.validate()

    def test_rejects_pallet_id_mismatch(self):
        world = WorldState(
            make_problem(pallets=[make_pallet(0, (5, 5), sku=0)])
        )
        world.pallets[0].pallet_id = 7

        with self.assertRaisesRegex(ValueError, "does not match pallet id"):
            world.validate()

    def test_busy_world_becomes_invalid_after_bad_mutations(self):
        mutation_cases = {
            "robot leaves grid": lambda world: setattr(
                world.robots[3], "position", (60, 5)
            ),
            "robot enters occupied pallet cell": lambda world: setattr(
                world.robots[2], "position", world.pallets[2].position
            ),
            "pallet inventory becomes negative": lambda world: setattr(
                world.pallets[6], "count", -1
            ),
            "pallet inventory exceeds capacity": lambda world: setattr(
                world.pallets[4], "count", world.pallets[4].max_count + 1
            ),
        }

        for name, mutate in mutation_cases.items():
            with self.subTest(mutation=name):
                world = make_busy_valid_world()
                world.validate()
                mutate(world)

                with self.assertRaises(ValueError):
                    world.validate()


if __name__ == "__main__":
    unittest.main()
