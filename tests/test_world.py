"""Unit tests for warehouse world state queries and validation."""

import unittest

from src.models import Order, Pallet, ProblemInstance, Robot
from src.world import WorldState


def make_problem(
    robots=None,
    pallets=None,
    orders=None,
    sku_capacities=None,
):
    """Create a small synthetic problem for world-state tests."""
    return ProblemInstance(
        robots=robots or [],
        sku_capacities=sku_capacities or [5, 3, 8],
        pallets=pallets or [],
        orders=orders or [],
    )


def make_pallet(
    pallet_id,
    position,
    sku=0,
    count=None,
    max_count=None,
):
    """Create a pallet whose capacity matches the default synthetic world."""
    capacities = [5, 3, 8]
    capacity = capacities[sku] if max_count is None else max_count
    current_count = capacity if count is None else count
    return Pallet(
        pallet_id=pallet_id,
        position=position,
        sku=sku,
        count=current_count,
        max_count=capacity,
        original_position=position,
    )


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

    def test_adjacent_positions_at_corner(self):
        self.assertEqual(
            set(self.world.adjacent_positions((0, 0))),
            {(1, 0), (0, 1)},
        )

    def test_occupied_positions(self):
        self.assertEqual(
            self.world.occupied_positions(),
            {(2, 2), (10, 10), (3, 2), (7, 7), (8, 7)},
        )

    def test_entity_at(self):
        self.assertIs(self.world.entity_at((2, 2)), self.world.robots[0])
        self.assertIs(self.world.entity_at((7, 7)), self.world.pallets[1])
        self.assertIsNone(self.world.entity_at((20, 20)))

    def test_pallets_for_sku(self):
        pallets = self.world.pallets_for_sku(1)
        self.assertEqual([pallet.pallet_id for pallet in pallets], [1, 2])
        self.assertEqual(self.world.pallets_for_sku(2), [])


class TestWorldValidation(unittest.TestCase):
    def test_valid_world_passes(self):
        world = WorldState(
            make_problem(
                robots=[Robot(robot_id=0, position=(1, 1))],
                pallets=[make_pallet(0, (2, 1), sku=0)],
            )
        )
        world.validate()

    def test_rejects_out_of_bounds_robot(self):
        world = WorldState(
            make_problem(robots=[Robot(robot_id=0, position=(60, 5))])
        )
        with self.assertRaisesRegex(ValueError, "out of bounds"):
            world.validate()

    def test_rejects_out_of_bounds_pallet(self):
        world = WorldState(
            make_problem(pallets=[make_pallet(0, (-1, 5), sku=0)])
        )
        with self.assertRaisesRegex(ValueError, "out of bounds"):
            world.validate()

    def test_rejects_overlapping_entities(self):
        world = WorldState(
            make_problem(
                robots=[Robot(robot_id=0, position=(5, 5))],
                pallets=[make_pallet(0, (5, 5), sku=0)],
            )
        )
        with self.assertRaisesRegex(ValueError, "overlaps"):
            world.validate()

    def test_rejects_negative_pallet_count(self):
        world = WorldState(
            make_problem(pallets=[make_pallet(0, (5, 5), sku=0, count=-1)])
        )
        with self.assertRaisesRegex(ValueError, "invalid count"):
            world.validate()

    def test_rejects_count_above_capacity(self):
        world = WorldState(
            make_problem(pallets=[make_pallet(0, (5, 5), sku=0, count=6)])
        )
        with self.assertRaisesRegex(ValueError, "invalid count"):
            world.validate()

    def test_rejects_invalid_sku(self):
        pallet = make_pallet(0, (5, 5), sku=0)
        pallet.sku = 99
        world = WorldState(make_problem(pallets=[pallet]))
        with self.assertRaisesRegex(ValueError, "invalid SKU"):
            world.validate()

    def test_rejects_capacity_mismatch(self):
        pallet = make_pallet(0, (5, 5), sku=0, max_count=7)
        world = WorldState(make_problem(pallets=[pallet]))
        with self.assertRaisesRegex(ValueError, "does not match"):
            world.validate()


if __name__ == "__main__":
    unittest.main()
