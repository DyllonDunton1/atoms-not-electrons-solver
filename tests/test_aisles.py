"""Tests for shared aisle geometry and aisle-level collection planning."""

from pathlib import Path
import unittest

from src.aisles import AislePlanner, build_aisle_layout
from src.models import Pallet, ProblemInstance, Robot
from src.parser import parse_problem
from src.world import WorldState


REPO_ROOT = Path(__file__).resolve().parents[1]
BIG_ORDER_PATH = REPO_ROOT / "source_material" / "BIG_ORDER.txt"


def make_pallet(pallet_id, position, sku, count=10, max_count=10):
    return Pallet(
        pallet_id=pallet_id,
        position=position,
        sku=sku,
        count=count,
        max_count=max_count,
        original_position=position,
    )


def make_world(pallets, robot_position=(1, 1)):
    max_sku = max((pallet.sku for pallet in pallets), default=-1)
    return WorldState(
        ProblemInstance(
            robots=[Robot(0, robot_position)],
            sku_capacities=[10] * (max_sku + 1),
            pallets=pallets,
            orders=[],
        )
    )


class TestAisleLayout(unittest.TestCase):
    def test_connected_pallet_homes_form_deterministic_aisles(self):
        pallets = [
            make_pallet(0, (10, 10), 0),
            make_pallet(1, (11, 10), 1),
            make_pallet(2, (10, 11), 2),
            make_pallet(3, (30, 5), 3),
            make_pallet(4, (31, 5), 4),
        ]

        layout = build_aisle_layout(pallets)

        self.assertEqual(len(layout.aisles), 2)
        self.assertEqual(layout.pallet_to_aisle[3], 0)
        self.assertEqual(layout.pallet_to_aisle[4], 0)
        self.assertEqual(layout.pallet_to_aisle[0], 1)
        self.assertEqual(layout.pallet_to_aisle[1], 1)
        self.assertEqual(layout.pallet_to_aisle[2], 1)

    def test_big_order_has_twelve_twenty_pallet_aisles(self):
        problem = parse_problem(BIG_ORDER_PATH)

        layout = build_aisle_layout(problem.pallets)

        self.assertEqual(len(layout.aisles), 12)
        self.assertEqual(
            [len(aisle.pallet_ids) for aisle in layout.aisles],
            [20] * 12,
        )


class TestAislePlanner(unittest.TestCase):
    def test_quantity_can_outweigh_extra_entry_distance(self):
        world = make_world(
            [
                make_pallet(0, (3, 2), 0),
                make_pallet(1, (20, 2), 1),
            ],
            robot_position=(1, 2),
        )
        planner = AislePlanner(world)

        plan = planner.choose_plan(
            (1, 2),
            {0: 1, 1: 8},
            congestion_by_aisle={},
        )

        self.assertIsNotNone(plan)
        self.assertEqual(plan.aisle_id, planner.aisle_for_pallet(1))
        self.assertEqual(plan.useful_quantity, 8)

    def test_congestion_is_a_soft_aisle_penalty(self):
        world = make_world(
            [
                make_pallet(0, (5, 5), 0),
                make_pallet(1, (5, 9), 1),
            ],
            robot_position=(1, 7),
        )
        planner = AislePlanner(world)
        aisle_zero = planner.aisle_for_pallet(0)
        aisle_one = planner.aisle_for_pallet(1)

        plan = planner.choose_plan(
            (1, 7),
            {0: 4, 1: 4},
            congestion_by_aisle={aisle_zero: 2, aisle_one: 0},
        )

        self.assertIsNotNone(plan)
        self.assertEqual(plan.aisle_id, aisle_one)

    def test_plan_services_only_required_skus_in_selected_aisle(self):
        world = make_world(
            [
                make_pallet(0, (10, 10), 0),
                make_pallet(1, (11, 10), 1),
                make_pallet(2, (10, 11), 2),
                make_pallet(3, (11, 11), 3),
            ],
            robot_position=(8, 10),
        )
        planner = AislePlanner(world)
        aisle_id = planner.aisle_for_pallet(0)

        plan = planner.plan_aisle(
            aisle_id,
            (8, 10),
            {0: 3, 2: 2},
        )

        self.assertIsNotNone(plan)
        self.assertEqual({stop.sku for stop in plan.stops}, {0, 2})
        self.assertEqual(len(plan.stops), 2)
        self.assertEqual(plan.useful_quantity, 5)

    def test_stocked_duplicate_pallet_is_preferred_within_aisle(self):
        world = make_world(
            [
                make_pallet(0, (10, 10), 0, count=0),
                make_pallet(1, (11, 10), 0, count=10),
            ],
            robot_position=(8, 10),
        )
        planner = AislePlanner(world)
        aisle_id = planner.aisle_for_pallet(0)

        plan = planner.plan_aisle(
            aisle_id,
            (8, 10),
            {0: 4},
        )

        self.assertIsNotNone(plan)
        self.assertEqual(len(plan.stops), 1)
        self.assertEqual(plan.stops[0].pallet_id, 1)


if __name__ == "__main__":
    unittest.main()
