"""Tests for avoiding immediate re-entry into a finished aisle."""

import unittest

from src.aisle_solver import AisleAwareSolver
from src.aisles import AislePlanner
from src.models import Order, Pallet, ProblemInstance, Robot
from src.world import WorldState


def make_pallet(pallet_id, position, sku):
    return Pallet(
        pallet_id=pallet_id,
        position=position,
        sku=sku,
        count=10,
        max_count=10,
        original_position=position,
    )


def make_world(pallets, orders=None):
    max_sku = max((pallet.sku for pallet in pallets), default=-1)
    return WorldState(
        ProblemInstance(
            robots=[Robot(0, (1, 2))],
            sku_capacities=[10] * (max_sku + 1),
            pallets=pallets,
            orders=[] if orders is None else orders,
        )
    )


class TestPreviousAisleSelection(unittest.TestCase):
    def test_choose_plan_omits_excluded_aisle_from_candidate_search(self):
        world = make_world(
            [
                make_pallet(0, (3, 2), 0),
                make_pallet(1, (20, 2), 1),
            ]
        )
        planner = AislePlanner(world)
        near_aisle = planner.aisle_for_pallet(0)
        far_aisle = planner.aisle_for_pallet(1)

        normal = planner.choose_plan(
            (1, 2),
            {0: 1, 1: 1},
            congestion_by_aisle={},
        )
        alternate = planner.choose_plan(
            (1, 2),
            {0: 1, 1: 1},
            congestion_by_aisle={},
            excluded_aisle_ids=[near_aisle],
        )

        self.assertIsNotNone(normal)
        self.assertIsNotNone(alternate)
        self.assertEqual(normal.aisle_id, near_aisle)
        self.assertEqual(alternate.aisle_id, far_aisle)

    def test_solver_does_not_immediately_reenter_previous_aisle(self):
        world = make_world(
            [
                make_pallet(0, (3, 2), 0),
                make_pallet(1, (20, 2), 1),
            ],
            orders=[Order(0, [0, 1])],
        )
        solver = AisleAwareSolver(
            world,
            robot_ids=[0],
            order_ids=[0],
            max_timesteps=100,
        )
        solver._assign_free_robots()

        near_aisle = solver.aisle_planner.aisle_for_pallet(0)
        far_aisle = solver.aisle_planner.aisle_for_pallet(1)

        self.assertTrue(solver._select_new_aisle(0))
        self.assertEqual(solver.states[0].active_aisle_id, near_aisle)
        self.assertIsNone(solver.states[0].previous_aisle_id)

        solver._finish_active_aisle(0)
        self.assertIsNone(solver.states[0].active_aisle_id)
        self.assertEqual(solver.states[0].previous_aisle_id, near_aisle)

        self.assertTrue(solver._select_new_aisle(0))
        self.assertEqual(solver.states[0].active_aisle_id, far_aisle)

        solver._finish_active_aisle(0)
        self.assertEqual(solver.states[0].previous_aisle_id, far_aisle)

        self.assertTrue(solver._select_new_aisle(0))
        self.assertEqual(solver.states[0].active_aisle_id, near_aisle)

    def test_previous_aisle_is_allowed_when_it_is_the_only_useful_option(self):
        world = make_world(
            [make_pallet(0, (3, 2), 0)],
            orders=[Order(0, [0])],
        )
        solver = AisleAwareSolver(
            world,
            robot_ids=[0],
            order_ids=[0],
            max_timesteps=100,
        )
        solver._assign_free_robots()

        only_aisle = solver.aisle_planner.aisle_for_pallet(0)
        self.assertTrue(solver._select_new_aisle(0))
        self.assertEqual(solver.states[0].active_aisle_id, only_aisle)

        solver._finish_active_aisle(0)
        self.assertEqual(solver.states[0].previous_aisle_id, only_aisle)

        self.assertTrue(solver._select_new_aisle(0))
        self.assertEqual(solver.states[0].active_aisle_id, only_aisle)


if __name__ == "__main__":
    unittest.main()
