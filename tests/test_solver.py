"""Tests for the Step 11 autonomous single-robot baseline solver."""

from pathlib import Path
import unittest

from src.models import ActionType, Order, Pallet, ProblemInstance, Robot
from src.parser import parse_problem
from src.solver import Solver
from src.world import WorldState


REPO_ROOT = Path(__file__).resolve().parents[1]
BIG_ORDER_PATH = REPO_ROOT / "source_material" / "BIG_ORDER.txt"


def make_pallet(pallet_id, position, sku, count, max_count=10):
    return Pallet(
        pallet_id=pallet_id,
        position=position,
        sku=sku,
        count=count,
        max_count=max_count,
        original_position=position,
    )


def make_world(robots, pallets, orders, capacities):
    return WorldState(
        ProblemInstance(
            robots=robots,
            sku_capacities=capacities,
            pallets=pallets,
            orders=orders,
        )
    )


class TestOneRobotSolverSynthetic(unittest.TestCase):
    def test_solves_one_order_with_repeated_skus(self):
        world = make_world(
            robots=[Robot(0, (5, 5))],
            pallets=[
                make_pallet(0, (6, 5), sku=0, count=10),
                make_pallet(1, (5, 8), sku=1, count=10),
            ],
            orders=[Order(0, [0, 0, 1, 0, 1])],
            capacities=[10, 10],
        )
        solver = Solver(world)

        actions = solver.solve_orders([0])

        self.assertTrue(world.orders[0].fulfilled)
        self.assertEqual(world.robots[0].storage, [])
        self.assertEqual(world.robots[0].position[1], 0)
        self.assertEqual(world.pallets[0].count, 7)
        self.assertEqual(world.pallets[1].count, 8)
        self.assertEqual(
            sum(action.action == ActionType.PICK for action in actions),
            5,
        )
        self.assertEqual(
            sum(action.action == ActionType.FULFILL for action in actions),
            1,
        )
        world.validate()

    def test_forced_replenishment_returns_pallet_home_and_completes_order(self):
        pallet = make_pallet(0, (6, 5), sku=0, count=2, max_count=10)
        world = make_world(
            robots=[Robot(0, (5, 5))],
            pallets=[pallet],
            orders=[Order(0, [0, 0, 0, 0, 0])],
            capacities=[10],
        )
        solver = Solver(world)

        actions = solver.solve_orders([0])

        action_types = [action.action for action in actions]
        self.assertIn(ActionType.DOCK, action_types)
        self.assertIn(ActionType.UNDOCK, action_types)
        self.assertTrue(world.orders[0].fulfilled)
        self.assertEqual(world.pallets[0].position, (6, 5))
        self.assertIsNone(world.pallets[0].docked_to)
        self.assertIsNone(world.pallets[0].docked_offset)
        self.assertEqual(world.robots[0].docked_pallets, [])

        # The pallet began with only two items, so fulfilling five items is
        # impossible unless the autonomous replenish-and-return branch ran.
        # It refills to 10, then five picks leave exactly five.
        self.assertEqual(world.pallets[0].count, 5)
        self.assertEqual(world.robots[0].storage, [])
        world.validate()

    def test_chooses_nearest_reachable_pallet_with_enough_stock(self):
        world = make_world(
            robots=[Robot(0, (5, 5))],
            pallets=[
                make_pallet(0, (6, 5), sku=0, count=10),
                make_pallet(1, (20, 20), sku=0, count=10),
            ],
            orders=[Order(0, [0, 0])],
            capacities=[10],
        )
        solver = Solver(world)

        actions = solver.solve_orders([0])
        pick_targets = [
            action.target
            for action in actions
            if action.action == ActionType.PICK
        ]

        self.assertEqual(pick_targets, [(6, 5), (6, 5)])
        self.assertEqual(world.pallets[0].count, 8)
        self.assertEqual(world.pallets[1].count, 10)

    def test_solves_multiple_orders_sequentially(self):
        world = make_world(
            robots=[Robot(0, (5, 5))],
            pallets=[
                make_pallet(0, (6, 5), sku=0, count=10),
                make_pallet(1, (8, 5), sku=1, count=10),
            ],
            orders=[
                Order(0, [0, 0, 1]),
                Order(1, [1, 1]),
                Order(2, [0, 1]),
            ],
            capacities=[10, 10],
        )
        solver = Solver(world)

        actions = solver.solve_orders([0, 1, 2])

        self.assertEqual(
            sum(order.fulfilled for order in world.orders.values()),
            3,
        )
        self.assertEqual(
            sum(action.action == ActionType.FULFILL for action in actions),
            3,
        )
        self.assertEqual(world.robots[0].storage, [])
        world.validate()


class TestOneRobotSolverBigOrder(unittest.TestCase):
    def test_solves_first_real_order(self):
        problem = parse_problem(BIG_ORDER_PATH)
        world = WorldState(problem)
        solver = Solver(world)

        actions = solver.solve_orders([0])

        self.assertTrue(world.orders[0].fulfilled)
        self.assertEqual(
            sum(order.fulfilled for order in world.orders.values()),
            1,
        )
        self.assertEqual(world.robots[0].storage, [])
        self.assertEqual(world.robots[0].docked_pallets, [])
        self.assertTrue(actions)
        world.validate()

    def test_solves_first_five_real_orders(self):
        problem = parse_problem(BIG_ORDER_PATH)
        world = WorldState(problem)
        solver = Solver(world)

        actions = solver.solve_orders(range(5))

        self.assertTrue(all(world.orders[order_id].fulfilled for order_id in range(5)))
        self.assertEqual(
            sum(order.fulfilled for order in world.orders.values()),
            5,
        )
        self.assertEqual(
            sum(action.action == ActionType.FULFILL for action in actions),
            5,
        )
        self.assertEqual(world.robots[0].storage, [])
        self.assertEqual(world.robots[0].docked_pallets, [])
        world.validate()


if __name__ == "__main__":
    unittest.main()
