"""Integration tests for the aisle-aware five-robot solver."""

from pathlib import Path
import unittest

from src.aisle_solver import AisleAwareSolver
from src.models import ActionType, Order, Pallet, ProblemInstance, Robot
from src.parser import parse_problem
from src.simulator import Simulator
from src.world import WorldState


REPO_ROOT = Path(__file__).resolve().parents[1]
BIG_ORDER_PATH = REPO_ROOT / "source_material" / "BIG_ORDER.txt"


class TestAisleAwareSolver(unittest.TestCase):
    def _run_prefix(self, robot_ids, order_count):
        problem = parse_problem(BIG_ORDER_PATH)
        world = WorldState(problem)
        order_ids = list(range(order_count))
        solver = AisleAwareSolver(
            world,
            robot_ids=robot_ids,
            order_ids=order_ids,
            max_timesteps=20_000,
        )

        actions = solver.solve()

        self.assertEqual(solver.assigned_ids, set(order_ids))
        self.assertEqual(solver.completed_ids, set(order_ids))
        self.assertEqual(len(solver.queue), 0)
        self.assertTrue(all(world.orders[i].fulfilled for i in order_ids))
        self.assertEqual(solver.pallet_claims, {})
        world.validate()

        replay_world = WorldState(parse_problem(BIG_ORDER_PATH))
        Simulator(replay_world).run(actions)
        self.assertTrue(all(replay_world.orders[i].fulfilled for i in order_ids))
        replay_world.validate()
        return actions

    def test_one_robot_solves_first_three_orders(self):
        actions = self._run_prefix([0], 3)
        self.assertTrue(actions)

    def test_five_robots_solve_first_ten_orders(self):
        actions = self._run_prefix([0, 1, 2, 3, 4], 10)
        self.assertTrue(actions)

    def test_refill_returns_to_same_aisle_and_finishes_stop(self):
        pallet = Pallet(
            pallet_id=0,
            position=(6, 5),
            sku=0,
            count=1,
            max_count=2,
            original_position=(6, 5),
        )
        order = Order(order_id=0, skus=[0, 0, 0])
        problem = ProblemInstance(
            robots=[Robot(0, (5, 5))],
            sku_capacities=[2],
            pallets=[pallet],
            orders=[order],
        )
        world = WorldState(problem)
        solver = AisleAwareSolver(
            world,
            robot_ids=[0],
            order_ids=[0],
            max_timesteps=500,
        )

        actions = solver.solve()

        self.assertTrue(world.orders[0].fulfilled)
        self.assertEqual(
            sum(action.action == ActionType.DOCK for action in actions),
            1,
        )
        self.assertEqual(
            sum(action.action == ActionType.UNDOCK for action in actions),
            1,
        )
        self.assertEqual(
            sum(action.action == ActionType.PICK for action in actions),
            3,
        )

        replay_world = WorldState(
            ProblemInstance(
                robots=[Robot(0, (5, 5))],
                sku_capacities=[2],
                pallets=[
                    Pallet(
                        pallet_id=0,
                        position=(6, 5),
                        sku=0,
                        count=1,
                        max_count=2,
                        original_position=(6, 5),
                    )
                ],
                orders=[Order(order_id=0, skus=[0, 0, 0])],
            )
        )
        Simulator(replay_world).run(actions)
        self.assertTrue(replay_world.orders[0].fulfilled)


if __name__ == "__main__":
    unittest.main()
