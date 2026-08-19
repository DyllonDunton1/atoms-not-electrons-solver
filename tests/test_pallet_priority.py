"""Focused tests for pallet-aware one-step traffic priority."""

import unittest

from src.models import Order, Pallet, ProblemInstance, Robot
from src.multi_robot_solver import Intent, MultiRobotSolver
from src.world import WorldState


class TestPalletAwarePriority(unittest.TestCase):
    @staticmethod
    def _solver_with_returning_pallet():
        pallet = Pallet(
            22,
            (16, 9),
            0,
            179,
            179,
            (16, 9),
            4,
            (1, 0),
        )
        problem = ProblemInstance(
            robots=[
                Robot(2, (16, 7)),
                Robot(4, (15, 9), docked_pallets=[22]),
            ],
            sku_capacities=[179],
            pallets=[pallet],
            orders=[Order(0, []), Order(1, [])],
        )
        world = WorldState(problem)
        solver = MultiRobotSolver(
            world,
            robot_ids=[2, 4],
            order_ids=[0, 1],
            max_timesteps=50,
        )
        return world, solver

    def test_carried_pallet_overrides_robot_id_priority(self):
        world, solver = self._solver_with_returning_pallet()

        self.assertEqual(solver._priority_key(2), (0, 2))
        self.assertEqual(solver._priority_key(4), (-1, 4))
        self.assertEqual(solver._priority_order(), [4, 2])

        # The no-pallet robot must route around both the current and committed
        # rigid footprint of the pallet-carrying robot.
        blocked_for_r2 = solver._priority_blocked_cells(2, {4: (15, 8)})
        self.assertTrue(
            {(15, 9), (16, 9), (15, 8), (16, 8)} <= blocked_for_r2
        )

        # The pallet-carrying robot has priority and therefore does not shape
        # its preferred path around R2.
        blocked_for_r4 = solver._priority_blocked_cells(4, {})
        self.assertNotIn((16, 7), blocked_for_r4)

        actions = solver._plan_moves(
            {
                2: Intent(move_goal=(16, 23)),
                4: Intent(move_goal=(16, 8)),
            }
        )
        by_robot = {action.robot_id: action for action in actions}
        self.assertIn(4, by_robot)
        world.validate()

    def test_equal_pallet_counts_fall_back_to_robot_id(self):
        world, solver = self._solver_with_returning_pallet()
        # Removing R4's pallet returns the normal lower-ID-first ordering.
        world.robots[4].docked_pallets.clear()
        world.pallets[22].docked_to = None
        world.pallets[22].docked_offset = None
        self.assertEqual(solver._priority_order(), [2, 4])


if __name__ == "__main__":
    unittest.main()
