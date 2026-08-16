"""End-to-end simulator test for collecting and fulfilling one small order."""

import unittest

from src.models import Action, ActionType, Order, Pallet, ProblemInstance, Robot
from src.simulator import Simulator
from src.world import WorldState


class TestOrderSimulation(unittest.TestCase):
    def test_collect_repeated_skus_and_fulfill(self):
        """A robot can collect [4, 4, 7], reach y=0, and fulfill exactly."""
        problem = ProblemInstance(
            robots=[Robot(0, (2, 3))],
            sku_capacities=[10] * 10,
            pallets=[
                Pallet(
                    pallet_id=0,
                    position=(3, 3),
                    sku=4,
                    count=2,
                    max_count=10,
                    original_position=(3, 3),
                ),
                Pallet(
                    pallet_id=1,
                    position=(2, 5),
                    sku=7,
                    count=1,
                    max_count=10,
                    original_position=(2, 5),
                ),
            ],
            orders=[Order(0, [4, 4, 7])],
        )
        world = WorldState(problem)
        simulator = Simulator(world)

        actions = [
            Action(0, 0, ActionType.PICK, (3, 3)),
            Action(1, 0, ActionType.PICK, (3, 3)),
            Action(2, 0, ActionType.MOVE, (2, 4)),
            Action(3, 0, ActionType.PICK, (2, 5)),
            Action(4, 0, ActionType.MOVE, (2, 3)),
            Action(5, 0, ActionType.MOVE, (2, 2)),
            Action(6, 0, ActionType.MOVE, (2, 1)),
            Action(7, 0, ActionType.MOVE, (2, 0)),
            Action(8, 0, ActionType.FULFILL, (0, 0)),
        ]

        final_timestep = simulator.run(actions)

        self.assertEqual(world.pallets[0].count, 0)
        self.assertEqual(world.pallets[1].count, 0)
        self.assertEqual(world.robots[0].position, (2, 0))
        self.assertEqual(world.robots[0].storage, [])
        self.assertTrue(world.orders[0].fulfilled)
        self.assertEqual(final_timestep, 9)
        self.assertEqual(world.timestep, 9)


if __name__ == "__main__":
    unittest.main()
