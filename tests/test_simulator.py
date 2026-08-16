"""Tests for local movement, picking, and fulfillment simulation."""

import unittest

from src.models import Action, ActionType, Order, Pallet, ProblemInstance, Robot
from src.simulator import SimulationError, Simulator
from src.world import WorldState


def make_pallet(pallet_id, position, sku=0, count=10):
    """Create a synthetic pallet used by simulator tests."""
    return Pallet(
        pallet_id=pallet_id,
        position=position,
        sku=sku,
        count=count,
        max_count=10,
        original_position=position,
    )


def make_world(robots=None, pallets=None, orders=None):
    """Create a small valid world for simulator tests."""
    problem = ProblemInstance(
        robots=robots or [],
        sku_capacities=[10] * 10,
        pallets=pallets or [],
        orders=orders or [],
    )
    return WorldState(problem)


class TestSimulatorStep(unittest.TestCase):
    def test_legal_single_move(self):
        world = make_world([Robot(0, (5, 5))])
        simulator = Simulator(world)

        simulator.step([Action(0, 0, ActionType.MOVE, (6, 5))])

        self.assertEqual(world.robots[0].position, (6, 5))
        self.assertEqual(world.timestep, 1)

    def test_multiple_robots_can_make_independent_moves(self):
        world = make_world(
            [
                Robot(0, (1, 1)),
                Robot(1, (10, 10)),
                Robot(2, (20, 20)),
            ]
        )
        simulator = Simulator(world)

        simulator.step(
            [
                Action(0, 0, ActionType.MOVE, (2, 1)),
                Action(0, 2, ActionType.MOVE, (20, 21)),
            ]
        )

        self.assertEqual(world.robots[0].position, (2, 1))
        self.assertEqual(world.robots[1].position, (10, 10))
        self.assertEqual(world.robots[2].position, (20, 21))
        self.assertEqual(world.timestep, 1)

    def test_empty_step_means_every_robot_waits(self):
        world = make_world(
            [Robot(0, (4, 4)), Robot(1, (8, 8))]
        )
        simulator = Simulator(world)

        simulator.step([])

        self.assertEqual(world.robots[0].position, (4, 4))
        self.assertEqual(world.robots[1].position, (8, 8))
        self.assertEqual(world.timestep, 1)

    def test_rejects_diagonal_move(self):
        world = make_world([Robot(0, (5, 5))])
        simulator = Simulator(world)

        with self.assertRaisesRegex(SimulationError, "one-cell orthogonal"):
            simulator.step([Action(0, 0, ActionType.MOVE, (6, 6))])

    def test_rejects_two_cell_jump(self):
        world = make_world([Robot(0, (5, 5))])
        simulator = Simulator(world)

        with self.assertRaisesRegex(SimulationError, "one-cell orthogonal"):
            simulator.step([Action(0, 0, ActionType.MOVE, (7, 5))])

    def test_rejects_zero_cell_move(self):
        world = make_world([Robot(0, (5, 5))])
        simulator = Simulator(world)

        with self.assertRaisesRegex(SimulationError, "one-cell orthogonal"):
            simulator.step([Action(0, 0, ActionType.MOVE, (5, 5))])

    def test_rejects_out_of_bounds_move(self):
        world = make_world([Robot(0, (0, 0))])
        simulator = Simulator(world)

        with self.assertRaisesRegex(SimulationError, "out of bounds"):
            simulator.step([Action(0, 0, ActionType.MOVE, (-1, 0))])

    def test_rejects_move_into_pallet(self):
        world = make_world(
            [Robot(0, (4, 5))],
            [make_pallet(0, (5, 5))],
        )
        simulator = Simulator(world)

        with self.assertRaisesRegex(SimulationError, "move into pallet"):
            simulator.step([Action(0, 0, ActionType.MOVE, (5, 5))])

    def test_rejects_move_into_stationary_robot(self):
        world = make_world(
            [Robot(0, (4, 5)), Robot(1, (5, 5))]
        )
        simulator = Simulator(world)

        with self.assertRaisesRegex(SimulationError, "move into robot 1"):
            simulator.step([Action(0, 0, ActionType.MOVE, (5, 5))])

    def test_rejects_move_into_cell_robot_is_leaving(self):
        """For now, destinations must be empty at timestep start."""
        world = make_world(
            [Robot(0, (4, 5)), Robot(1, (5, 5))]
        )
        simulator = Simulator(world)

        with self.assertRaisesRegex(SimulationError, "move into robot 1"):
            simulator.step(
                [
                    Action(0, 0, ActionType.MOVE, (5, 5)),
                    Action(0, 1, ActionType.MOVE, (6, 5)),
                ]
            )

        self.assertEqual(world.robots[0].position, (4, 5))
        self.assertEqual(world.robots[1].position, (5, 5))
        self.assertEqual(world.timestep, 0)

    def test_rejects_two_robots_targeting_same_empty_cell(self):
        world = make_world(
            [Robot(0, (1, 1)), Robot(1, (3, 1))]
        )
        simulator = Simulator(world)

        with self.assertRaisesRegex(SimulationError, "both target"):
            simulator.step(
                [
                    Action(0, 0, ActionType.MOVE, (2, 1)),
                    Action(0, 1, ActionType.MOVE, (2, 1)),
                ]
            )

    def test_rejects_multiple_actions_for_same_robot(self):
        world = make_world([Robot(0, (5, 5))])
        simulator = Simulator(world)

        with self.assertRaisesRegex(SimulationError, "multiple actions"):
            simulator.step(
                [
                    Action(0, 0, ActionType.MOVE, (6, 5)),
                    Action(0, 0, ActionType.PICK, (5, 6)),
                ]
            )

    def test_rejects_wrong_timestep(self):
        world = make_world([Robot(0, (5, 5))])
        simulator = Simulator(world)

        with self.assertRaisesRegex(SimulationError, "expected 0"):
            simulator.step([Action(1, 0, ActionType.MOVE, (6, 5))])

    def test_rejects_unknown_robot(self):
        world = make_world([Robot(0, (5, 5))])
        simulator = Simulator(world)

        with self.assertRaisesRegex(SimulationError, "Unknown robot id 9"):
            simulator.step([Action(0, 9, ActionType.MOVE, (6, 5))])

    def test_rejects_docking_actions_for_now(self):
        world = make_world([Robot(0, (5, 5))])
        simulator = Simulator(world)

        with self.assertRaisesRegex(SimulationError, "not supported yet"):
            simulator.step([Action(0, 0, ActionType.DOCK, (6, 5))])

    def test_failed_step_does_not_partially_move_robots(self):
        world = make_world(
            [Robot(0, (1, 1)), Robot(1, (10, 10))]
        )
        simulator = Simulator(world)

        with self.assertRaises(SimulationError):
            simulator.step(
                [
                    Action(0, 0, ActionType.MOVE, (2, 1)),
                    Action(0, 1, ActionType.MOVE, (12, 10)),
                ]
            )

        self.assertEqual(world.robots[0].position, (1, 1))
        self.assertEqual(world.robots[1].position, (10, 10))
        self.assertEqual(world.timestep, 0)


class TestPicking(unittest.TestCase):
    def test_pick_decrements_stock_and_adds_sku_to_storage(self):
        pallet = make_pallet(0, (6, 5), sku=4, count=2)
        world = make_world([Robot(0, (5, 5))], [pallet])
        simulator = Simulator(world)

        simulator.step([Action(0, 0, ActionType.PICK, (6, 5))])

        self.assertEqual(world.pallets[0].count, 1)
        self.assertEqual(world.robots[0].storage, [4])
        self.assertEqual(world.robots[0].position, (5, 5))

    def test_rejects_non_adjacent_pick(self):
        pallet = make_pallet(0, (8, 5), sku=4)
        world = make_world([Robot(0, (5, 5))], [pallet])
        simulator = Simulator(world)

        with self.assertRaisesRegex(SimulationError, "not adjacent"):
            simulator.step([Action(0, 0, ActionType.PICK, (8, 5))])

        self.assertEqual(world.pallets[0].count, 10)
        self.assertEqual(world.robots[0].storage, [])
        self.assertEqual(world.timestep, 0)

    def test_rejects_pick_when_target_has_no_pallet(self):
        world = make_world([Robot(0, (5, 5))])
        simulator = Simulator(world)

        with self.assertRaisesRegex(SimulationError, "no pallet is there"):
            simulator.step([Action(0, 0, ActionType.PICK, (6, 5))])

    def test_rejects_empty_pallet_pick(self):
        pallet = make_pallet(0, (6, 5), sku=4, count=0)
        world = make_world([Robot(0, (5, 5))], [pallet])
        simulator = Simulator(world)

        with self.assertRaisesRegex(SimulationError, "0 items"):
            simulator.step([Action(0, 0, ActionType.PICK, (6, 5))])

        self.assertEqual(world.pallets[0].count, 0)
        self.assertEqual(world.robots[0].storage, [])
        self.assertEqual(world.timestep, 0)

    def test_multiple_robots_can_pick_same_pallet_when_stock_is_sufficient(self):
        pallet = make_pallet(0, (5, 5), sku=4, count=2)
        world = make_world(
            [Robot(0, (4, 5)), Robot(1, (5, 4))],
            [pallet],
        )
        simulator = Simulator(world)

        simulator.step(
            [
                Action(0, 0, ActionType.PICK, (5, 5)),
                Action(0, 1, ActionType.PICK, (5, 5)),
            ]
        )

        self.assertEqual(world.pallets[0].count, 0)
        self.assertEqual(world.robots[0].storage, [4])
        self.assertEqual(world.robots[1].storage, [4])

    def test_rejects_combined_picks_that_exceed_stock_without_partial_mutation(self):
        pallet = make_pallet(0, (5, 5), sku=4, count=1)
        world = make_world(
            [Robot(0, (4, 5)), Robot(1, (5, 4))],
            [pallet],
        )
        simulator = Simulator(world)

        with self.assertRaisesRegex(SimulationError, "1 items but 2 picks"):
            simulator.step(
                [
                    Action(0, 0, ActionType.PICK, (5, 5)),
                    Action(0, 1, ActionType.PICK, (5, 5)),
                ]
            )

        self.assertEqual(world.pallets[0].count, 1)
        self.assertEqual(world.robots[0].storage, [])
        self.assertEqual(world.robots[1].storage, [])
        self.assertEqual(world.timestep, 0)


class TestFulfillment(unittest.TestCase):
    def test_exact_match_fulfills_order_and_clears_storage(self):
        robot = Robot(0, (12, 0), storage=[7, 4, 4])
        order = Order(0, [4, 4, 7])
        world = make_world([robot], orders=[order])
        simulator = Simulator(world)

        # Challenge rules ignore the coordinates on a fulfill action.
        simulator.step([Action(0, 0, ActionType.FULFILL, (99, 99))])

        self.assertTrue(world.orders[0].fulfilled)
        self.assertEqual(world.robots[0].storage, [])
        self.assertEqual(world.robots[0].position, (12, 0))

    def test_rejects_fulfillment_when_one_item_is_missing(self):
        robot = Robot(0, (12, 0), storage=[4, 7])
        order = Order(0, [4, 4, 7])
        world = make_world([robot], orders=[order])
        simulator = Simulator(world)

        with self.assertRaisesRegex(SimulationError, "does not exactly match"):
            simulator.step([Action(0, 0, ActionType.FULFILL, (0, 0))])

        self.assertFalse(world.orders[0].fulfilled)
        self.assertEqual(world.robots[0].storage, [4, 7])

    def test_rejects_fulfillment_when_one_extra_item_exists(self):
        robot = Robot(0, (12, 0), storage=[4, 4, 7, 9])
        order = Order(0, [4, 4, 7])
        world = make_world([robot], orders=[order])
        simulator = Simulator(world)

        with self.assertRaisesRegex(SimulationError, "does not exactly match"):
            simulator.step([Action(0, 0, ActionType.FULFILL, (0, 0))])

        self.assertFalse(world.orders[0].fulfilled)
        self.assertEqual(world.robots[0].storage, [4, 4, 7, 9])

    def test_rejects_fulfillment_away_from_top_row(self):
        robot = Robot(0, (12, 1), storage=[4, 4, 7])
        order = Order(0, [4, 4, 7])
        world = make_world([robot], orders=[order])
        simulator = Simulator(world)

        with self.assertRaisesRegex(SimulationError, "must be on y=0"):
            simulator.step([Action(0, 0, ActionType.FULFILL, (0, 0))])

    def test_fulfillment_ignores_already_fulfilled_order(self):
        robot = Robot(0, (12, 0), storage=[4, 4, 7])
        order = Order(0, [4, 4, 7], fulfilled=True)
        world = make_world([robot], orders=[order])
        simulator = Simulator(world)

        with self.assertRaisesRegex(SimulationError, "available unfulfilled order"):
            simulator.step([Action(0, 0, ActionType.FULFILL, (0, 0))])

    def test_two_robots_cannot_claim_the_same_single_order(self):
        robots = [
            Robot(0, (10, 0), storage=[4, 4, 7]),
            Robot(1, (20, 0), storage=[7, 4, 4]),
        ]
        order = Order(0, [4, 4, 7])
        world = make_world(robots, orders=[order])
        simulator = Simulator(world)

        with self.assertRaisesRegex(SimulationError, "available unfulfilled order"):
            simulator.step(
                [
                    Action(0, 0, ActionType.FULFILL, (0, 0)),
                    Action(0, 1, ActionType.FULFILL, (0, 0)),
                ]
            )

        self.assertFalse(world.orders[0].fulfilled)
        self.assertEqual(world.robots[0].storage, [4, 4, 7])
        self.assertEqual(world.robots[1].storage, [7, 4, 4])
        self.assertEqual(world.timestep, 0)

    def test_two_identical_orders_can_be_fulfilled_simultaneously(self):
        robots = [
            Robot(0, (10, 0), storage=[4, 4, 7]),
            Robot(1, (20, 0), storage=[7, 4, 4]),
        ]
        orders = [Order(0, [4, 4, 7]), Order(1, [7, 4, 4])]
        world = make_world(robots, orders=orders)
        simulator = Simulator(world)

        simulator.step(
            [
                Action(0, 0, ActionType.FULFILL, (0, 0)),
                Action(0, 1, ActionType.FULFILL, (0, 0)),
            ]
        )

        self.assertTrue(world.orders[0].fulfilled)
        self.assertTrue(world.orders[1].fulfilled)
        self.assertEqual(world.robots[0].storage, [])
        self.assertEqual(world.robots[1].storage, [])


class TestSimulatorRun(unittest.TestCase):
    def test_run_orders_actions_by_timestep(self):
        world = make_world([Robot(0, (1, 1))])
        simulator = Simulator(world)

        final_timestep = simulator.run(
            [
                Action(1, 0, ActionType.MOVE, (3, 1)),
                Action(0, 0, ActionType.MOVE, (2, 1)),
            ]
        )

        self.assertEqual(world.robots[0].position, (3, 1))
        self.assertEqual(final_timestep, 2)
        self.assertEqual(world.timestep, 2)

    def test_run_inserts_wait_timesteps(self):
        world = make_world([Robot(0, (1, 1))])
        simulator = Simulator(world)

        final_timestep = simulator.run(
            [Action(2, 0, ActionType.MOVE, (2, 1))]
        )

        self.assertEqual(world.robots[0].position, (2, 1))
        self.assertEqual(final_timestep, 3)
        self.assertEqual(world.timestep, 3)

    def test_empty_run_does_not_advance_time(self):
        world = make_world([Robot(0, (1, 1))])
        simulator = Simulator(world)

        self.assertEqual(simulator.run([]), 0)
        self.assertEqual(world.robots[0].position, (1, 1))
        self.assertEqual(world.timestep, 0)


if __name__ == "__main__":
    unittest.main()
