"""Tests for local movement validation and simulation."""

import unittest

from src.models import Action, ActionType, Pallet, ProblemInstance, Robot
from src.simulator import SimulationError, Simulator
from src.world import WorldState


def make_pallet(pallet_id, position):
    """Create a full synthetic pallet used as a movement obstacle."""
    return Pallet(
        pallet_id=pallet_id,
        position=position,
        sku=0,
        count=10,
        max_count=10,
        original_position=position,
    )


def make_world(robots=None, pallets=None):
    """Create a small valid world for movement tests."""
    problem = ProblemInstance(
        robots=robots or [],
        sku_capacities=[10],
        pallets=pallets or [],
        orders=[],
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
                    Action(0, 0, ActionType.MOVE, (5, 6)),
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

    def test_rejects_non_move_action_for_now(self):
        world = make_world([Robot(0, (5, 5))])
        simulator = Simulator(world)

        with self.assertRaisesRegex(SimulationError, "movement-only simulator"):
            simulator.step([Action(0, 0, ActionType.PICK, (6, 5))])

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
