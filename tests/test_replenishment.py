"""Tests for automatic end-of-timestep pallet replenishment."""

import unittest

from src.models import Action, ActionType, Pallet, ProblemInstance, Robot
from src.simulator import Simulator
from src.world import WorldState


def make_pallet(
    pallet_id,
    position,
    *,
    sku=0,
    count=1,
    max_count=10,
    docked_to=None,
    docked_offset=None,
):
    """Create a synthetic pallet for replenishment tests."""
    return Pallet(
        pallet_id=pallet_id,
        position=position,
        sku=sku,
        count=count,
        max_count=max_count,
        original_position=position,
        docked_to=docked_to,
        docked_offset=docked_offset,
    )


def make_world(robots, pallets, capacities=None):
    """Create a valid synthetic warehouse world."""
    problem = ProblemInstance(
        robots=robots,
        sku_capacities=capacities or [10],
        pallets=pallets,
        orders=[],
    )
    return WorldState(problem)


class TestReplenishment(unittest.TestCase):
    def test_move_onto_replenishment_row_refills_docked_pallet(self):
        robot = Robot(0, (5, 38), docked_pallets=[0])
        pallet = make_pallet(
            0,
            (6, 38),
            count=2,
            docked_to=0,
            docked_offset=(1, 0),
        )
        world = make_world([robot], [pallet])
        simulator = Simulator(world)

        simulator.step([Action(0, 0, ActionType.MOVE, (5, 39))])

        self.assertEqual(world.robots[0].position, (5, 39))
        self.assertEqual(world.pallets[0].position, (6, 39))
        self.assertEqual(world.pallets[0].count, 10)

    def test_undocked_pallet_does_not_refill(self):
        robot = Robot(0, (5, 39))
        pallet = make_pallet(0, (6, 39), count=2)
        world = make_world([robot], [pallet])
        simulator = Simulator(world)

        simulator.step([])

        self.assertEqual(world.pallets[0].count, 2)
        self.assertEqual(world.timestep, 1)

    def test_multiple_docked_pallets_refill_together(self):
        robot = Robot(0, (5, 39), docked_pallets=[0, 1, 2])
        pallets = [
            make_pallet(
                0,
                (4, 39),
                sku=0,
                count=1,
                max_count=10,
                docked_to=0,
                docked_offset=(-1, 0),
            ),
            make_pallet(
                1,
                (6, 39),
                sku=1,
                count=5,
                max_count=20,
                docked_to=0,
                docked_offset=(1, 0),
            ),
            make_pallet(
                2,
                (5, 38),
                sku=2,
                count=0,
                max_count=30,
                docked_to=0,
                docked_offset=(0, -1),
            ),
        ]
        world = make_world([robot], pallets, capacities=[10, 20, 30])
        simulator = Simulator(world)

        # Waiting still advances the timestep, so the automatic end-of-step
        # refill should happen even though the robot takes no explicit action.
        simulator.step([])

        self.assertEqual(world.pallets[0].count, 10)
        self.assertEqual(world.pallets[1].count, 20)
        self.assertEqual(world.pallets[2].count, 30)

    def test_pick_on_replenishment_timestep_happens_before_refill(self):
        robot = Robot(0, (5, 39), docked_pallets=[0])
        pallet = make_pallet(
            0,
            (6, 39),
            sku=0,
            count=2,
            docked_to=0,
            docked_offset=(1, 0),
        )
        world = make_world([robot], [pallet])
        simulator = Simulator(world)

        simulator.step([Action(0, 0, ActionType.PICK, (6, 39))])

        # The pick succeeds against the starting count of 2 and puts the item
        # into storage; then the automatic refill restores the pallet to 10.
        self.assertEqual(world.robots[0].storage, [0])
        self.assertEqual(world.pallets[0].count, 10)

    def test_docking_on_replenishment_row_refills_same_timestep(self):
        robot = Robot(0, (5, 39))
        pallet = make_pallet(0, (6, 39), count=3)
        world = make_world([robot], [pallet])
        simulator = Simulator(world)

        simulator.step([Action(0, 0, ActionType.DOCK, (6, 39))])

        self.assertEqual(world.robots[0].docked_pallets, [0])
        self.assertEqual(world.pallets[0].docked_to, 0)
        self.assertEqual(world.pallets[0].count, 10)

    def test_undocking_on_replenishment_row_prevents_refill(self):
        robot = Robot(0, (5, 39), docked_pallets=[0])
        pallet = make_pallet(
            0,
            (6, 39),
            count=3,
            docked_to=0,
            docked_offset=(1, 0),
        )
        world = make_world([robot], [pallet])
        simulator = Simulator(world)

        simulator.step([Action(0, 0, ActionType.UNDOCK, (6, 39))])

        self.assertEqual(world.robots[0].docked_pallets, [])
        self.assertIsNone(world.pallets[0].docked_to)
        self.assertEqual(world.pallets[0].count, 3)


if __name__ == "__main__":
    unittest.main()
