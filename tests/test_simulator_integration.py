"""Integration tests connecting BIG_ORDER pathfinding to local simulation."""

from pathlib import Path
import unittest

from src.models import Action, ActionType
from src.parser import parse_problem
from src.pathfinding import PathPlanner
from src.simulator import Simulator
from src.world import WorldState


BIG_ORDER_PATH = (
    Path(__file__).resolve().parents[1]
    / "source_material"
    / "BIG_ORDER.txt"
)


class TestPathfindingSimulationIntegration(unittest.TestCase):
    """Replay the same kind of route already accepted by the browser testbench."""

    def test_robot_zero_top_to_bottom_route(self):
        problem = parse_problem(BIG_ORDER_PATH)
        world = WorldState(problem)
        planner = PathPlanner(world)
        simulator = Simulator(world)

        robot_id = 0
        robot = world.robots[robot_id]
        start = robot.position
        top_goal = (start[0], world.fulfillment_y)
        bottom_goal = (start[0], world.replenishment_y)

        original_other_positions = {
            other_id: other.position
            for other_id, other in world.robots.items()
            if other_id != robot_id
        }
        blocked_robots = set(original_other_positions.values())

        path_to_top = planner.find_path(
            start,
            top_goal,
            blocked=blocked_robots,
        )
        self.assertTrue(path_to_top)

        path_to_bottom = planner.find_path(
            top_goal,
            bottom_goal,
            blocked=blocked_robots,
        )
        self.assertTrue(path_to_bottom)

        full_path = path_to_top + path_to_bottom[1:]
        actions = [
            Action(
                timestep=timestep,
                robot_id=robot_id,
                action=ActionType.MOVE,
                target=position,
            )
            for timestep, position in enumerate(full_path[1:])
        ]

        final_timestep = simulator.run(actions)

        self.assertEqual(world.robots[robot_id].position, bottom_goal)
        self.assertEqual(final_timestep, len(actions))
        self.assertEqual(world.timestep, len(actions))

        for other_id, expected_position in original_other_positions.items():
            with self.subTest(robot_id=other_id):
                self.assertEqual(
                    world.robots[other_id].position,
                    expected_position,
                )


if __name__ == "__main__":
    unittest.main()
