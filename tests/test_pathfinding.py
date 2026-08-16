"""Tests for shortest-path planning and robot footprints."""

from collections import deque
from random import Random
from typing import Iterable, Optional, Set
import unittest

from src.models import Pallet, Position, ProblemInstance
from src.pathfinding import PathPlanner
from src.world import WorldState


def make_world(pallets=None) -> WorldState:
    """Create a small synthetic world for pathfinding tests."""
    problem = ProblemInstance(
        robots=[],
        sku_capacities=[10],
        pallets=pallets or [],
        orders=[],
    )
    return WorldState(problem)


def make_pallet(pallet_id: int, position: Position) -> Pallet:
    """Create a full synthetic pallet used as a static obstacle."""
    return Pallet(
        pallet_id=pallet_id,
        position=position,
        sku=0,
        count=10,
        max_count=10,
        original_position=position,
    )


def bfs_distance(
    world: WorldState,
    start: Position,
    goal: Position,
    blocked: Iterable[Position] = (),
) -> Optional[int]:
    """Simple BFS reference used to verify A* optimality."""
    blocked_positions: Set[Position] = {
        pallet.position for pallet in world.pallets.values()
    }
    blocked_positions.update(blocked)

    if not world.in_bounds(start) or not world.in_bounds(goal):
        return None
    if start in blocked_positions or goal in blocked_positions:
        return None

    queue = deque([(start, 0)])
    visited = {start}

    while queue:
        current, distance = queue.popleft()
        if current == goal:
            return distance

        for neighbor in world.adjacent_positions(current):
            if neighbor in blocked_positions or neighbor in visited:
                continue
            visited.add(neighbor)
            queue.append((neighbor, distance + 1))

    return None


def assert_valid_path(
    test_case: unittest.TestCase,
    path,
    blocked=(),
) -> None:
    """Check that consecutive path cells are legal orthogonal moves."""
    blocked_positions = set(blocked)
    for position in path:
        test_case.assertNotIn(position, blocked_positions)

    for first, second in zip(path, path[1:]):
        distance = abs(first[0] - second[0]) + abs(first[1] - second[1])
        test_case.assertEqual(distance, 1)


class TestPathPlanner(unittest.TestCase):
    def setUp(self) -> None:
        self.world = make_world()
        self.planner = PathPlanner(self.world)

    def test_empty_grid_path_length_equals_manhattan_distance(self) -> None:
        start = (2, 3)
        goal = (17, 11)
        path = self.planner.find_path(start, goal)

        self.assertEqual(path[0], start)
        self.assertEqual(path[-1], goal)
        self.assertEqual(len(path) - 1, 23)
        assert_valid_path(self, path)

    def test_start_equal_to_goal(self) -> None:
        self.assertEqual(self.planner.find_path((8, 8), (8, 8)), [(8, 8)])

    def test_pallet_cell_is_blocked(self) -> None:
        pallet = make_pallet(0, (2, 1))
        world = make_world([pallet])
        planner = PathPlanner(world)

        path = planner.find_path((1, 1), (3, 1))

        self.assertEqual(len(path) - 1, 4)
        self.assertNotIn((2, 1), path)
        assert_valid_path(self, path, blocked={(2, 1)})

    def test_artificial_wall_forces_known_detour(self) -> None:
        wall = {(3, y) for y in range(5)}
        path = self.planner.find_path((1, 2), (5, 2), blocked=wall)

        self.assertEqual(len(path) - 1, 10)
        assert_valid_path(self, path, blocked=wall)

    def test_unreachable_goal_returns_empty_path(self) -> None:
        blocked = {(2, 1), (1, 2), (3, 2), (2, 3)}
        self.assertEqual(
            self.planner.find_path((2, 2), (8, 8), blocked=blocked),
            [],
        )

    def test_out_of_bounds_or_blocked_endpoints_return_empty_path(self) -> None:
        self.assertEqual(self.planner.find_path((-1, 0), (3, 3)), [])
        self.assertEqual(self.planner.find_path((3, 3), (60, 3)), [])
        self.assertEqual(
            self.planner.find_path((3, 3), (5, 5), blocked={(5, 5)}),
            [],
        )

    def test_random_small_maps_match_bfs_path_lengths(self) -> None:
        random = Random(20260816)

        for case_number in range(100):
            width = random.randint(5, 12)
            height = random.randint(5, 12)
            world = make_world()
            world.width = width
            world.height = height
            planner = PathPlanner(world)

            start = (random.randrange(width), random.randrange(height))
            goal = (random.randrange(width), random.randrange(height))
            while goal == start:
                goal = (random.randrange(width), random.randrange(height))

            blocked = set()
            for x in range(width):
                for y in range(height):
                    position = (x, y)
                    if position in (start, goal):
                        continue
                    if random.random() < 0.28:
                        blocked.add(position)

            expected_distance = bfs_distance(world, start, goal, blocked)
            path = planner.find_path(start, goal, blocked=blocked)

            if expected_distance is None:
                self.assertEqual(
                    path,
                    [],
                    msg=f"A* found a path on unreachable random case {case_number}",
                )
            else:
                self.assertTrue(
                    path,
                    msg=f"A* missed reachable random case {case_number}",
                )
                self.assertEqual(
                    len(path) - 1,
                    expected_distance,
                    msg=f"A* was non-optimal on random case {case_number}",
                )
                assert_valid_path(self, path, blocked=blocked)


if __name__ == "__main__":
    unittest.main()
