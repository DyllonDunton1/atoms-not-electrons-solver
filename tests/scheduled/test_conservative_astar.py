import unittest

from scheduled_solver.conservative_astar import ConservativeFastPathSpaceTimeAStar
from scheduled_solver.geometry import WarehouseGeometry
from scheduled_solver.reservations import ReservationTable


class ConservativeFastPathTests(unittest.TestCase):
    def make_geometry(self, blocked=()):
        return WarehouseGeometry(
            width=8,
            height=8,
            fulfillment_y=0,
            replenishment_y=7,
            static_blocked=frozenset(blocked),
            columns=(),
            pallet_to_column={},
        )

    def test_clear_straight_point_route_uses_fast_path(self):
        planner = ConservativeFastPathSpaceTimeAStar(
            self.make_geometry(), ReservationTable(0), path_horizon=20
        )
        path = planner.find_path((1, 1), 0, (4, 1), owner=0)

        self.assertEqual(
            path,
            [((1, 1), 0), ((2, 1), 1), ((3, 1), 2), ((4, 1), 3)],
        )
        self.assertEqual(planner.counters.fast_point_hits, 1)
        self.assertEqual(planner.counters.expansions, 0)

    def test_diagonal_point_route_preserves_astar_tie_breaking(self):
        planner = ConservativeFastPathSpaceTimeAStar(
            self.make_geometry(), ReservationTable(0), path_horizon=20
        )
        path = planner.find_path((1, 1), 0, (3, 3), owner=0)

        self.assertIsNotNone(path)
        self.assertEqual(path[-1][0], (3, 3))
        self.assertEqual(planner.counters.fast_point_hits, 0)
        self.assertGreater(planner.counters.expansions, 0)

    def test_blocked_straight_route_falls_back_to_astar_detour(self):
        planner = ConservativeFastPathSpaceTimeAStar(
            self.make_geometry({(2, 1)}), ReservationTable(0), path_horizon=20
        )
        path = planner.find_path((1, 1), 0, (3, 1), owner=0)

        self.assertIsNotNone(path)
        self.assertNotIn(((2, 1), 1), path)
        self.assertGreater(len(path) - 1, 2)
        self.assertEqual(planner.counters.fast_point_hits, 0)
        self.assertGreater(planner.counters.expansions, 0)

    def test_min_goal_time_wait_uses_astar_even_on_straight_route(self):
        planner = ConservativeFastPathSpaceTimeAStar(
            self.make_geometry(), ReservationTable(0), path_horizon=20
        )
        path = planner.find_path(
            (1, 1),
            0,
            (2, 1),
            owner=0,
            min_goal_time=4,
        )

        self.assertIsNotNone(path)
        self.assertEqual(path[-1], ((2, 1), 4))
        self.assertEqual(planner.counters.fast_point_hits, 0)
        self.assertGreater(planner.counters.expansions, 0)


if __name__ == "__main__":
    unittest.main()
