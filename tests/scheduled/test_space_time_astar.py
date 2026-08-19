import unittest

from scheduled_solver.geometry import WarehouseGeometry
from scheduled_solver.reservations import ReservationTable
from scheduled_solver.space_time_astar import SpaceTimeAStar


class SpaceTimeAStarTests(unittest.TestCase):
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

    def test_direct_path_without_reservations(self):
        planner = SpaceTimeAStar(self.make_geometry(), ReservationTable(0), path_horizon=20)
        path = planner.find_path((1, 1), 0, (3, 1), owner=0)
        self.assertEqual(path, [((1, 1), 0), ((2, 1), 1), ((3, 1), 2)])

    def test_static_obstacle_forces_detour(self):
        planner = SpaceTimeAStar(
            self.make_geometry({(2, 1)}), ReservationTable(0), path_horizon=20
        )
        path = planner.find_path((1, 1), 0, (3, 1), owner=0)
        self.assertIsNotNone(path)
        self.assertNotIn(((2, 1), 1), path)
        self.assertGreater(len(path) - 1, 2)

    def test_vertex_reservation_can_force_wait(self):
        reservations = ReservationTable(0)
        reservations.reserve_pose([(2, 1)], 1, 9)
        planner = SpaceTimeAStar(self.make_geometry(), reservations, path_horizon=20)
        path = planner.find_path((1, 1), 0, (3, 1), owner=0)
        self.assertIsNotNone(path)
        self.assertNotEqual(path[1], ((2, 1), 1))

    def test_reverse_edge_swap_is_avoided(self):
        reservations = ReservationTable(0)
        reservations.reserve_edges([((2, 1), (1, 1))], 0, 9)
        planner = SpaceTimeAStar(self.make_geometry(), reservations, path_horizon=20)
        path = planner.find_path((1, 1), 0, (2, 1), owner=0)
        self.assertIsNotNone(path)
        self.assertGreater(path[-1][1], 1)

    def test_full_footprint_is_checked(self):
        geometry = self.make_geometry({(3, 1)})
        planner = SpaceTimeAStar(geometry, ReservationTable(0), path_horizon=20)
        footprint = frozenset({(0, 0), (1, 0)})
        path = planner.find_path((1, 1), 0, (2, 1), owner=0, footprint_offsets=footprint)
        self.assertIsNone(path)

    def test_carried_pallet_home_exemption_allows_return_pose(self):
        geometry = self.make_geometry({(3, 1)})
        planner = SpaceTimeAStar(geometry, ReservationTable(0), path_horizon=20)
        footprint = frozenset({(0, 0), (1, 0)})
        path = planner.find_path(
            (2, 2),
            0,
            (2, 1),
            owner=0,
            footprint_offsets=footprint,
            static_exemptions={(1, 0): (3, 1)},
        )
        self.assertIsNotNone(path)
        self.assertEqual(path[-1][0], (2, 1))

    def test_min_goal_time_delays_arrival(self):
        planner = SpaceTimeAStar(self.make_geometry(), ReservationTable(0), path_horizon=20)
        path = planner.find_path((1, 1), 0, (2, 1), owner=0, min_goal_time=4)
        self.assertEqual(path[-1], ((2, 1), 4))

    def test_goal_hold_requires_future_occupancy(self):
        reservations = ReservationTable(0)
        reservations.reserve_pose([(2, 1)], 2, 9)
        planner = SpaceTimeAStar(self.make_geometry(), reservations, path_horizon=20)
        path = planner.find_path((1, 1), 0, (2, 1), owner=0, goal_hold_steps=1)
        self.assertIsNotNone(path)
        self.assertGreaterEqual(path[-1][1], 3)

    def test_find_path_to_row_keeps_nearest_x_when_clear(self):
        planner = SpaceTimeAStar(self.make_geometry(), ReservationTable(0), path_horizon=20)
        path = planner.find_path_to_row(
            (4, 4), 0, 7, owner=0, footprint_offsets=frozenset({(0, 0)})
        )
        self.assertIsNotNone(path)
        self.assertEqual(path[-1][0], (4, 7))
        self.assertEqual(path[-1][1], 3)
        self.assertEqual(planner.counters.fast_row_hits, 1)
        self.assertEqual(planner.counters.expansions, 0)

    def test_find_path_to_row_uses_nearest_available_row_cell(self):
        geometry = self.make_geometry({(4, 7)})
        planner = SpaceTimeAStar(geometry, ReservationTable(0), path_horizon=20)
        path = planner.find_path_to_row(
            (4, 4), 0, 7, owner=0, footprint_offsets=frozenset({(0, 0)})
        )
        self.assertIsNotNone(path)
        self.assertIn(path[-1][0], {(3, 7), (5, 7)})
        self.assertEqual(path[-1][1], 4)
        self.assertEqual(planner.counters.fast_row_hits, 0)
        self.assertGreater(planner.counters.expansions, 0)

    def test_row_goal_can_require_safety_through_future_reservations(self):
        reservations = ReservationTable(0)
        reservations.reserve_pose([(4, 7)], 10, 9)
        planner = SpaceTimeAStar(self.make_geometry(), reservations, path_horizon=20)
        path = planner.find_path_to_row(
            (4, 4),
            0,
            7,
            owner=0,
            footprint_offsets=frozenset({(0, 0)}),
            goal_hold_until=10,
        )
        self.assertIsNotNone(path)
        self.assertNotEqual(path[-1][0], (4, 7))
        self.assertEqual(planner.counters.fast_row_hits, 0)


if __name__ == "__main__":
    unittest.main()
