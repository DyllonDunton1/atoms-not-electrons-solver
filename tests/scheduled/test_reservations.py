import unittest

from scheduled_solver.models import PalletReservation
from scheduled_solver.reservations import ReservationConflict, ReservationTable


class ReservationTests(unittest.TestCase):
    def test_vertex_padding_blocks_neighboring_times(self):
        table = ReservationTable(padding=1)
        table.reserve_pose([(5, 5)], 10, 0)
        self.assertFalse(table.vertex_is_free([(5, 5)], 9, 1))
        self.assertFalse(table.vertex_is_free([(5, 5)], 10, 1))
        self.assertFalse(table.vertex_is_free([(5, 5)], 11, 1))
        self.assertTrue(table.vertex_is_free([(5, 5)], 12, 1))

    def test_owner_may_overlap_its_own_reservation(self):
        table = ReservationTable(padding=1)
        table.reserve_pose([(5, 5)], 10, 0)
        self.assertTrue(table.vertex_is_free([(5, 5)], 10, 0))
        table.reserve_pose([(5, 5)], 10, 0)

    def test_reverse_edge_is_blocked(self):
        table = ReservationTable(padding=0)
        table.reserve_edges([((1, 1), (2, 1))], 4, 0)
        self.assertFalse(table.edge_is_free([((2, 1), (1, 1))], 4, 1))
        self.assertTrue(table.edge_is_free([((2, 1), (3, 1))], 4, 1))

    def test_edge_padding_applies(self):
        table = ReservationTable(padding=1)
        table.reserve_edges([((1, 1), (2, 1))], 4, 0)
        self.assertFalse(table.edge_is_free([((2, 1), (1, 1))], 3, 1))
        self.assertFalse(table.edge_is_free([((2, 1), (1, 1))], 5, 1))

    def test_pallet_interval_gets_padding(self):
        table = ReservationTable(padding=1)
        table.reserve_pallet(PalletReservation(7, 10, 12, 0, 3))
        self.assertFalse(table.pallet_is_free(7, 8, 8, 1))
        self.assertFalse(table.pallet_is_free(7, 14, 14, 1))
        self.assertTrue(table.pallet_is_free(7, 15, 15, 1))

    def test_conflicting_pallet_reservation_raises(self):
        table = ReservationTable(padding=1)
        table.reserve_pallet(PalletReservation(7, 10, 12, 0, 3))
        with self.assertRaises(ReservationConflict):
            table.reserve_pallet(PalletReservation(7, 12, 13, 1, 4))

    def test_permanent_cell_blocks_other_robots_only(self):
        table = ReservationTable(padding=1)
        table.reserve_permanent_cell((0, 0), 0)
        self.assertFalse(table.vertex_is_free([(0, 0)], 10_000, 1))
        self.assertTrue(table.vertex_is_free([(0, 0)], 10_000, 0))

    def test_terminal_hold_begins_at_finish_time(self):
        table = ReservationTable(padding=1)
        table.set_terminal_hold((9, 0), 20, 0)
        self.assertTrue(table.vertex_is_free([(9, 0)], 19, 1))
        self.assertFalse(table.vertex_is_free([(9, 0)], 20, 1))
        self.assertFalse(table.vertex_is_free([(9, 0)], 10_000, 1))
        self.assertTrue(table.vertex_is_free([(9, 0)], 10_000, 0))

    def test_same_robot_replaces_its_old_terminal_hold(self):
        table = ReservationTable(padding=1)
        table.set_terminal_hold((9, 0), 20, 0)
        table.set_terminal_hold((30, 0), 50, 0)
        self.assertEqual(table.terminal_hold(0), ((30, 0), 50))
        self.assertTrue(table.vertex_is_free([(9, 0)], 100, 1))
        self.assertFalse(table.vertex_is_free([(30, 0)], 100, 1))

    def test_terminal_hold_rejects_future_finite_conflict(self):
        table = ReservationTable(padding=0)
        table.reserve_pose([(9, 0)], 30, 1)
        self.assertFalse(table.terminal_hold_is_free((9, 0), 20, 0))
        with self.assertRaises(ReservationConflict):
            table.set_terminal_hold((9, 0), 20, 0)

    def test_compaction_keeps_padding_boundary_for_future_vertex_checks(self):
        table = ReservationTable(padding=1)
        table.reserve_pose([(5, 5)], 8, 0)   # stored through t=9
        table.reserve_pose([(6, 5)], 10, 0)  # stored from t=9
        table.compact_before(10)

        # A new raw t=10 pose checks t=9..11, so the t=9 boundary must remain.
        self.assertFalse(table.vertex_reservation_is_free([(5, 5)], 10, 1))
        self.assertFalse(table.vertex_reservation_is_free([(6, 5)], 10, 1))

    def test_compaction_discards_pallet_intervals_that_cannot_overlap_future(self):
        table = ReservationTable(padding=1)
        table.reserve_pallet(PalletReservation(7, 0, 5, 0, 1))   # stored end=6
        table.reserve_pallet(PalletReservation(7, 10, 12, 1, 2)) # stored 9..13
        table.compact_before(10)  # cutoff=9
        self.assertEqual(table.pallet_intervals(7), ((9, 13, 1, 2),))

    def test_cached_reservation_horizon_tracks_latest_finite_reservation(self):
        table = ReservationTable(padding=1)
        table.reserve_pose([(1, 1)], 10, 0)
        self.assertEqual(table.reservation_horizon(), 11)
        table.reserve_pallet(PalletReservation(7, 20, 25, 0, 1))
        self.assertEqual(table.reservation_horizon(), 26)
        table.compact_before(100)
        # A stale cached value is intentionally allowed only in the past.
        self.assertEqual(table.reservation_horizon(), 26)


if __name__ == "__main__":
    unittest.main()
