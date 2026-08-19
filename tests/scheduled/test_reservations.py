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


if __name__ == "__main__":
    unittest.main()
