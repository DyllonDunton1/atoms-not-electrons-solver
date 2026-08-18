"""Tests for one-timestep fleet collision reservations."""

import unittest

from src.scheduler import ReservationTable


class TestReservationTable(unittest.TestCase):
    def test_cell_reservations_are_time_specific(self):
        table = ReservationTable()
        position = (10, 10)

        table.reserve_cell(3, position)

        self.assertFalse(table.cell_is_free(3, position))
        self.assertTrue(table.cell_is_free(2, position))
        self.assertTrue(table.cell_is_free(4, position))

    def test_head_on_edge_swaps_are_rejected(self):
        table = ReservationTable()
        start = (4, 4)
        end = (5, 4)

        table.reserve_edge(7, start, end)

        self.assertFalse(table.edge_is_free(7, start, end))
        self.assertFalse(table.edge_is_free(7, end, start))
        self.assertTrue(table.edge_is_free(8, end, start))

    def test_move_reserves_current_and_destination_rigid_footprint(self):
        table = ReservationTable()
        footprint = frozenset({(0, 0), (1, 0)})

        table.reserve_transition(0, (4, 5), (5, 5), footprint)

        # Robot + east-docked pallet occupy x=4,5 before the move and x=5,6
        # after it. The whole swept footprint is unavailable to later robots.
        self.assertEqual(table.cells[0], {(4, 5), (5, 5), (6, 5)})
        self.assertEqual(table.cells[1], {(5, 5), (6, 5)})
        self.assertIn(((4, 5), (5, 5)), table.edges[0])
        self.assertIn(((5, 5), (6, 5)), table.edges[0])

    def test_wait_reserves_complete_rigid_footprint(self):
        table = ReservationTable()
        footprint = frozenset({(0, 0), (0, 1)})

        table.reserve_transition(4, (8, 8), (8, 8), footprint)

        self.assertEqual(table.cells[4], {(8, 8), (8, 9)})
        self.assertEqual(table.cells[5], {(8, 8), (8, 9)})
        self.assertFalse(
            table.transition_is_free(
                4,
                (7, 8),
                (8, 8),
                frozenset({(0, 0)}),
            )
        )

    def test_docked_footprint_edge_swap_is_rejected(self):
        table = ReservationTable()
        footprint = frozenset({(0, 0), (1, 0)})

        # The attached east-side pallet would traverse (5,5)->(6,5), opposite
        # an already committed lower-ID entity transition.
        table.reserve_edge(0, (6, 5), (5, 5))

        self.assertFalse(
            table.transition_is_free(
                0,
                (4, 5),
                (5, 5),
                footprint,
            )
        )


if __name__ == "__main__":
    unittest.main()
