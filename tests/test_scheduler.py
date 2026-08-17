"""Tests for time-aware multi-robot collision avoidance."""

import unittest

from src.models import ProblemInstance
from src.scheduler import ReservationTable, Scheduler
from src.world import WorldState


def make_empty_world():
    return WorldState(
        ProblemInstance(
            robots=[],
            sku_capacities=[],
            pallets=[],
            orders=[],
        )
    )


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

    def test_reserve_trajectory_tracks_full_footprint(self):
        table = ReservationTable()
        footprint = frozenset({(0, 0), (1, 0)})
        trajectory = [
            (0, (4, 5)),
            (1, (5, 5)),
        ]

        table.reserve_trajectory(trajectory, footprint)

        # At the action-start state, both the current footprint and the cells
        # it intends to enter are reserved to match simulator movement rules.
        self.assertEqual(table.cells[0], {(4, 5), (5, 5), (6, 5)})
        self.assertEqual(table.cells[1], {(5, 5), (6, 5)})
        self.assertIn(((4, 5), (5, 5)), table.edges[0])
        self.assertIn(((5, 5), (6, 5)), table.edges[0])

    def test_future_destination_is_reserved_before_arrival(self):
        table = ReservationTable()
        footprint = frozenset({(0, 0)})
        trajectory = [
            (0, (10, 10)),
            (1, (11, 10)),
            (2, (12, 10)),
        ]

        table.reserve_trajectory(trajectory, footprint)

        # The robot enters (12, 10) during timestep 1, so a lower-priority
        # robot may not still occupy that cell at the start of timestep 1.
        self.assertFalse(table.cell_is_free(1, (12, 10)))
        self.assertFalse(
            table.transition_is_free(
                1,
                (12, 10),
                (12, 10),
                footprint,
            )
        )

    def test_docked_footprint_edge_swap_is_rejected(self):
        table = ReservationTable()
        footprint = frozenset({(0, 0), (1, 0)})

        # Another entity is already moving left across the edge that the
        # attached right-side pallet would traverse in the opposite direction.
        table.reserve_edge(0, (6, 5), (5, 5))

        self.assertFalse(
            table.transition_is_free(
                0,
                (4, 5),
                (5, 5),
                footprint,
            )
        )


class TestTimedPlanning(unittest.TestCase):
    def test_reserved_same_cell_same_timestep_is_avoided(self):
        scheduler = Scheduler(make_empty_world())
        scheduler.reservations.reserve_cell(2, (2, 1))

        trajectory = scheduler.plan_timed_path(
            (0, 1),
            (3, 1),
            start_timestep=0,
            max_timestep=10,
        )

        self.assertTrue(trajectory)
        self.assertNotIn((2, (2, 1)), trajectory)

    def test_robot_can_wait_then_continue_through_same_location(self):
        scheduler = Scheduler(make_empty_world())
        scheduler.reservations.reserve_cell(1, (2, 1))

        # Enclose the route so waiting is the only way around the temporary
        # reservation. Because the simulator forbids entering a cell occupied
        # at the start of a timestep, the robot must wait through timestep 1.
        blocked = {
            (0, 1),
            (1, 0),
            (1, 2),
            (2, 0),
            (2, 2),
            (3, 0),
            (3, 2),
            (4, 1),
        }
        trajectory = scheduler.plan_timed_path(
            (1, 1),
            (3, 1),
            start_timestep=0,
            blocked=blocked,
            max_timestep=10,
        )

        self.assertEqual(
            trajectory,
            [
                (0, (1, 1)),
                (1, (1, 1)),
                (2, (1, 1)),
                (3, (2, 1)),
                (4, (3, 1)),
            ],
        )

    def test_prioritized_head_on_paths_spatially_detour(self):
        scheduler = Scheduler(make_empty_world())

        higher_priority = scheduler.plan_and_reserve(
            (55, 5),
            (59, 5),
            start_timestep=0,
            max_timestep=20,
        )
        lower_priority = scheduler.plan_timed_path(
            (59, 5),
            (55, 5),
            start_timestep=0,
            max_timestep=30,
        )

        self.assertTrue(higher_priority)
        self.assertTrue(lower_priority)

        # Starting at the right boundary means the lower-priority robot cannot
        # simply retreat farther right. It must leave y=5 to get around the
        # higher-priority reserved trajectory.
        self.assertTrue(
            any(position[1] != 5 for _, position in lower_priority)
        )

    def test_docked_footprint_avoids_reserved_cell(self):
        scheduler = Scheduler(make_empty_world())
        footprint = frozenset({(0, 0), (1, 0)})

        # If the center moved directly from x=4 to x=5 at timestep 0, the
        # right-side pallet would occupy reserved cell (6, 5) at state t=1.
        scheduler.reservations.reserve_cell(1, (6, 5))

        trajectory = scheduler.plan_timed_path(
            (4, 5),
            (6, 5),
            start_timestep=0,
            footprint=footprint,
            max_timestep=12,
        )

        self.assertTrue(trajectory)
        self.assertNotIn((1, (5, 5)), trajectory)

    def test_plan_and_reserve_establishes_priority_by_call_order(self):
        scheduler = Scheduler(make_empty_world())

        first = scheduler.plan_and_reserve(
            (1, 10),
            (5, 10),
            start_timestep=0,
            max_timestep=20,
        )
        second = scheduler.plan_and_reserve(
            (3, 8),
            (3, 12),
            start_timestep=0,
            max_timestep=30,
        )

        self.assertTrue(first)
        self.assertTrue(second)

        first_by_time = dict(first)
        second_by_time = dict(second)
        shared_times = set(first_by_time) & set(second_by_time)
        for timestep in shared_times:
            self.assertNotEqual(
                first_by_time[timestep],
                second_by_time[timestep],
            )


if __name__ == "__main__":
    unittest.main()
