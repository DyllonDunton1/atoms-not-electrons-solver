import unittest

from scheduled_solver.config import SchedulerConfig
from scheduled_solver.geometry import build_geometry
from scheduled_solver.inventory import InventoryTimeline
from scheduled_solver.models import OrderSpec, PalletReservation, PalletSpec, PlannerStats
from scheduled_solver.planner import FullHorizonBeamPlanner
from scheduled_solver.reservations import ReservationTable


class PlannerTests(unittest.TestCase):
    def make_planner(self, pallets, *, beam_width=4, padding=0):
        geometry = build_geometry(pallets, require_24_columns=False)
        reservations = ReservationTable(padding)
        inventory = InventoryTimeline(pallets)
        config = SchedulerConfig(
            beam_width=beam_width,
            reservation_padding=padding,
            path_horizon=256,
            max_path_expansions=100_000,
            max_beam_depth=12,
            require_24_columns=False,
        )
        stats = PlannerStats()
        return (
            FullHorizonBeamPlanner(
                geometry, pallets, reservations, inventory, config, stats
            ),
            reservations,
            inventory,
            stats,
        )

    def test_full_horizon_plan_finishes_simple_order(self):
        pallets = (
            PalletSpec(0, (10, 7), 0, 5),
            PalletSpec(1, (10, 8), 1, 5),
            PalletSpec(2, (11, 7), 2, 5),
            PalletSpec(3, (11, 8), 3, 5),
        )
        planner, _, _, _ = self.make_planner(pallets)
        schedule = planner.plan_order(0, OrderSpec(0, (0, 1)), (9, 10), 0)
        self.assertEqual(schedule.finish_timestep, schedule.poses[-1].timestep)
        self.assertEqual(schedule.end_position, (0, 0))
        self.assertEqual(sum(a.action.value == "pick" for a in schedule.actions), 2)
        self.assertEqual(schedule.actions[-1].action.value, "fulfill")
        self.assertTrue(schedule.column_visits)

    def test_direction_is_part_of_each_column_visit(self):
        pallets = (
            PalletSpec(0, (10, 7), 0, 5),
            PalletSpec(1, (10, 8), 1, 5),
            PalletSpec(2, (11, 7), 2, 5),
            PalletSpec(3, (11, 8), 3, 5),
        )
        planner, _, _, _ = self.make_planner(pallets)
        schedule = planner.plan_order(0, OrderSpec(0, (0, 1)), (9, 10), 0)
        self.assertIn(schedule.column_visits[0].direction, {"up", "down"})

    def test_empty_stock_triggers_refill_round_trip(self):
        # Capacity one and quantity two guarantee one refill after the first pick.
        pallets = (
            PalletSpec(0, (10, 7), 0, 1),
            PalletSpec(1, (11, 7), 1, 1),
        )
        planner, _, _, _ = self.make_planner(pallets, beam_width=2)
        schedule = planner.plan_order(0, OrderSpec(0, (0, 0)), (9, 7), 0)
        kinds = [action.action.value for action in schedule.actions]
        self.assertIn("dock", kinds)
        self.assertIn("undock", kinds)
        self.assertEqual(kinds.count("pick"), 2)
        self.assertEqual(sum(e.kind == "refill" for e in schedule.inventory_events), 1)

    def test_pallet_home_remains_static_during_refill(self):
        pallets = (
            PalletSpec(0, (10, 7), 0, 1),
            PalletSpec(1, (11, 7), 1, 1),
        )
        planner, _, _, _ = self.make_planner(pallets, beam_width=2)
        schedule = planner.plan_order(0, OrderSpec(0, (0, 0)), (9, 7), 0)
        carried_poses = [pose for pose in schedule.poses if len(pose.footprint_offsets) == 2]
        self.assertTrue(carried_poses)
        self.assertIn((10, 7), planner.geometry.static_blocked)
        self.assertTrue(any((1, 0) in pose.exemptions for pose in carried_poses))

    def test_existing_pallet_reservation_delays_or_avoids_service(self):
        pallets = (
            PalletSpec(0, (10, 7), 0, 3),
            PalletSpec(1, (11, 7), 1, 3),
        )
        planner, reservations, _, _ = self.make_planner(pallets, beam_width=2, padding=1)
        reservations.reserve_pallet(PalletReservation(0, 0, 10, 9, 99))
        schedule = planner.plan_order(0, OrderSpec(0, (0,)), (9, 7), 0)
        first_pick = next(a for a in schedule.actions if a.action.value == "pick")
        self.assertGreater(first_pick.timestep, 10)

    def test_planner_reports_astar_activity(self):
        pallets = (
            PalletSpec(0, (10, 7), 0, 5),
            PalletSpec(1, (11, 7), 1, 5),
        )
        planner, _, _, stats = self.make_planner(pallets, beam_width=2)
        planner.plan_order(0, OrderSpec(0, (0,)), (9, 10), 0)
        self.assertGreater(stats.astar_calls, 0)
        self.assertGreater(stats.astar_expansions, 0)


if __name__ == "__main__":
    unittest.main()
