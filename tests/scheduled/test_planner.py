import unittest

from scheduled_solver.config import SchedulerConfig
from scheduled_solver.geometry import build_geometry
from scheduled_solver.inventory import InventoryTimeline
from scheduled_solver.models import (
    InventoryEvent,
    OrderSpec,
    PalletReservation,
    PalletSpec,
    PlannerStats,
)
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

    def test_full_horizon_plan_finishes_simple_order_at_closest_fulfillment_x(self):
        pallets = (
            PalletSpec(0, (10, 7), 0, 5),
            PalletSpec(1, (10, 8), 1, 5),
            PalletSpec(2, (11, 7), 2, 5),
            PalletSpec(3, (11, 8), 3, 5),
        )
        planner, _, _, _ = self.make_planner(pallets)
        schedule = planner.plan_order(0, OrderSpec(0, (0, 1)), (9, 10), 0)
        self.assertEqual(schedule.finish_timestep, schedule.poses[-1].timestep)
        # Both serviced pallets use the exposed x=9 service lane, so the
        # minimum-distance fulfillment route is straight up to (9, 0), not the
        # old robot-id parking target (0, 0).
        self.assertEqual(schedule.end_position, (9, 0))
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

    def test_refill_uses_closest_replenishment_x(self):
        pallets = (
            PalletSpec(0, (10, 7), 0, 1),
            PalletSpec(1, (11, 7), 1, 1),
        )
        planner, _, _, _ = self.make_planner(pallets, beam_width=2)
        schedule = planner.plan_order(0, OrderSpec(0, (0, 0)), (9, 7), 0)
        refill_pose = next(
            pose for pose in schedule.poses
            if pose.center[1] == planner.geometry.replenishment_y
            and len(pose.footprint_offsets) == 2
        )
        # The robot docks from x=9, and the x=9 vertical route is clear, so the
        # earliest reachable replenishment position keeps that same x.
        self.assertEqual(refill_pose.center[0], 9)

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

    def test_existing_pallet_reservation_delays_service_past_both_padding_windows(self):
        pallets = (
            PalletSpec(0, (10, 7), 0, 3),
            PalletSpec(1, (11, 7), 1, 3),
        )
        planner, reservations, _, _ = self.make_planner(pallets, beam_width=2, padding=1)
        reservations.reserve_pallet(PalletReservation(0, 0, 10, 9, 99))
        schedule = planner.plan_order(0, OrderSpec(0, (0,)), (9, 7), 0)
        first_pick = next(a for a in schedule.actions if a.action.value == "pick")

        # The committed [0,10] interval is stored as [-1,11].  The new
        # candidate service interval is also expanded by one timestep, so t=12
        # still overlaps at t=11.  The first legal raw service timestep is 13.
        self.assertEqual(first_pick.timestep, 13)

    def test_pallet_retry_timing_scales_with_padding_width(self):
        pallets = (
            PalletSpec(0, (10, 7), 0, 3),
            PalletSpec(1, (11, 7), 1, 3),
        )
        planner, reservations, _, _ = self.make_planner(pallets, beam_width=2, padding=2)
        reservations.reserve_pallet(PalletReservation(0, 0, 10, 9, 99))
        schedule = planner.plan_order(0, OrderSpec(0, (0,)), (9, 7), 0)
        first_pick = next(a for a in schedule.actions if a.action.value == "pick")

        # Raw [0,10] is stored as [-2,12], and the new candidate is itself
        # expanded by two timesteps.  It must therefore begin at t=15.
        self.assertEqual(first_pick.timestep, 15)

    def test_later_plan_may_take_only_surplus_before_future_reserved_service(self):
        pallets = (
            PalletSpec(0, (10, 7), 0, 5),
            PalletSpec(1, (11, 7), 1, 5),
        )
        planner, reservations, inventory, _ = self.make_planner(
            pallets, beam_width=2, padding=0
        )
        inventory.commit([InventoryEvent(100, 0, "pick", 3, 9)])
        reservations.reserve_pallet(PalletReservation(0, 100, 102, 9, 99))

        schedule = planner.plan_order(0, OrderSpec(0, (0, 0)), (9, 7), 0)
        picks = [action for action in schedule.actions if action.action.value == "pick"]

        self.assertEqual(len(picks), 2)
        self.assertTrue(all(action.timestep < 100 for action in picks))
        self.assertFalse(any(action.action.value == "dock" for action in schedule.actions))
        self.assertTrue(inventory.events_are_feasible(schedule.inventory_events))

    def test_later_plan_waits_when_full_quantity_would_steal_reserved_stock(self):
        pallets = (
            PalletSpec(0, (10, 7), 0, 5),
            PalletSpec(1, (11, 7), 1, 5),
        )
        planner, reservations, inventory, _ = self.make_planner(
            pallets, beam_width=2, padding=0
        )
        inventory.commit([InventoryEvent(100, 0, "pick", 4, 9)])
        reservations.reserve_pallet(PalletReservation(0, 100, 102, 9, 99))

        schedule = planner.plan_order(0, OrderSpec(0, (0, 0)), (9, 7), 0)
        picks = [action for action in schedule.actions if action.action.value == "pick"]
        docks = [action for action in schedule.actions if action.action.value == "dock"]

        # Only one unit is surplus before the earlier commitment.  Because the
        # SKU is serviced as one complete stop, the failed early partial attempt
        # is discarded; the robot waits until that commitment is over instead.
        self.assertTrue(picks)
        self.assertGreater(min(action.timestep for action in picks), 102)
        self.assertTrue(docks)
        self.assertGreater(min(action.timestep for action in docks), 102)
        self.assertTrue(inventory.events_are_feasible(schedule.inventory_events))

    def test_later_plan_never_refills_pallet_before_future_reserved_service(self):
        pallets = (
            PalletSpec(0, (10, 7), 0, 1),
            PalletSpec(1, (11, 7), 1, 1),
        )
        planner, reservations, inventory, _ = self.make_planner(
            pallets, beam_width=2, padding=0
        )
        inventory.commit([InventoryEvent(100, 0, "pick", 1, 9)])
        reservations.reserve_pallet(PalletReservation(0, 100, 102, 9, 99))

        schedule = planner.plan_order(0, OrderSpec(0, (0,)), (9, 7), 0)
        docks = [action for action in schedule.actions if action.action.value == "dock"]
        picks = [action for action in schedule.actions if action.action.value == "pick"]

        self.assertTrue(docks)
        self.assertGreater(min(action.timestep for action in docks), 102)
        self.assertGreater(min(action.timestep for action in picks), 102)
        self.assertTrue(inventory.events_are_feasible(schedule.inventory_events))

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
