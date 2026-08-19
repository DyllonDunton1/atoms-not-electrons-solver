import unittest
from unittest.mock import patch

from scheduled_solver.bounded_planner import BudgetAwareFullHorizonBeamPlanner
from scheduled_solver.config import SchedulerConfig
from scheduled_solver.geometry import build_geometry
from scheduled_solver.inventory import InventoryTimeline
from scheduled_solver.models import PalletSpec, PlannerStats
from scheduled_solver.planner import FullHorizonBeamPlanner
from scheduled_solver.reservations import ReservationTable


class BudgetAwarePlannerTests(unittest.TestCase):
    def make_planner(self):
        pallets = (
            PalletSpec(0, (10, 7), 0, 5),
            PalletSpec(1, (11, 7), 1, 5),
        )
        geometry = build_geometry(pallets, require_24_columns=False)
        reservations = ReservationTable(0)
        inventory = InventoryTimeline(pallets)
        config = SchedulerConfig(
            beam_width=1,
            candidate_width=1,
            reservation_padding=0,
            path_horizon=32,
            max_path_expansions=100,
            candidate_max_path_expansions=10,
            max_beam_depth=8,
            require_24_columns=False,
        )
        stats = PlannerStats()
        planner = BudgetAwareFullHorizonBeamPlanner(
            geometry,
            pallets,
            reservations,
            inventory,
            config,
            stats,
        )
        return planner, stats

    def test_capped_low_budget_column_is_rejected_even_if_base_returns_partial(self):
        planner, stats = self.make_planner()
        sentinel = object()

        def fake_expand(base_self, *args, **kwargs):
            base_self._astar_counters.capped_calls += 1
            return sentinel

        with patch.object(FullHorizonBeamPlanner, "_expand_column", new=fake_expand):
            result = planner._expand_column(
                object(),
                planner.geometry.columns[0],
                "down",
                0,
                0,
                candidate_budget=True,
            )

        self.assertIsNone(result)
        self.assertEqual(stats.capped_candidate_rejections, 1)

    def test_full_budget_column_is_not_rejected_by_cap_wrapper(self):
        planner, stats = self.make_planner()
        sentinel = object()

        def fake_expand(base_self, *args, **kwargs):
            base_self._astar_counters.capped_calls += 1
            return sentinel

        with patch.object(FullHorizonBeamPlanner, "_expand_column", new=fake_expand):
            result = planner._expand_column(
                object(),
                planner.geometry.columns[0],
                "down",
                0,
                0,
                candidate_budget=False,
            )

        self.assertIs(result, sentinel)
        self.assertEqual(stats.capped_candidate_rejections, 0)

    def test_failed_capped_order_retries_once_with_full_budget(self):
        planner, stats = self.make_planner()
        sentinel = object()
        budgets_seen = []

        def fake_plan(base_self, *args, **kwargs):
            budgets_seen.append(planner._force_full_order_budget)
            if len(budgets_seen) == 1:
                raise RuntimeError("capped attempt failed")
            return sentinel

        with patch.object(FullHorizonBeamPlanner, "plan_order", new=fake_plan):
            result = planner.plan_order(0, object(), (0, 0), 0)

        self.assertIs(result, sentinel)
        self.assertEqual(budgets_seen, [False, True])
        self.assertEqual(stats.order_full_budget_rescues, 1)
        self.assertFalse(planner._force_full_order_budget)


if __name__ == "__main__":
    unittest.main()
