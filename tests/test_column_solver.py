"""Focused tests for the 24-column / 48-directed-route collection strategy."""

from collections import Counter
from pathlib import Path
import unittest

from src.column_solver import (
    DOWN,
    UP,
    ColumnAwareSolver,
    DirectedColumnPlanner,
)
from src.parser import parse_problem
from src.world import WorldState


REPO_ROOT = Path(__file__).resolve().parents[1]
BIG_ORDER_PATH = REPO_ROOT / "source_material" / "BIG_ORDER.txt"


class TestDirectedColumnPlanner(unittest.TestCase):
    def setUp(self):
        self.world = WorldState(parse_problem(BIG_ORDER_PATH))
        self.planner = DirectedColumnPlanner(self.world)

    def test_layout_splits_twelve_islands_into_twenty_four_columns(self):
        self.assertEqual(len(self.planner.layout.columns), 24)
        self.assertEqual(len(self.planner.layout.pallet_to_column), 240)

        for column in self.planner.layout.columns:
            self.assertEqual(len(column.pallet_ids), 10)
            self.assertEqual(len(column.home_positions), 10)
            self.assertTrue(all(x == column.pallet_x for x, _ in column.home_positions))
            self.assertTrue(
                all(abs(column.service_x - x) == 1 for x, _ in column.home_positions)
            )
            self.assertEqual(
                [position[1] for position in column.home_positions],
                sorted(position[1] for position in column.home_positions),
            )

    def test_both_directions_use_one_exposed_lane_and_monotonic_stops(self):
        column = self.planner.layout.columns[0]
        remaining = Counter(
            self.world.pallets[pallet_id].sku for pallet_id in column.pallet_ids
        )
        # Make quantity deliberately larger than one so the plan's utility can
        # be checked as distinct-SKU count rather than requested item count.
        remaining = {sku: count + 4 for sku, count in remaining.items()}
        start = (column.service_x, 18)

        up = self.planner.plan_column_direction(
            column.column_id,
            UP,
            start,
            remaining,
        )
        down = self.planner.plan_column_direction(
            column.column_id,
            DOWN,
            start,
            remaining,
        )

        self.assertIsNotNone(up)
        self.assertIsNotNone(down)
        for plan, reverse in ((up, True), (down, False)):
            ys = [stop.pickup[1] for stop in plan.stops]
            self.assertEqual(ys, sorted(ys, reverse=reverse))
            self.assertTrue(
                all(stop.pickup[0] == column.service_x for stop in plan.stops)
            )
            self.assertEqual(plan.useful_sku_count, len(plan.stops))
            self.assertEqual(plan.useful_quantity, len(plan.stops))
            span = abs(ys[-1] - ys[0]) if len(ys) > 1 else 0
            self.assertGreaterEqual(plan.planned_distance, span)

    def test_completed_column_pass_does_not_final_rescan(self):
        solver = ColumnAwareSolver(
            self.world,
            robot_ids=[0],
            order_ids=[0],
            max_timesteps=1000,
        )
        solver._assign_free_robots()
        self.assertTrue(solver._select_new_aisle(0))

        state = solver.states[0]
        active_column = state.active_aisle_id
        self.assertIsNotNone(active_column)
        self.assertIsNotNone(state.aisle_plan)
        self.assertTrue(state.remaining_by_sku)

        state.aisle_stop_index = len(state.aisle_plan.stops) - 1
        solver._advance_stop(0)

        # Remaining work is intentionally left for a new global route choice.
        # The just-finished column is remembered and excluded unless the normal
        # fallback later determines it is the only useful choice.
        self.assertIsNone(state.active_aisle_id)
        self.assertIsNone(state.aisle_plan)
        self.assertEqual(state.previous_aisle_id, active_column)
        self.assertTrue(state.remaining_by_sku)

    def test_far_column_activation_ignores_persistent_adjacency(self):
        solver = ColumnAwareSolver(
            self.world,
            robot_ids=[0],
            order_ids=[0],
            max_timesteps=1000,
        )
        solver._assign_free_robots()
        self.assertTrue(solver._select_new_aisle(0))

        stop = solver._current_stop(0)
        self.assertIsNotNone(stop)
        robot = solver.world.robots[0]
        self.assertGreater(
            abs(robot.position[0] - stop.pickup[0])
            + abs(robot.position[1] - stop.pickup[1]),
            1,
        )
        solver._persistent_priority_blocked_pallet_ids = lambda robot_id: {
            stop.pallet_id
        }

        self.assertTrue(solver._activate_current_stop(0))
        self.assertEqual(solver.pallet_claims.get(stop.pallet_id), 0)

    def test_local_column_activation_defers_persistent_adjacency(self):
        solver = ColumnAwareSolver(
            self.world,
            robot_ids=[0],
            order_ids=[0],
            max_timesteps=1000,
        )
        solver._assign_free_robots()
        self.assertTrue(solver._select_new_aisle(0))

        stop = solver._current_stop(0)
        self.assertIsNotNone(stop)
        solver.world.robots[0].position = (stop.pickup[0], stop.pickup[1] + 1)
        solver._persistent_priority_blocked_pallet_ids = lambda robot_id: {
            stop.pallet_id
        }

        self.assertFalse(solver._activate_current_stop(0))
        self.assertNotEqual(solver.pallet_claims.get(stop.pallet_id), 0)
        self.assertIsNone(solver.states[0].pallet_id)

    def test_claimed_stop_is_deferred_when_blocker_persists_locally(self):
        solver = ColumnAwareSolver(
            self.world,
            robot_ids=[0],
            order_ids=[0],
            max_timesteps=1000,
        )
        solver._assign_free_robots()
        self.assertTrue(solver._select_new_aisle(0))

        stop = solver._current_stop(0)
        self.assertIsNotNone(stop)
        solver._persistent_priority_blocked_pallet_ids = lambda robot_id: set()
        self.assertTrue(solver._activate_current_stop(0))
        self.assertEqual(solver.pallet_claims.get(stop.pallet_id), 0)

        solver.world.robots[0].position = (stop.pickup[0], stop.pickup[1] + 1)
        solver._persistent_priority_blocked_pallet_ids = lambda robot_id: {
            stop.pallet_id
        }
        solver._replan_active_aisle(0)

        self.assertNotEqual(solver.pallet_claims.get(stop.pallet_id), 0)
        self.assertIsNone(solver.states[0].pallet_id)

    def test_previous_column_fallback_alone_applies_persistent_adjacency(self):
        solver = ColumnAwareSolver(
            self.world,
            robot_ids=[0],
            order_ids=[0],
            max_timesteps=1000,
        )
        solver._assign_free_robots()
        state = solver.states[0]

        previous_column_id = 0
        previous_blocker = solver.aisle_planner.layout.columns[
            previous_column_id
        ].pallet_ids[0]
        other_blocker = solver.aisle_planner.layout.columns[1].pallet_ids[0]
        state.previous_aisle_id = previous_column_id

        calls = []

        def fake_choose_plan(
            start,
            remaining_by_sku,
            *,
            congestion_by_aisle,
            unavailable_pallet_ids=(),
            blocked=(),
            excluded_aisle_ids=(),
        ):
            calls.append(
                (
                    set(unavailable_pallet_ids),
                    set(excluded_aisle_ids),
                )
            )
            return None

        solver.aisle_planner.choose_plan = fake_choose_plan
        solver._persistent_priority_blocked_pallet_ids = lambda robot_id: {
            previous_blocker,
            other_blocker,
        }

        self.assertFalse(solver._select_new_aisle(0))
        self.assertEqual(len(calls), 2)

        normal_unavailable, normal_excluded = calls[0]
        fallback_unavailable, fallback_excluded = calls[1]

        self.assertEqual(normal_excluded, {previous_column_id})
        self.assertNotIn(previous_blocker, normal_unavailable)
        self.assertNotIn(other_blocker, normal_unavailable)

        self.assertEqual(fallback_excluded, set())
        self.assertIn(previous_blocker, fallback_unavailable)
        self.assertNotIn(other_blocker, fallback_unavailable)


if __name__ == "__main__":
    unittest.main()
