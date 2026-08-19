import unittest

from scheduled_solver.config import SchedulerConfig
from scheduled_solver.models import OrderSpec, PalletSpec, ProblemInstance, RobotSpec
from scheduled_solver.solver import ScheduledSolver
from scheduled_solver.validation import validate_action_uniqueness


def tiny_problem(order_skus=((0,), (1,), (0,))):
    pallets = (
        PalletSpec(0, (10, 7), 0, 5),
        PalletSpec(1, (10, 8), 1, 5),
        PalletSpec(2, (11, 7), 0, 5),
        PalletSpec(3, (11, 8), 1, 5),
    )
    return ProblemInstance(
        robots=(RobotSpec(0, (9, 10)), RobotSpec(1, (12, 10))),
        sku_capacities=(5, 5),
        pallets=pallets,
        orders=tuple(OrderSpec(i, tuple(skus)) for i, skus in enumerate(order_skus)),
    )


def config():
    return SchedulerConfig(
        beam_width=3,
        reservation_padding=1,
        path_horizon=256,
        max_path_expansions=100_000,
        max_beam_depth=12,
        require_24_columns=False,
    )


class SolverTests(unittest.TestCase):
    def test_initial_orders_assign_fifo_to_free_robots(self):
        solver = ScheduledSolver(tiny_problem(order_skus=((0,), (1,))), config=config())
        actions = solver.solve()
        self.assertTrue(actions)
        self.assertEqual(solver.assignment[0], 0)
        self.assertEqual(solver.assignment[1], 1)

    def test_third_order_goes_to_next_free_robot(self):
        solver = ScheduledSolver(tiny_problem(), config=config())
        solver.solve()
        first = {s.order_id: s for s in solver.schedules if s.order_id in {0, 1}}
        expected = min(
            [(first[0].finish_timestep, 0), (first[1].finish_timestep, 1)]
        )[1]
        self.assertEqual(solver.assignment[2], expected)

    def test_all_actions_are_unique_by_robot_and_timestep(self):
        solver = ScheduledSolver(tiny_problem(), config=config())
        actions = solver.solve()
        validate_action_uniqueness(actions)
        self.assertEqual(
            len(actions), len({(a.timestep, a.robot_id) for a in actions})
        )

    def test_each_schedule_is_full_horizon_and_ends_fulfilled(self):
        solver = ScheduledSolver(tiny_problem(), config=config())
        solver.solve()
        self.assertEqual(len(solver.schedules), 3)
        for schedule in solver.schedules:
            self.assertTrue(schedule.column_visits)
            self.assertEqual(schedule.actions[-1].action.value, "fulfill")
            self.assertEqual(schedule.end_position[1], 0)

    def test_fulfillment_is_not_forced_to_robot_id_x(self):
        solver = ScheduledSolver(tiny_problem(order_skus=((0,),)), config=config())
        solver.solve()
        schedule = solver.schedules[0]
        self.assertEqual(schedule.robot_id, 0)
        self.assertEqual(schedule.end_position, (9, 0))
        self.assertNotEqual(schedule.end_position, (schedule.robot_id, 0))

    def test_finished_robot_holds_its_actual_fulfillment_cell(self):
        solver = ScheduledSolver(tiny_problem(order_skus=((0,),)), config=config())
        solver.solve()
        schedule = solver.schedules[0]
        self.assertEqual(
            solver.reservations.terminal_hold(0),
            (schedule.end_position, schedule.finish_timestep),
        )
        self.assertFalse(
            solver.reservations.vertex_is_free(
                [schedule.end_position], schedule.finish_timestep + 100, 1
            )
        )
        self.assertTrue(
            solver.reservations.vertex_is_free(
                [schedule.end_position], schedule.finish_timestep + 100, 0
            )
        )

    def test_new_order_replaces_same_robots_previous_terminal_hold(self):
        problem = tiny_problem(order_skus=((0,), (1,)))
        solver = ScheduledSolver(problem, robot_ids=[0], config=config())
        solver.solve()
        final_schedule = solver.schedules[-1]
        self.assertEqual(
            solver.reservations.terminal_hold(0),
            (final_schedule.end_position, final_schedule.finish_timestep),
        )

    def test_inactive_robot_start_becomes_static_obstacle(self):
        problem = tiny_problem(order_skus=((0,),))
        solver = ScheduledSolver(problem, robot_ids=[0], config=config())
        self.assertIn((12, 10), solver.geometry.static_blocked)

    def test_committed_pallet_inventory_changes_future_plans(self):
        problem = tiny_problem(order_skus=((0, 0, 0, 0, 0), (0,)))
        solver = ScheduledSolver(problem, robot_ids=[0], config=config())
        solver.solve()
        self.assertEqual(len(solver.schedules), 2)
        self.assertTrue(solver.inventory.events_for(0) or solver.inventory.events_for(2))

    def test_planner_stats_accumulate(self):
        solver = ScheduledSolver(tiny_problem(), config=config())
        solver.solve()
        self.assertEqual(solver.stats.orders_planned, 3)
        self.assertGreater(solver.stats.beam_expansions, 0)
        self.assertGreater(solver.stats.planning_seconds, 0.0)


if __name__ == "__main__":
    unittest.main()
