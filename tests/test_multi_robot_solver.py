"""Integration tests for the concurrent FIFO baseline solver."""

from pathlib import Path
import threading
import unittest

from src.models import ActionType, Order, Pallet, ProblemInstance, Robot
from src.multi_robot_solver import Intent, MultiRobotSolver
from src.parser import parse_problem
from src.simulator import Simulator
from src.world import WorldState


REPO_ROOT = Path(__file__).resolve().parents[1]
BIG_ORDER_PATH = REPO_ROOT / "source_material" / "BIG_ORDER.txt"


class TestMultiRobotSolver(unittest.TestCase):
    def _solve_with_progress(self, solver, label):
        result = {}
        error = {}

        def run_solver():
            try:
                result["actions"] = solver.solve()
            except BaseException as exception:
                error["exception"] = exception

        thread = threading.Thread(target=run_solver, daemon=True)
        thread.start()

        while thread.is_alive():
            thread.join(timeout=2.0)
            if not thread.is_alive():
                break

            active = {}
            for robot_id in solver.active_robot_ids:
                state = solver.states[robot_id]
                if state.task is None:
                    active[robot_id] = "idle"
                else:
                    active[robot_id] = (
                        f"order {state.task.order_id} / {state.phase}"
                    )

            print(
                f"[{label}] t={solver.world.timestep} "
                f"completed={len(solver.completed_ids)}/{len(solver.target_ids)} "
                f"assigned={len(solver.assigned_ids)} "
                f"queued={len(solver.queue)} "
                f"active={active}",
                flush=True,
            )

        if "exception" in error:
            raise error["exception"]
        return result["actions"]

    def _run_prefix(self, robot_ids, order_count):
        problem = parse_problem(BIG_ORDER_PATH)
        world = WorldState(problem)
        order_ids = list(range(order_count))
        solver = MultiRobotSolver(
            world,
            robot_ids=robot_ids,
            order_ids=order_ids,
        )
        label = f"{len(robot_ids)} robots / {order_count} orders"
        actions = self._solve_with_progress(solver, label)

        self.assertEqual(solver.assigned_ids, set(order_ids))
        self.assertEqual(solver.completed_ids, set(order_ids))
        self.assertEqual(len(solver.queue), 0)
        self.assertTrue(all(world.orders[i].fulfilled for i in order_ids))
        self.assertTrue(all(pallet.count >= 0 for pallet in world.pallets.values()))

        action_keys = [(action.timestep, action.robot_id) for action in actions]
        self.assertEqual(len(action_keys), len(set(action_keys)))
        world.validate()

        # Replay from a fresh parse so correctness does not depend only on the
        # solver's in-place simulation while generating the schedule.
        replay_world = WorldState(parse_problem(BIG_ORDER_PATH))
        Simulator(replay_world).run(actions)
        self.assertTrue(all(replay_world.orders[i].fulfilled for i in order_ids))
        self.assertTrue(
            all(pallet.count >= 0 for pallet in replay_world.pallets.values())
        )
        replay_world.validate()

        return world, actions

    def test_lower_priority_robot_yields_immediately(self):
        problem = ProblemInstance(
            robots=[Robot(0, (5, 6)), Robot(1, (5, 5))],
            sku_capacities=[],
            pallets=[],
            orders=[Order(0, []), Order(1, [])],
        )
        world = WorldState(problem)
        solver = MultiRobotSolver(
            world,
            robot_ids=[0, 1],
            order_ids=[0, 1],
            max_timesteps=20,
        )
        intents = {
            0: Intent(move_goal=(5, 4)),
            1: Intent(move_goal=(5, 7)),
        }

        first_actions, chosen = solver._plan_moves(intents)
        first_by_robot = {action.robot_id: action for action in first_actions}

        # Robot 0 has priority and must wait one timestep because robot 1
        # occupies the next cell at action start. Robot 1 must yield immediately
        # rather than also waiting and creating a repeated optimistic deadlock.
        self.assertNotIn(0, first_by_robot)
        self.assertIn(1, first_by_robot)
        self.assertEqual(first_by_robot[1].action, ActionType.MOVE)
        self.assertNotEqual(first_by_robot[1].target, (5, 6))

        solver.simulator.step(first_actions)
        solver._advance_movement_cache(chosen)

        second_actions, chosen = solver._plan_moves(intents)
        second_by_robot = {action.robot_id: action for action in second_actions}
        self.assertIn(0, second_by_robot)
        self.assertEqual(second_by_robot[0].target, (5, 5))

        solver.simulator.step(second_actions)
        solver._advance_movement_cache(chosen)

        for _ in range(8):
            if (
                world.robots[0].position == (5, 4)
                and world.robots[1].position == (5, 7)
            ):
                break
            actions, chosen = solver._plan_moves(intents)
            solver.simulator.step(actions)
            solver._advance_movement_cache(chosen)

        self.assertEqual(world.robots[0].position, (5, 4))
        self.assertEqual(world.robots[1].position, (5, 7))

    def test_docked_lower_priority_robot_keeps_immediate_escape(self):
        # Mirrors the t=7461 long-run failure: robot 1 carries a pallet on its
        # east side in a service lane. Up/down keep that pallet in occupied
        # pallet cells and moving right puts the robot center on a pallet home,
        # so moving left is its only legal immediate yield move.
        pallets = [
            Pallet(0, (17, 10), 0, 1, 1, (17, 10), 1, (1, 0)),
            Pallet(1, (17, 9), 1, 1, 1, (17, 9)),
            Pallet(2, (17, 11), 2, 1, 1, (17, 11)),
            Pallet(3, (18, 10), 3, 1, 1, (18, 10)),
        ]
        problem = ProblemInstance(
            robots=[
                Robot(0, (16, 11)),
                Robot(1, (16, 10), docked_pallets=[0]),
            ],
            sku_capacities=[1, 1, 1, 1],
            pallets=pallets,
            orders=[Order(0, []), Order(1, [])],
        )
        world = WorldState(problem)
        solver = MultiRobotSolver(
            world,
            robot_ids=[0, 1],
            order_ids=[0, 1],
            max_timesteps=20,
        )
        intents = {
            0: Intent(move_goal=(16, 9)),
            1: Intent(move_goal=(16, 39)),
        }

        first_actions, chosen = solver._plan_moves(intents)
        first_by_robot = {action.robot_id: action for action in first_actions}

        self.assertIn(1, first_by_robot)
        self.assertEqual(first_by_robot[1].action, ActionType.MOVE)
        self.assertEqual(first_by_robot[1].target, (15, 10))

        # The rigid footprint move itself must be simulator-legal: robot 1
        # shifts left and its docked pallet occupies robot 1's old center cell.
        solver.simulator.step(first_actions)
        solver._advance_movement_cache(chosen)
        self.assertEqual(world.robots[1].position, (15, 10))
        self.assertEqual(world.pallets[0].position, (16, 10))
        world.validate()

    def test_forced_yield_does_not_require_full_continuation(self):
        problem = ProblemInstance(
            robots=[Robot(0, (5, 5))],
            sku_capacities=[],
            pallets=[],
            orders=[Order(0, [])],
        )
        world = WorldState(problem)
        solver = MultiRobotSolver(
            world,
            robot_ids=[0],
            order_ids=[0],
            max_timesteps=20,
        )

        class FirstStepOnlyScheduler:
            def __init__(self):
                self.calls = 0

            def plan_timed_path(self, *args, **kwargs):
                self.calls += 1
                if self.calls == 1:
                    return [(0, (5, 5)), (1, (4, 5))]
                return []

        scheduler = FirstStepOnlyScheduler()
        trajectory = solver._plan_with_forced_first_move(
            0,
            (10, 5),
            scheduler,
            (4, 5),
        )

        # The forced yield is a one-timestep commitment, not a promise that the
        # robot can already see its complete route after yielding. It must move
        # now and let normal planning continue from the new state next timestep.
        self.assertEqual(trajectory, [(0, (5, 5)), (1, (4, 5))])
        self.assertEqual(scheduler.calls, 2)

    def test_two_robots_solve_first_ten_orders(self):
        world, actions = self._run_prefix([0, 1], 10)

        assigned_robots = {
            world.orders[order_id].assigned_robot
            for order_id in range(10)
        }
        self.assertEqual(assigned_robots, {0, 1})
        self.assertTrue(actions)

    def test_five_robots_solve_first_ten_orders(self):
        world, actions = self._run_prefix([0, 1, 2, 3, 4], 10)

        assigned_robots = {
            world.orders[order_id].assigned_robot
            for order_id in range(10)
        }
        self.assertEqual(assigned_robots, {0, 1, 2, 3, 4})
        self.assertTrue(actions)


if __name__ == "__main__":
    unittest.main()
