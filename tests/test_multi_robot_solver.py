"""Integration and traffic tests for the concurrent FIFO solver."""

from pathlib import Path
import threading
import unittest

from src.models import Action, ActionType, Order, Pallet, ProblemInstance, Robot
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

        replay_world = WorldState(parse_problem(BIG_ORDER_PATH))
        Simulator(replay_world).run(actions)
        self.assertTrue(all(replay_world.orders[i].fulfilled for i in order_ids))
        self.assertTrue(
            all(pallet.count >= 0 for pallet in replay_world.pallets.values())
        )
        replay_world.validate()

        return world, actions

    @staticmethod
    def _dummy_solver(robots, pallets=None, capacities=None):
        problem = ProblemInstance(
            robots=robots,
            sku_capacities=capacities or [],
            pallets=pallets or [],
            orders=[Order(0, []), Order(1, [])],
        )
        world = WorldState(problem)
        solver = MultiRobotSolver(
            world,
            robot_ids=[robot.robot_id for robot in robots],
            order_ids=[0, 1],
            max_timesteps=50,
        )
        return world, solver

    def test_lower_id_waits_while_higher_id_detours(self):
        world, solver = self._dummy_solver(
            [Robot(2, (5, 6)), Robot(4, (5, 5))]
        )
        intents = {
            2: Intent(move_goal=(5, 4)),
            4: Intent(move_goal=(5, 7)),
        }

        preferred = solver._preferred_path(2, (5, 4), {})
        self.assertEqual(preferred[:3], [(5, 6), (5, 5), (5, 4)])

        first_actions = solver._plan_moves(intents)
        first_by_robot = {action.robot_id: action for action in first_actions}

        # R2 does not reroute around higher-ID R4. Its desired next cell is
        # physically occupied, so it waits. R4 sees lower-ID R2 as a static
        # obstacle and immediately starts a spatial detour.
        self.assertNotIn(2, first_by_robot)
        self.assertIn(4, first_by_robot)
        self.assertIn(first_by_robot[4].target, {(4, 5), (6, 5)})

        solver.simulator.step(first_actions)
        second_actions = solver._plan_moves(intents)
        second_by_robot = {action.robot_id: action for action in second_actions}
        self.assertIn(2, second_by_robot)
        self.assertEqual(second_by_robot[2].target, (5, 5))
        world.validate()

    def test_exact_goal_swap_requests_one_step_yield(self):
        world, solver = self._dummy_solver(
            [Robot(2, (5, 6)), Robot(4, (5, 5))]
        )
        intents = {
            2: Intent(move_goal=(5, 5)),
            4: Intent(move_goal=(5, 6)),
        }

        first_actions = solver._plan_moves(intents)
        first_by_robot = {action.robot_id: action for action in first_actions}

        # R2 wants R4's current cell, so R2 waits. R4 cannot produce a full
        # path to its own goal because that goal is R2's current cell, but the
        # one-step priority request still makes R4 clear sideways/up rather
        # than letting both robots wait forever.
        self.assertNotIn(2, first_by_robot)
        self.assertIn(4, first_by_robot)
        self.assertNotEqual(first_by_robot[4].target, (5, 6))
        self.assertNotEqual(first_by_robot[4].target, (5, 5))

        solver.simulator.step(first_actions)
        second_actions = solver._plan_moves(intents)
        second_by_robot = {action.robot_id: action for action in second_actions}
        self.assertEqual(second_by_robot[2].target, (5, 5))
        world.validate()

    def test_replenishment_rigid_robots_clear_without_future_prediction(self):
        pallets = [
            Pallet(0, (37, 37), 0, 1, 1, (37, 37), 2, (1, 0)),
            Pallet(1, (37, 39), 0, 1, 1, (37, 39), 4, (1, 0)),
        ]
        world, solver = self._dummy_solver(
            [
                Robot(2, (36, 37), docked_pallets=[0]),
                Robot(4, (36, 39), docked_pallets=[1]),
            ],
            pallets=pallets,
            capacities=[1],
        )
        intents = {
            2: Intent(move_goal=(36, 39)),
            4: Intent(move_goal=(36, 35)),
        }

        first_actions = solver._plan_moves(intents)
        first_by_robot = {action.robot_id: action for action in first_actions}
        self.assertEqual(first_by_robot[2].target, (36, 38))
        self.assertEqual(first_by_robot[4].target, (35, 39))
        solver.simulator.step(first_actions)

        second_actions = solver._plan_moves(intents)
        second_by_robot = {action.robot_id: action for action in second_actions}
        self.assertNotIn(2, second_by_robot)
        self.assertEqual(second_by_robot[4].target, (34, 39))
        solver.simulator.step(second_actions)

        third_actions = solver._plan_moves(intents)
        third_by_robot = {action.robot_id: action for action in third_actions}
        self.assertEqual(third_by_robot[2].target, (36, 39))

        solver.simulator.step(third_actions)
        self.assertEqual(world.robots[2].position, (36, 39))
        world.validate()

    def test_south_docked_higher_robot_does_not_distort_lower_route(self):
        wall = [
            Pallet(pallet_id, (pallet_id, 4), 0, 1, 1, (pallet_id, 4))
            for pallet_id in range(10)
        ]
        docked = [
            Pallet(20, (3, 6), 0, 1, 1, (3, 6), 2, (0, 1)),
            Pallet(21, (5, 6), 0, 1, 1, (5, 6), 4, (0, 1)),
        ]
        world, solver = self._dummy_solver(
            [
                Robot(2, (3, 5), docked_pallets=[20]),
                Robot(4, (5, 5), docked_pallets=[21]),
            ],
            pallets=wall + docked,
            capacities=[1],
        )

        # R2's preferred route goes straight through the complete higher-ID R4
        # assembly, including R4's south-docked pallet. Priority affects route
        # choice, not physical execution legality.
        lower_path = solver._preferred_path(2, (7, 5), {})
        self.assertEqual(
            lower_path[:5],
            [(3, 5), (4, 5), (5, 5), (6, 5), (7, 5)],
        )

        # Once R2 has committed its first step, R4 must plan around both R2's
        # current and destination rigid footprints. The pallet wall removes the
        # upper escape, so the first legal detour is to the right.
        higher_path = solver._preferred_path(4, (1, 5), {2: (4, 5)})
        self.assertTrue(higher_path)
        self.assertEqual(higher_path[1], (6, 5))

        actions = solver._plan_moves(
            {
                2: Intent(move_goal=(7, 5)),
                4: Intent(move_goal=(1, 5)),
            }
        )
        by_robot = {action.robot_id: action for action in actions}
        self.assertEqual(by_robot[2].target, (4, 5))
        self.assertEqual(by_robot[4].target, (6, 5))
        solver.simulator.step(actions)
        world.validate()

    def test_lower_id_waits_while_higher_id_is_picking(self):
        pallet = Pallet(0, (5, 4), 0, 3, 3, (5, 4))
        world, solver = self._dummy_solver(
            [Robot(2, (4, 5)), Robot(4, (5, 5))],
            pallets=[pallet],
            capacities=[3],
        )

        for _ in range(3):
            intents = {
                2: Intent(move_goal=(7, 5)),
                4: Intent(ActionType.PICK, (5, 4)),
            }
            move_actions = solver._plan_moves(intents)
            self.assertNotIn(2, {action.robot_id for action in move_actions})
            fixed_pick = Action(
                world.timestep,
                4,
                ActionType.PICK,
                (5, 4),
            )
            solver.simulator.step(move_actions + [fixed_pick])
            self.assertEqual(world.robots[2].position, (4, 5))

        # Once R4 is allowed to move, it clears around lower-ID R2. R2 still
        # waits for that physical timestep, then takes the newly vacated cell.
        moving_intents = {
            2: Intent(move_goal=(7, 5)),
            4: Intent(move_goal=(5, 7)),
        }
        clear_actions = solver._plan_moves(moving_intents)
        clear_by_robot = {action.robot_id: action for action in clear_actions}
        self.assertNotIn(2, clear_by_robot)
        self.assertIn(4, clear_by_robot)
        solver.simulator.step(clear_actions)

        next_actions = solver._plan_moves(moving_intents)
        next_by_robot = {action.robot_id: action for action in next_actions}
        self.assertEqual(next_by_robot[2].target, (5, 5))
        world.validate()

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
