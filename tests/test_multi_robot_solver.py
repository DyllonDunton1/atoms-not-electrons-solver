"""Integration tests for the concurrent FIFO baseline solver."""

from pathlib import Path
import threading
import unittest

from src.multi_robot_solver import MultiRobotSolver
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
