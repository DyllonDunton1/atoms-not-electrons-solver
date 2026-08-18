"""Integration tests for the aisle-aware five-robot solver."""

from pathlib import Path
import unittest

from src.aisle_solver import AisleAwareSolver
from src.models import Action, ActionType, Order, Pallet, ProblemInstance, Robot
from src.multi_robot_solver import Intent
from src.parser import parse_problem
from src.simulator import Simulator
from src.world import WorldState


REPO_ROOT = Path(__file__).resolve().parents[1]
BIG_ORDER_PATH = REPO_ROOT / "source_material" / "BIG_ORDER.txt"


class TestAisleAwareSolver(unittest.TestCase):
    def _run_prefix(self, robot_ids, order_count):
        problem = parse_problem(BIG_ORDER_PATH)
        world = WorldState(problem)
        order_ids = list(range(order_count))
        solver = AisleAwareSolver(
            world,
            robot_ids=robot_ids,
            order_ids=order_ids,
            max_timesteps=20_000,
        )

        actions = solver.solve()

        self.assertEqual(solver.assigned_ids, set(order_ids))
        self.assertEqual(solver.completed_ids, set(order_ids))
        self.assertEqual(len(solver.queue), 0)
        self.assertTrue(all(world.orders[i].fulfilled for i in order_ids))
        self.assertEqual(solver.pallet_claims, {})
        world.validate()

        replay_world = WorldState(parse_problem(BIG_ORDER_PATH))
        Simulator(replay_world).run(actions)
        self.assertTrue(all(replay_world.orders[i].fulfilled for i in order_ids))
        replay_world.validate()
        return actions

    def test_one_robot_solves_first_three_orders(self):
        actions = self._run_prefix([0], 3)
        self.assertTrue(actions)

    def test_five_robots_solve_first_ten_orders(self):
        actions = self._run_prefix([0, 1, 2, 3, 4], 10)
        self.assertTrue(actions)

    def test_lower_priority_robot_skips_pallet_beside_higher_priority_robot(self):
        # The local aisle is:
        #
        #     P2  P4  Pnext
        #         R4
        #
        # R4 initially plans P2 -> Pnext. Then R2 arrives beneath P2 and wants
        # P4. R4 must recheck the stored future P2 stop, see higher-priority R2
        # beside it, skip P2 for now, and move to Pnext. That clears P4 for R2.
        pallets = [
            Pallet(0, (5, 4), 0, 1, 1, (5, 4)),
            Pallet(1, (6, 4), 1, 1, 1, (6, 4)),
            Pallet(2, (7, 4), 2, 1, 1, (7, 4)),
        ]
        problem = ProblemInstance(
            robots=[Robot(2, (4, 5)), Robot(4, (6, 5))],
            sku_capacities=[1, 1, 1],
            pallets=pallets,
            orders=[Order(0, [1]), Order(1, [0, 2])],
        )
        world = WorldState(problem)
        solver = AisleAwareSolver(
            world,
            robot_ids=[2, 4],
            order_ids=[0, 1],
            max_timesteps=100,
        )
        solver._assign_free_robots()

        # R4 plans before R2 is adjacent to P2. The deterministic stored plan
        # starts with P2 and then Pnext.
        self.assertTrue(solver._select_new_aisle(4))
        self.assertEqual(
            [stop.pallet_id for stop in solver.states[4].aisle_plan.stops],
            [0, 2],
        )

        # R2 moves beneath P2. This is the state where both robots could
        # otherwise choose each other's neighboring pallet positions.
        solver.simulator.step(
            [Action(world.timestep, 2, ActionType.MOVE, (5, 5))]
        )
        self.assertIn(0, solver._priority_adjacent_pallet_ids(4))

        # Higher-priority R2 is still allowed to select P4 even though R4 is
        # standing on P4's best pickup cell.
        self.assertTrue(solver._select_new_aisle(2))
        self.assertTrue(solver._activate_current_stop(2))
        self.assertEqual(solver.states[2].pallet_id, 1)
        self.assertEqual(solver.states[2].pickup, (6, 5))

        first_intents = {
            2: solver._intent(2),
            4: solver._intent(4),
        }
        self.assertEqual(first_intents[2].move_goal, (6, 5))

        # Activating R4's stale stored P2 stop triggers a live adjacency check.
        # R4 replans the aisle, skips P2, activates Pnext, and moves right.
        self.assertEqual(solver.states[4].pallet_id, 2)
        self.assertEqual(first_intents[4].move_goal, (7, 5))

        first_moves = solver._plan_moves(first_intents)
        first_by_robot = {action.robot_id: action for action in first_moves}
        self.assertNotIn(2, first_by_robot)
        self.assertEqual(first_by_robot[4].target, (7, 5))
        solver.simulator.step(first_moves)

        # R4 has now cleared R2's desired pickup cell. On the next timestep R2
        # enters it while R4 picks Pnext.
        second_intents = {
            2: solver._intent(2),
            4: solver._intent(4),
        }
        self.assertEqual(second_intents[4].action, ActionType.PICK)
        self.assertEqual(second_intents[4].target, (7, 4))

        second_moves = solver._plan_moves(second_intents)
        second_by_robot = {action.robot_id: action for action in second_moves}
        self.assertEqual(second_by_robot[2].target, (6, 5))

        second_fixed = [
            Action(world.timestep, 4, ActionType.PICK, (7, 4)),
        ]
        solver.simulator.step(second_moves + second_fixed)
        solver._post_action(4, second_intents[4])

        # The final same-aisle rescan now sees that R2 moved away from P2, so
        # R4 can come back for the previously skipped requirement.
        state4 = solver.states[4]
        self.assertIsNotNone(state4.aisle_plan)
        self.assertEqual(state4.aisle_stop_index, 0)
        self.assertEqual(state4.aisle_plan.stops[0].pallet_id, 0)
        self.assertEqual(state4.remaining_by_sku, {0: 1})
        world.validate()

    def test_higher_priority_robot_passing_does_not_preempt_active_pick(self):
        pallet = Pallet(0, (6, 4), 1, 3, 3, (6, 4))
        other = Pallet(1, (10, 10), 0, 1, 1, (10, 10))
        problem = ProblemInstance(
            robots=[Robot(2, (5, 3)), Robot(4, (6, 5))],
            sku_capacities=[1, 3],
            pallets=[pallet, other],
            orders=[Order(0, [0]), Order(1, [1, 1])],
        )
        world = WorldState(problem)
        solver = AisleAwareSolver(
            world,
            robot_ids=[2, 4],
            order_ids=[0, 1],
            max_timesteps=100,
        )
        solver._assign_free_robots()

        self.assertTrue(solver._select_new_aisle(4))
        self.assertTrue(solver._activate_current_stop(4))
        self.assertEqual(solver.states[4].pallet_id, 0)
        self.assertEqual(solver.states[4].pickup, (6, 5))
        self.assertEqual(solver.pallet_claims[0], 4)

        first_pick = solver._intent(4)
        self.assertEqual(first_pick.action, ActionType.PICK)
        self.assertEqual(first_pick.target, (6, 4))

        # R2 drives onto the opposite side of P4 while R4 is actively picking.
        # The adjacency rule is only for choosing future stops, so this must not
        # revoke R4's active claim or force it to replan away from P4.
        solver.simulator.step(
            [
                Action(world.timestep, 2, ActionType.MOVE, (6, 3)),
                Action(world.timestep, 4, ActionType.PICK, (6, 4)),
            ]
        )
        solver._post_action(4, first_pick)

        self.assertIn(0, solver._priority_adjacent_pallet_ids(4))
        self.assertNotIn(0, solver._unavailable_pallet_ids(4))
        self.assertEqual(solver.pallet_claims[0], 4)
        self.assertEqual(solver.states[4].pallet_id, 0)

        next_pick = solver._intent(4)
        self.assertEqual(next_pick.action, ActionType.PICK)
        self.assertEqual(next_pick.target, (6, 4))
        world.validate()

    def test_final_aisle_rescan_adds_newly_released_pallet(self):
        pallets = [
            Pallet(0, (6, 5), 0, 10, 10, (6, 5)),
            Pallet(1, (6, 6), 1, 10, 10, (6, 6)),
        ]
        problem = ProblemInstance(
            robots=[Robot(0, (5, 5)), Robot(1, (9, 9))],
            sku_capacities=[10, 10],
            pallets=pallets,
            orders=[Order(0, [0, 1])],
        )
        world = WorldState(problem)
        solver = AisleAwareSolver(
            world,
            robot_ids=[0],
            order_ids=[0],
            max_timesteps=100,
        )
        solver._assign_free_robots()

        # Pallet 1 is busy when robot 0 first plans this aisle, so the first
        # plan can only include SKU 0.
        solver.pallet_claims[1] = 1
        self.assertTrue(solver._select_new_aisle(0))
        state = solver.states[0]
        aisle_id = state.active_aisle_id
        self.assertEqual([stop.sku for stop in state.aisle_plan.stops], [0])
        self.assertTrue(solver._activate_current_stop(0))

        # The competing robot releases pallet 1 before robot 0 finishes its
        # final planned stop. Completing SKU 0 must rescan the same aisle and
        # extend the plan with SKU 1 instead of releasing the aisle.
        del solver.pallet_claims[1]
        solver._post_action(0, Intent(ActionType.PICK, (6, 5)))

        state = solver.states[0]
        self.assertEqual(state.active_aisle_id, aisle_id)
        self.assertIsNotNone(state.aisle_plan)
        self.assertEqual(state.aisle_stop_index, 0)
        self.assertEqual([stop.sku for stop in state.aisle_plan.stops], [1])
        self.assertEqual(state.remaining_by_sku, {1: 1})

    def test_refill_returns_to_same_aisle_and_finishes_stop(self):
        pallet = Pallet(
            pallet_id=0,
            position=(6, 5),
            sku=0,
            count=1,
            max_count=2,
            original_position=(6, 5),
        )
        order = Order(order_id=0, skus=[0, 0, 0])
        problem = ProblemInstance(
            robots=[Robot(0, (5, 5))],
            sku_capacities=[2],
            pallets=[pallet],
            orders=[order],
        )
        world = WorldState(problem)
        solver = AisleAwareSolver(
            world,
            robot_ids=[0],
            order_ids=[0],
            max_timesteps=500,
        )

        actions = solver.solve()

        self.assertTrue(world.orders[0].fulfilled)
        self.assertEqual(
            sum(action.action == ActionType.DOCK for action in actions),
            1,
        )
        self.assertEqual(
            sum(action.action == ActionType.UNDOCK for action in actions),
            1,
        )
        self.assertEqual(
            sum(action.action == ActionType.PICK for action in actions),
            3,
        )

        replay_world = WorldState(
            ProblemInstance(
                robots=[Robot(0, (5, 5))],
                sku_capacities=[2],
                pallets=[
                    Pallet(
                        pallet_id=0,
                        position=(6, 5),
                        sku=0,
                        count=1,
                        max_count=2,
                        original_position=(6, 5),
                    )
                ],
                orders=[Order(order_id=0, skus=[0, 0, 0])],
            )
        )
        Simulator(replay_world).run(actions)
        self.assertTrue(replay_world.orders[0].fulfilled)


if __name__ == "__main__":
    unittest.main()
