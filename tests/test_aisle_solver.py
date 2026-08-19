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

    def test_persistent_priority_blocker_replans_traveling_robot(self):
        # The local aisle is:
        #
        #     P2  P4  Pnext
        #         R4
        #
        # R4 claims P2 before R2 arrives. One timestep of R2 standing beside
        # P2 is treated as a drive-by and R4 keeps its target. On the second
        # consecutive timestep, P2 is deferred for this greedy pass and R4
        # immediately replans to Pnext while it is still traveling.
        pallets = [
            Pallet(0, (5, 4), 0, 1, 1, (5, 4)),
            Pallet(1, (6, 4), 1, 1, 1, (6, 4)),
            Pallet(2, (7, 4), 2, 1, 1, (7, 4)),
        ]
        problem = ProblemInstance(
            robots=[Robot(2, (5, 6)), Robot(4, (6, 5))],
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

        self.assertTrue(solver._select_new_aisle(4))
        self.assertEqual(
            [stop.pallet_id for stop in solver.states[4].aisle_plan.stops],
            [0, 2],
        )
        self.assertTrue(solver._activate_current_stop(4))
        self.assertEqual(solver.states[4].pallet_id, 0)
        self.assertEqual(solver.states[4].pickup, (5, 5))
        self.assertEqual(solver.pallet_claims[0], 4)

        # R2 arrives beside P2. At the first action-start snapshot this is only
        # a one-timestep adjacency, so the per-timestep greedy replan keeps P2.
        solver.simulator.step(
            [Action(world.timestep, 2, ActionType.MOVE, (5, 5))]
        )
        first_intent = solver._intent(4)
        self.assertNotIn(
            0,
            solver._persistent_priority_blocked_pallet_ids(4),
        )
        self.assertEqual(solver.states[4].pallet_id, 0)
        self.assertEqual(first_intent.move_goal, (5, 5))
        self.assertEqual(solver.states[4].deferred_pallet_ids, set())

        # R2 stays for one more timestep. Now the adjacency streak reaches two,
        # so R4 checks P2 off this greedy pass, releases its claim, and chooses
        # Pnext immediately instead of waiting for the blocked pickup cell.
        solver.simulator.step([])
        second_intent = solver._intent(4)
        self.assertIn(
            0,
            solver._persistent_priority_blocked_pallet_ids(4),
        )
        self.assertIn(0, solver.states[4].deferred_pallet_ids)
        self.assertEqual(solver.states[4].pallet_id, 2)
        self.assertEqual(second_intent.move_goal, (7, 5))
        self.assertNotIn(0, solver.pallet_claims)
        self.assertEqual(solver.pallet_claims[2], 4)

        # R2 leaves while R4 moves to Pnext. R4 services Pnext, then the final
        # same-aisle rescan is allowed to reconsider the deferred P2 because its
        # higher-priority blocker is gone.
        solver.simulator.step(
            [
                Action(world.timestep, 2, ActionType.MOVE, (5, 6)),
                Action(world.timestep, 4, ActionType.MOVE, (7, 5)),
            ]
        )
        third_intent = solver._intent(4)
        self.assertEqual(third_intent.action, ActionType.PICK)
        self.assertEqual(third_intent.target, (7, 4))

        solver.simulator.step(
            [Action(world.timestep, 4, ActionType.PICK, (7, 4))]
        )
        solver._post_action(4, third_intent)

        state4 = solver.states[4]
        self.assertIsNotNone(state4.aisle_plan)
        self.assertEqual(state4.aisle_stop_index, 0)
        self.assertEqual(state4.aisle_plan.stops[0].pallet_id, 0)
        self.assertEqual(state4.remaining_by_sku, {0: 1})
        world.validate()

    def test_persistent_adjacency_uses_pallet_priority_and_full_footprint(self):
        # R3 carries P1 on its east side. Target P0 is adjacent to the carried
        # pallet cell, but not to R3's center. Because a docked pallet gives R3
        # higher traffic priority than pallet-free R1, P0 must become a
        # persistent blocker for R1 after two consecutive snapshots.
        target = Pallet(0, (17, 24), 0, 1, 1, (17, 24))
        carried = Pallet(
            1,
            (16, 24),
            1,
            1,
            1,
            (17, 8),
            docked_to=3,
            docked_offset=(1, 0),
        )
        problem = ProblemInstance(
            robots=[
                Robot(1, (16, 23)),
                Robot(3, (15, 24), docked_pallets=[1]),
            ],
            sku_capacities=[1, 1],
            pallets=[target, carried],
            orders=[Order(0, [0]), Order(1, [1])],
        )
        world = WorldState(problem)
        solver = AisleAwareSolver(
            world,
            robot_ids=[1, 3],
            order_ids=[0, 1],
            max_timesteps=100,
        )

        self.assertTrue(solver._has_higher_priority(3, 1))
        self.assertFalse(solver._has_higher_priority(1, 3))
        self.assertNotIn(target.position, world.adjacent_positions((15, 24)))
        self.assertIn((16, 24), solver._footprint_cells(3))

        solver._refresh_priority_adjacency_streaks()
        self.assertNotIn(0, solver._persistent_priority_blocked_pallet_ids(1))
        solver.simulator.step([])
        solver._refresh_priority_adjacency_streaks()

        self.assertIn(0, solver._persistent_priority_blocked_pallet_ids(1))
        world.validate()

    def test_persistent_adjacency_does_not_affect_aisle_selection(self):
        # Persistent adjacency is a local greedy traversal rule only. Even when
        # P0 has been beside higher-priority R2 for two timesteps, R4's cheap
        # and detailed aisle-selection heuristics must still consider that aisle.
        pallets = [
            Pallet(0, (6, 4), 1, 1, 1, (6, 4)),
            Pallet(1, (10, 10), 0, 1, 1, (10, 10)),
        ]
        problem = ProblemInstance(
            robots=[Robot(2, (6, 5)), Robot(4, (8, 5))],
            sku_capacities=[1, 1],
            pallets=pallets,
            orders=[Order(0, [0]), Order(1, [1])],
        )
        world = WorldState(problem)
        solver = AisleAwareSolver(
            world,
            robot_ids=[2, 4],
            order_ids=[0, 1],
            max_timesteps=100,
        )
        solver._assign_free_robots()

        solver._refresh_priority_adjacency_streaks()
        solver.simulator.step([])
        solver._refresh_priority_adjacency_streaks()
        self.assertIn(
            0,
            solver._persistent_priority_blocked_pallet_ids(4),
        )

        self.assertTrue(solver._select_new_aisle(4))
        self.assertEqual(solver.states[4].aisle_plan.stops[0].pallet_id, 0)

    def test_persistent_higher_priority_robot_does_not_preempt_active_pick(self):
        pallet = Pallet(0, (6, 4), 1, 3, 3, (6, 4))
        other = Pallet(1, (10, 10), 0, 1, 1, (10, 10))
        problem = ProblemInstance(
            robots=[Robot(2, (5, 3)), Robot(4, (6, 5))],
            sku_capacities=[1, 3],
            pallets=[pallet, other],
            orders=[Order(0, [0]), Order(1, [1, 1, 1])],
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

        # R2 moves onto the opposite side of P0 while R4 is already servicing
        # it. The first adjacent timestep must not interrupt the active pick.
        solver.simulator.step(
            [
                Action(world.timestep, 2, ActionType.MOVE, (6, 3)),
                Action(world.timestep, 4, ActionType.PICK, (6, 4)),
            ]
        )
        solver._post_action(4, first_pick)

        second_pick = solver._intent(4)
        self.assertEqual(second_pick.action, ActionType.PICK)
        self.assertEqual(second_pick.target, (6, 4))
        solver.simulator.step(
            [Action(world.timestep, 4, ActionType.PICK, (6, 4))]
        )
        solver._post_action(4, second_pick)

        # R2 has now remained adjacent for two consecutive action-start states,
        # so P0 is persistently blocked for future greedy choices. R4 is already
        # at the pickup cell, however, so active service remains protected.
        third_pick = solver._intent(4)
        self.assertIn(
            0,
            solver._persistent_priority_blocked_pallet_ids(4),
        )
        self.assertEqual(solver.pallet_claims[0], 4)
        self.assertEqual(solver.states[4].pallet_id, 0)
        self.assertEqual(third_pick.action, ActionType.PICK)
        self.assertEqual(third_pick.target, (6, 4))
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
