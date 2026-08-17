"""Tests for schedule-level performance metrics."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from src.metrics import analyze_actions, read_submission
from src.models import Action, ActionType, Order, Pallet, ProblemInstance, Robot


def make_problem() -> ProblemInstance:
    return ProblemInstance(
        robots=[Robot(0, (0, 0)), Robot(1, (5, 5))],
        sku_capacities=[10],
        pallets=[
            Pallet(0, (2, 2), 0, 10, 10, (2, 2)),
            Pallet(1, (2, 3), 0, 10, 10, (2, 3)),
            Pallet(2, (8, 2), 0, 10, 10, (8, 2)),
            Pallet(3, (8, 3), 0, 10, 10, (8, 3)),
        ],
        orders=[Order(0, [0])],
    )


class TestMetrics(unittest.TestCase):
    def test_action_ratios_waits_and_movement_purpose(self) -> None:
        actions = [
            Action(0, 0, ActionType.MOVE, (1, 0)),
            Action(1, 0, ActionType.PICK, (2, 2)),
            Action(2, 0, ActionType.DOCK, (2, 2)),
            Action(3, 0, ActionType.MOVE, (1, 1)),
            Action(4, 0, ActionType.MOVE, (1, 2)),
            Action(5, 0, ActionType.UNDOCK, (2, 2)),
            Action(6, 0, ActionType.MOVE, (1, 0)),
            Action(7, 0, ActionType.FULFILL, (0, 0)),
        ]

        report = analyze_actions(
            actions,
            make_problem(),
            robot_ids=[0, 1],
            end_timestep=8,
        )

        self.assertEqual(report["total_robot_timesteps"], 16)
        self.assertEqual(report["explicit_actions"], 8)
        self.assertEqual(report["wait_timesteps"], 8)
        self.assertEqual(report["movement"]["collection"], 1)
        self.assertEqual(report["movement"]["refill"], 2)
        self.assertEqual(report["movement"]["fulfillment"], 1)
        self.assertEqual(report["orders_completed"], 1)
        self.assertEqual(report["refill_trips"], 1)

        robot_zero = report["robots"][0]
        robot_one = report["robots"][1]
        self.assertEqual(robot_zero["waits"], 0)
        self.assertEqual(robot_one["waits"], 8)
        self.assertEqual(robot_one["terminal_idle"], 8)

    def test_aisle_visits_and_reentries_use_service_sequence(self) -> None:
        actions = [
            Action(0, 0, ActionType.PICK, (2, 2)),
            Action(1, 0, ActionType.MOVE, (5, 2)),
            Action(2, 0, ActionType.PICK, (8, 2)),
            Action(3, 0, ActionType.MOVE, (5, 2)),
            Action(4, 0, ActionType.PICK, (2, 3)),
            Action(5, 0, ActionType.MOVE, (0, 0)),
            Action(6, 0, ActionType.FULFILL, (0, 0)),
        ]

        report = analyze_actions(
            actions,
            make_problem(),
            robot_ids=[0],
            end_timestep=7,
        )

        self.assertEqual(report["aisle_count"], 2)
        self.assertEqual(report["aisle_visits"], 3)
        self.assertEqual(report["aisle_reentries"], 1)
        self.assertEqual(report["movement"]["collection"], 2)
        self.assertEqual(report["movement"]["fulfillment"], 1)

    def test_read_submission_parses_writer_format(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "schedule.txt"
            path.write_text(
                "0 0 move 1 0\n1 0 pick 2 2\n",
                encoding="utf-8",
            )
            actions = read_submission(path)

        self.assertEqual(
            actions,
            [
                Action(0, 0, ActionType.MOVE, (1, 0)),
                Action(1, 0, ActionType.PICK, (2, 2)),
            ],
        )


if __name__ == "__main__":
    unittest.main()
