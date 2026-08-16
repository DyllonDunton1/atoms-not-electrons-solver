"""Unit tests for challenge submission file writing."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from src.models import Action, ActionType
from src.writer import write_submission


class TestSubmissionWriter(unittest.TestCase):
    def write_and_read(self, actions):
        """Write actions to a temporary file and return the exact contents."""
        with TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "submission.txt"
            write_submission(actions, output_path)
            return output_path.read_text(encoding="utf-8")

    def test_writes_exact_submission_format(self):
        actions = [
            Action(
                timestep=0,
                robot_id=0,
                action=ActionType.MOVE,
                target=(25, 21),
            ),
            Action(
                timestep=1,
                robot_id=0,
                action=ActionType.PICK,
                target=(24, 21),
            ),
        ]

        self.assertEqual(
            self.write_and_read(actions),
            "0 0 move 25 21\n1 0 pick 24 21\n",
        )

    def test_sorts_by_timestep_then_robot_id(self):
        actions = [
            Action(3, 4, ActionType.MOVE, (10, 10)),
            Action(1, 3, ActionType.PICK, (8, 8)),
            Action(1, 0, ActionType.MOVE, (2, 2)),
            Action(2, 2, ActionType.DOCK, (5, 5)),
            Action(1, 2, ActionType.MOVE, (4, 4)),
        ]

        self.assertEqual(
            self.write_and_read(actions),
            "1 0 move 2 2\n"
            "1 2 move 4 4\n"
            "1 3 pick 8 8\n"
            "2 2 dock 5 5\n"
            "3 4 move 10 10\n",
        )

    def test_all_action_types_are_written(self):
        actions = [
            Action(0, 0, ActionType.MOVE, (1, 2)),
            Action(1, 0, ActionType.PICK, (2, 2)),
            Action(2, 0, ActionType.DOCK, (2, 2)),
            Action(3, 0, ActionType.UNDOCK, (2, 2)),
            Action(4, 0, ActionType.FULFILL, (0, 0)),
        ]

        self.assertEqual(
            self.write_and_read(actions),
            "0 0 move 1 2\n"
            "1 0 pick 2 2\n"
            "2 0 dock 2 2\n"
            "3 0 undock 2 2\n"
            "4 0 fulfill 0 0\n",
        )

    def test_allows_multiple_robots_at_same_timestep(self):
        actions = [
            Action(7, 4, ActionType.MOVE, (40, 10)),
            Action(7, 0, ActionType.PICK, (10, 7)),
            Action(7, 2, ActionType.MOVE, (21, 22)),
        ]

        self.assertEqual(
            self.write_and_read(actions),
            "7 0 pick 10 7\n"
            "7 2 move 21 22\n"
            "7 4 move 40 10\n",
        )

    def test_rejects_duplicate_robot_action_in_same_timestep(self):
        actions = [
            Action(5, 1, ActionType.MOVE, (10, 10)),
            Action(5, 1, ActionType.PICK, (11, 10)),
        ]

        with TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "submission.txt"
            with self.assertRaisesRegex(
                ValueError,
                "Robot 1 has multiple actions at timestep 5",
            ):
                write_submission(actions, output_path)

            self.assertFalse(output_path.exists())

    def test_empty_action_list_writes_empty_file(self):
        self.assertEqual(self.write_and_read([]), "")


if __name__ == "__main__":
    unittest.main()
