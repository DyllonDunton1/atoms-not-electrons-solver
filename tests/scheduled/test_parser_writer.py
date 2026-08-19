from pathlib import Path
import tempfile
import unittest

from scheduled_solver.models import Action, ActionType
from scheduled_solver.parser import parse_problem
from scheduled_solver.writer import write_submission


class ParserWriterTests(unittest.TestCase):
    def test_parse_problem(self):
        text = """\
2
1 2
3 4
2
5
7
2
10 10 0
11 10 1
2
0 1
1
"""
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "problem.txt"
            path.write_text(text)
            problem = parse_problem(path)
        self.assertEqual([r.start for r in problem.robots], [(1, 2), (3, 4)])
        self.assertEqual(problem.pallets[0].max_count, 5)
        self.assertEqual(problem.orders[0].skus, (0, 1))

    def test_parser_rejects_invalid_sku(self):
        text = """\
1
1 2
1
5
1
10 10 1
1
0
"""
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "problem.txt"
            path.write_text(text)
            with self.assertRaises(ValueError):
                parse_problem(path)

    def test_writer_sorts_and_formats(self):
        actions = [
            Action(2, 1, ActionType.PICK, (5, 5)),
            Action(1, 0, ActionType.MOVE, (3, 4)),
        ]
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "out.txt"
            write_submission(actions, path)
            self.assertEqual(
                path.read_text(),
                "1 0 move 3 4\n2 1 pick 5 5\n",
            )

    def test_writer_rejects_duplicate_robot_timestep(self):
        actions = [
            Action(1, 0, ActionType.MOVE, (3, 4)),
            Action(1, 0, ActionType.PICK, (4, 4)),
        ]
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(ValueError):
                write_submission(actions, Path(temp) / "out.txt")


if __name__ == "__main__":
    unittest.main()
