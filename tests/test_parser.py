"""Tests for BIG_ORDER.txt parsing."""

from pathlib import Path
import unittest

from src.parser import parse_problem


BIG_ORDER_PATH = (
    Path(__file__).resolve().parents[1]
    / "source_material"
    / "BIG_ORDER.txt"
)


class TestBigOrderParser(unittest.TestCase):
    """Verify that the official challenge file is parsed as expected."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.problem = parse_problem(BIG_ORDER_PATH)

    def test_problem_counts(self) -> None:
        self.assertEqual(len(self.problem.robots), 5)
        self.assertEqual(len(self.problem.sku_capacities), 100)
        self.assertEqual(len(self.problem.pallets), 240)
        self.assertEqual(len(self.problem.orders), 1000)

    def test_robot_starting_positions(self) -> None:
        positions = [robot.position for robot in self.problem.robots]
        self.assertEqual(
            positions,
            [(25, 22), (34, 15), (21, 23), (35, 29), (8, 19)],
        )

    def test_sku_capacities(self) -> None:
        self.assertEqual(
            self.problem.sku_capacities[:5],
            [212, 179, 268, 228, 118],
        )
        self.assertEqual(self.problem.sku_capacities[-1], 185)

    def test_pallet_data(self) -> None:
        first = self.problem.pallets[0]
        self.assertEqual(first.pallet_id, 0)
        self.assertEqual(first.position, (10, 7))
        self.assertEqual(first.original_position, (10, 7))
        self.assertEqual(first.sku, 33)
        self.assertEqual(first.count, self.problem.sku_capacities[33])
        self.assertEqual(first.max_count, self.problem.sku_capacities[33])

        last = self.problem.pallets[-1]
        self.assertEqual(last.pallet_id, 239)
        self.assertEqual(last.position, (46, 32))
        self.assertEqual(last.sku, 82)

    def test_first_order_preserves_sku_sequence(self) -> None:
        first = self.problem.orders[0]
        self.assertEqual(first.order_id, 0)
        self.assertEqual(
            first.skus[:10],
            [2, 19, 89, 5, 99, 33, 0, 4, 24, 74],
        )
        self.assertGreater(first.skus.count(0), 1)


if __name__ == "__main__":
    unittest.main()
