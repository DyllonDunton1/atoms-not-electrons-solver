from pathlib import Path
import ast
import unittest

from scheduled_solver.geometry import build_geometry
from scheduled_solver.parser import parse_problem


REPO_ROOT = Path(__file__).resolve().parents[2]
BIG_ORDER = REPO_ROOT / "source_material" / "BIG_ORDER.txt"


class ArchitectureTests(unittest.TestCase):
    def test_big_order_builds_exactly_twenty_four_ten_pallet_columns(self):
        problem = parse_problem(BIG_ORDER)
        geometry = build_geometry(problem.pallets, require_24_columns=True)
        self.assertEqual(len(problem.pallets), 240)
        self.assertEqual(len(geometry.columns), 24)
        self.assertTrue(all(len(column.pallet_ids) == 10 for column in geometry.columns))
        self.assertEqual(len(geometry.pallet_to_column), 240)

    def test_every_big_order_pallet_home_is_permanently_blocked(self):
        problem = parse_problem(BIG_ORDER)
        geometry = build_geometry(problem.pallets, require_24_columns=True)
        self.assertEqual(
            geometry.static_blocked,
            frozenset(pallet.home for pallet in problem.pallets),
        )

    def test_scheduled_package_has_no_src_imports(self):
        package_dir = REPO_ROOT / "scheduled_solver"
        violations = []
        for path in package_dir.glob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name == "src" or alias.name.startswith("src."):
                            violations.append((path.name, alias.name))
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    if module == "src" or module.startswith("src."):
                        violations.append((path.name, module))
        self.assertEqual(violations, [])


if __name__ == "__main__":
    unittest.main()
