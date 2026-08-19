import unittest

from scheduled_solver.geometry import build_geometry
from scheduled_solver.models import PalletSpec


class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.pallets = tuple(
            PalletSpec(i, home, i, 10)
            for i, home in enumerate(
                [(10, 7), (10, 8), (10, 9), (11, 7), (11, 8), (11, 9)]
            )
        )
        self.geometry = build_geometry(self.pallets, require_24_columns=False)

    def test_rectangular_island_splits_into_two_service_columns(self):
        self.assertEqual(len(self.geometry.columns), 2)
        left, right = self.geometry.columns
        self.assertEqual((left.pallet_x, left.service_x), (10, 9))
        self.assertEqual((right.pallet_x, right.service_x), (11, 12))
        self.assertEqual(left.service_cells, ((9, 7), (9, 8), (9, 9)))

    def test_pallet_homes_are_permanent_static_obstacles(self):
        self.assertIn((10, 7), self.geometry.static_blocked)
        self.assertFalse(self.geometry.pose_is_statically_valid((10, 7), {(0, 0)}))

    def test_carried_pallet_can_exempt_only_its_own_home(self):
        offsets = frozenset({(0, 0), (1, 0)})
        self.assertTrue(
            self.geometry.pose_is_statically_valid(
                (9, 7), offsets, {(1, 0): (10, 7)}
            )
        )
        self.assertFalse(self.geometry.pose_is_statically_valid((9, 7), offsets, {}))

    def test_invalid_nonrectangular_island_is_rejected(self):
        pallets = (
            PalletSpec(0, (10, 7), 0, 1),
            PalletSpec(1, (11, 7), 1, 1),
            PalletSpec(2, (10, 8), 2, 1),
        )
        with self.assertRaises(ValueError):
            build_geometry(pallets, require_24_columns=False)

    def test_actual_shape_requirement_can_be_enforced(self):
        with self.assertRaises(ValueError):
            build_geometry(self.pallets, require_24_columns=True)


if __name__ == "__main__":
    unittest.main()
