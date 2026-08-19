import unittest

from scheduled_solver.inventory import InventoryTimeline
from scheduled_solver.models import InventoryEvent, PalletSpec


class InventoryTests(unittest.TestCase):
    def setUp(self):
        self.timeline = InventoryTimeline([PalletSpec(0, (10, 10), 2, 5)])

    def test_pick_applies_after_action_timestep(self):
        self.timeline.commit([InventoryEvent(3, 0, "pick", 1, 0)])
        self.assertEqual(self.timeline.stock_at(0, 3), 5)
        self.assertEqual(self.timeline.stock_at(0, 4), 4)

    def test_refill_resets_capacity(self):
        self.timeline.commit(
            [
                InventoryEvent(1, 0, "pick", 1, 0),
                InventoryEvent(2, 0, "pick", 1, 0),
                InventoryEvent(5, 0, "refill", 0, 0),
            ]
        )
        self.assertEqual(self.timeline.stock_at(0, 5), 3)
        self.assertEqual(self.timeline.stock_at(0, 6), 5)

    def test_local_events_are_visible_without_commit(self):
        local = [InventoryEvent(1, 0, "pick", 2, 1)]
        self.assertEqual(self.timeline.stock_at(0, 2, local), 3)
        self.assertEqual(self.timeline.stock_at(0, 2), 5)

    def test_earlier_new_pick_cannot_break_future_committed_pick(self):
        self.timeline.commit([InventoryEvent(10, 0, "pick", 3, 0)])
        candidate = [InventoryEvent(2, 0, "pick", 3, 1)]
        self.assertFalse(self.timeline.events_are_feasible(candidate))
        with self.assertRaises(ValueError):
            self.timeline.commit(candidate)

    def test_earlier_refill_can_preserve_future_stock(self):
        self.timeline.commit([InventoryEvent(10, 0, "pick", 5, 0)])
        candidate = [
            InventoryEvent(2, 0, "pick", 5, 1),
            InventoryEvent(5, 0, "refill", 0, 1),
        ]
        self.assertTrue(self.timeline.events_are_feasible(candidate))

    def test_negative_stock_is_rejected(self):
        events = [InventoryEvent(i, 0, "pick", 1, 0) for i in range(6)]
        self.assertFalse(self.timeline.events_are_feasible(events))
        with self.assertRaises(ValueError):
            self.timeline.commit(events)


if __name__ == "__main__":
    unittest.main()
