"""Time-aware committed pallet inventory."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Sequence

from .models import InventoryEvent, PalletSpec


class InventoryTimeline:
    def __init__(self, pallets: Sequence[PalletSpec]) -> None:
        self._capacity = {pallet.pallet_id: pallet.max_count for pallet in pallets}
        self._base_stock = dict(self._capacity)
        self._base_timestep = 0
        self._events: Dict[int, List[InventoryEvent]] = defaultdict(list)

    @staticmethod
    def _event_key(event: InventoryEvent) -> tuple:
        return (event.timestep, 0 if event.kind == "pick" else 1)

    def _apply_event(self, pallet_id: int, stock: int, event: InventoryEvent) -> int:
        if event.kind == "pick":
            return stock - event.amount
        if event.kind == "refill":
            return self._capacity[pallet_id]
        raise ValueError(f"Unknown inventory event {event.kind!r}")

    def _merged_events(
        self,
        pallet_id: int,
        additional_events: Iterable[InventoryEvent] = (),
    ) -> List[InventoryEvent]:
        additional = [
            event for event in additional_events if event.pallet_id == pallet_id
        ]
        if any(event.timestep < self._base_timestep for event in additional):
            raise ValueError(
                "Inventory event precedes the compacted scheduling frontier"
            )
        if not additional:
            return list(self._events.get(pallet_id, ()))
        events = list(self._events.get(pallet_id, ())) + additional
        events.sort(key=self._event_key)
        return events

    def stock_at(
        self,
        pallet_id: int,
        timestep: int,
        local_events: Iterable[InventoryEvent] = (),
    ) -> int:
        if pallet_id not in self._capacity:
            raise KeyError(pallet_id)
        if timestep < self._base_timestep:
            raise ValueError(
                "Cannot query inventory before the compacted scheduling frontier"
            )
        stock = self._base_stock[pallet_id]
        for event in self._merged_events(pallet_id, local_events):
            if event.timestep >= timestep:
                break
            stock = self._apply_event(pallet_id, stock, event)
            if stock < 0:
                raise ValueError(f"Pallet {pallet_id} stock became negative")
        return stock

    def _pallet_events_are_feasible(
        self,
        pallet_id: int,
        additional_events: Iterable[InventoryEvent],
    ) -> bool:
        stock = self._base_stock[pallet_id]
        for event in self._merged_events(pallet_id, additional_events):
            stock = self._apply_event(pallet_id, stock, event)
            if stock < 0:
                return False
        return True

    def events_are_feasible(self, additional_events: Iterable[InventoryEvent]) -> bool:
        additional = list(additional_events)
        affected = {event.pallet_id for event in additional}
        unknown = affected - self._capacity.keys()
        if unknown:
            raise KeyError(min(unknown))
        by_pallet: Dict[int, List[InventoryEvent]] = defaultdict(list)
        for event in additional:
            by_pallet[event.pallet_id].append(event)
        return all(
            self._pallet_events_are_feasible(pallet_id, by_pallet[pallet_id])
            for pallet_id in affected
        )

    def pick_is_feasible(
        self,
        pallet_id: int,
        timestep: int,
        amount: int,
        robot_id: int,
        local_events: Iterable[InventoryEvent] = (),
    ) -> bool:
        """Return whether a new pick preserves every earlier committed promise.

        Only this pallet's timeline can be changed by the proposed pick, so do
        not revalidate unrelated pallet histories from the current beam state.
        Committed refill events still naturally separate inventory epochs.
        """
        if pallet_id not in self._capacity:
            raise KeyError(pallet_id)
        if amount <= 0:
            raise ValueError("pick amount must be positive")
        candidate = InventoryEvent(timestep, pallet_id, "pick", amount, robot_id)
        local_for_pallet = [
            event for event in local_events if event.pallet_id == pallet_id
        ]
        local_for_pallet.append(candidate)
        return self._pallet_events_are_feasible(pallet_id, local_for_pallet)

    def compact_before(self, frontier_timestep: int) -> int:
        """Fold committed history before a monotonic planning frontier into stock.

        Future orders can never begin before the scheduler's current minimum
        robot-availability time.  Events strictly before that frontier have
        already affected every possible future stock query, so retain only the
        resulting stock plus events at or after the frontier.

        Returns the number of committed events discarded.
        """
        if frontier_timestep < self._base_timestep:
            raise ValueError("Inventory compaction frontier moved backwards")
        if frontier_timestep == self._base_timestep:
            return 0

        removed = 0
        for pallet_id in self._capacity:
            stock = self._base_stock[pallet_id]
            retained: List[InventoryEvent] = []
            for event in self._events.get(pallet_id, ()):
                if event.timestep < frontier_timestep:
                    stock = self._apply_event(pallet_id, stock, event)
                    if stock < 0:
                        raise ValueError(f"Pallet {pallet_id} stock became negative")
                    removed += 1
                else:
                    retained.append(event)
            self._base_stock[pallet_id] = stock
            if retained:
                self._events[pallet_id] = retained
            else:
                self._events.pop(pallet_id, None)

        self._base_timestep = frontier_timestep
        return removed

    def commit(self, events: Iterable[InventoryEvent]) -> None:
        event_list = list(events)
        if not self.events_are_feasible(event_list):
            raise ValueError("Inventory events would invalidate committed stock timeline")
        for event in event_list:
            if event.pallet_id not in self._capacity:
                raise KeyError(event.pallet_id)
            if event.timestep < self._base_timestep:
                raise ValueError(
                    "Inventory event precedes the compacted scheduling frontier"
                )
            if event.kind not in {"pick", "refill"}:
                raise ValueError(f"Unknown inventory event {event.kind!r}")
            self._events[event.pallet_id].append(event)
            self._events[event.pallet_id].sort(key=self._event_key)

    def events_for(self, pallet_id: int) -> tuple:
        return tuple(self._events.get(pallet_id, ()))
